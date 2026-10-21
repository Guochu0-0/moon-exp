"""开销测试（#189「顺带测开销」）：按计划中的细化阶段训练设置跑若干训练步，记每步耗时与峰值显存。不存模型、不评测。

    python runs/E18/code/cost.py [--chains 16 8 4] [--steps 25] [--warmup 5]

设置同 E15（runs/E15/configs/_rl.toml + loo_t03）：训练集、难版扰动、DINOv2 qkv LoRA r 8、lr 3e-5、batch 1；另外解冻 VGG
（训练信号要到 VGG 与各级细化层）。每对图 N 条链，每条链是一次完整的「粗级按 softmax(gm_cls / 0.3) 每格抽锚点 →
尺度 16 细化（不加噪）→ 尺度 8、4、2、1 每级输出加高斯（σ = 0.5·s 原网格 px）」。
奖励：先正常推理一次（粗级 mode、各级不加噪）拟合参照仿射 A_ref（5000 点、RANSAC 3 px）。粗级照 E15 逐格打分
（δ 8、τ 16）；每级输出逐像素与 A_ref 映射的位置相距 d，d ≤ 2s 得 1 − ½(d/s)²，否则 −0.1。各级（含粗级）各自
RLOO（减同一位置其余 N − 1 条链的均值）。log π：粗级同 E15；细化每级为高斯 log 密度 −‖a − μ‖² / 2σ²，
μ = 这一级的输入 warp + 细化层输出的位移（带梯度），a 为抽到的输出。loss = 粗级项 + 四级之和，各项为 −mean(优势 · log π)。

实现：DINOv2 与 VGG 的特征每对只算一次（带梯度）。第一遍不求梯度跑完 N 条链，记下各级输入、抽到的输出与奖励；
第二遍逐级、每 chunk 条链一组，带梯度重算这一级的 μ 并立刻反向，梯度先累积到特征上，最后一次反向穿过编码器。
每步耗时含取数据之后的全部计算（torch.cuda.synchronize 计时）；峰值显存为 torch.cuda.max_memory_allocated，
每个 N 开始前清零。产物：runs/E18/extra/data/cost.json、cost_steps.csv。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from finetune import config as C                    # noqa: E402
from finetune.data import PairSet                   # noqa: E402
from finetune.geom import HW                        # noqa: E402
from finetune.models import roma                    # noqa: E402
from finetune.optim import Optim                    # noqa: E402
from finetune.parts import aug, rloo                # noqa: E402
from finetune.parts.pseudo import affine_norm, grid  # noqa: E402

CFG = {"model": {"name": "roma", "train_vgg": True, "lora_r": 8},
       "optim": {"lr": 3e-5, "steps": 4000},
       "run": {"seed": 0},
       "aug": {"shift": 128.0, "shift_disk": True, "rot": 20.0, "scale": 0.2, "photo": False, "erase": True},
       "rloo": {"reward": "ref", "baseline": "loo", "temp": 0.3}}
LEVELS = ("8", "4", "2", "1")
PX = HW / 2


def disp(dec, s, d_flow, h, w):
    return int(s) * torch.stack((d_flow[:, 0].float() / (dec.refine_init * w),
                                 d_flow[:, 1].float() / (dec.refine_init * h)), dim=1)


def ref_reward(a, A, s):
    """a (n,2,h,w) 这一级的输出；A 参照仿射 → 逐像素奖励 (n,h,w)。A 为 None 时全 0。"""
    n, _, h, w = a.shape
    if A is None:
        return torch.zeros(n, h, w, device=a.device)
    An = torch.from_numpy(affine_norm(A)).to(a.device).float()
    pred = torch.einsum("ij,hwj->hwi", An[:, :2], grid(h, w, a.device)) + An[:, 2]
    d = (a.permute(0, 2, 3, 1).float() - pred).norm(dim=-1) * PX
    return torch.where(d <= 2 * s, 1 - 0.5 * (d / s) ** 2, torch.full_like(d, -0.1))


@torch.no_grad()
def rollout_noisy(model, f1, f2, P, cert, k, sig, chunk):
    """N 条链：粗 warp → 尺度 16 细化 → 尺度 8…1 每级输出加高斯。返回 {s: (输入 warp, 输入 certainty, 抽到的输出)}，
    每项 (N,…)。"""
    dec = model.model.decoder
    H, W = f1[1].shape[-2:]
    rec = {s: ([], [], []) for s in LEVELS}
    for i in range(0, len(k), chunk):
        kk = k[i:i + chunk]
        n = len(kk)
        flow = rloo.anchor_flow(P.expand(n, -1, -1, -1), kk)
        ce = cert.expand(n, -1, -1, -1)
        for s in rloo.SCALES:
            a, b = f1[int(s)].expand(n, -1, -1, -1), f2[int(s)].expand(n, -1, -1, -1)
            d_flow, d_cert = dec.conv_refiner[s](a, b, flow, scale_factor=1, logits=ce)
            mu = flow + disp(dec, s, d_flow, H, W)
            if s in LEVELS:
                x = mu + sig[s] * torch.randn_like(mu)
                for lst, v in zip(rec[s], (flow, ce, x)):
                    lst.append(v)
                mu = x
            flow, ce = mu, ce + d_cert
            if s != "1":
                size = f1[int(s) // 2].shape[-2:]
                flow = F.interpolate(flow, size=size, mode=dec.flow_upsample_mode)
                ce = F.interpolate(ce, size=size, mode=dec.flow_upsample_mode)
    return {s: tuple(torch.cat(v) for v in rec[s]) for s in LEVELS}


def train_step(model, optim, batch, N, p, sig, chunk, dev):
    dec = model.model.decoder
    sc = optim.scaler
    i0, i1 = batch["image0"].to(dev), batch["image1"].to(dev)
    tm = {}
    t = time.time()
    f1, f2 = rloo.features(model, i0, i1)
    cls, cert = rloo.coarse(model, f1, f2)
    cls = cls.float()
    C_, h, w = cls.shape[1:]
    logp_all = F.log_softmax(cls / p["temp"], 1)
    with torch.no_grad():
        q = logp_all[0].exp()
        k = torch.multinomial(q.reshape(C_, -1).T, N, replacement=True).T.reshape(-1, h, w)
        P = cls.softmax(1)
        g1 = {s: x.detach() for s, x in f1.items()}
        g2 = {s: x.detach() for s, x in f2.items()}
        ce0 = cert.float().detach()
        torch.cuda.synchronize()
        tm["t_enc"] = time.time() - t
        t = time.time()
        fl, cr = rloo.refine(model, g1, g2, rloo.anchor_flow(P, P.argmax(1)), ce0)
        A_ref = rloo.fit(fl[0], cr[0], p["num"], p["ransac"])[0]
        rec = rollout_noisy(model, g1, g2, P, ce0, k, sig, chunk)
        torch.cuda.synchronize()
        tm["t_roll"] = time.time() - t
        t = time.time()
        r_c, _ = rloo.ref_cells(rloo.anchor_flow(P.expand(N, -1, -1, -1), k), A_ref, p["delta"], p["tau"], p["r_out"])
        adv_c = rloo.loo(r_c)
        adv = {s: rloo.loo(ref_reward(rec[s][2], A_ref, int(s))) for s in LEVELS}
        torch.cuda.synchronize()
        tm["t_rew"] = time.time() - t
    t = time.time()
    leaf1 = {s: f1[int(s)].detach().requires_grad_() for s in LEVELS}
    leaf2 = {s: f2[int(s)].detach().requires_grad_() for s in LEVELS}
    H, W = f1[1].shape[-2:]
    for s in LEVELS:                     # 给定存下的输入，各级互不相关：每级每组单独前向、立刻反向
        for i in range(0, N, chunk):
            n = min(chunk, N - i)
            fin, cin, x = (v[i:i + n] for v in rec[s])
            d_flow, _ = dec.conv_refiner[s](leaf1[s].expand(n, -1, -1, -1), leaf2[s].expand(n, -1, -1, -1), fin,
                                            scale_factor=1, logits=cin)
            mu = fin + disp(dec, s, d_flow, H, W)
            lp = -(x - mu).square().sum(1) / (2 * sig[s] ** 2)                      # (n,h,w)
            sc.scale(-(adv[s][i:i + n] * lp).mean() * n / N).backward()
    loss_c = -(adv_c * rloo._logp(logp_all[0], k)).mean()
    ts = [sc.scale(loss_c)] + [f1[int(s)] for s in LEVELS] + [f2[int(s)] for s in LEVELS]
    gs = [None] + [leaf1[s].grad for s in LEVELS] + [leaf2[s].grad for s in LEVELS]
    torch.autograd.backward(ts, gs)
    torch.cuda.synchronize()
    tm["t_grad"] = time.time() - t
    return tm, A_ref is not None


def grad_reach(model):
    """各模块：可训参数个数、梯度不全为 0 的参数所占比例、梯度是否全部有限。"""
    named = list(model.model.named_parameters())
    groups = {f"refiner_{s}": f"decoder.conv_refiner.{s}." for s in rloo.SCALES}
    groups |= {"vgg": "encoder.cnn.", "embedding_decoder": "decoder.embedding_decoder.", "gp16": "decoder.gps."}
    out = {}
    for g, pre in groups.items():
        out[g] = _reach([p for n, p in named if n.startswith(pre) and p.requires_grad])
    out["dino_lora"] = _reach([q for l in model.lora.values() for q in (l.A, l.B)])
    return out


def _reach(ps):
    if not ps:
        return None
    nz = [p.grad is not None and bool(p.grad.abs().sum() > 0) for p in ps]
    fin = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in ps)
    return {"n": len(ps), "frac_nonzero": round(sum(nz) / len(ps), 3), "finite": fin}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--chains", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--steps", type=int, default=25, help="每个 N 计时的步数")
    ap.add_argument("--warmup", type=int, default=5, help="每个 N 先跑几步不计")
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=8, help="第二遍一次带梯度跑几条链")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    cfg = C.expand(CFG)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    dev = "cuda"
    model = roma.Model(cfg, a.weights_root)
    p = cfg["rloo"]
    ds = PairSet(a.data, "train", model.resize, **model.input, seed=0, **aug.data_kw(cfg["aug"]))
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(0),
                                     num_workers=cfg["run"]["workers"], drop_last=True)
    optim = Optim(model.params, cfg["optim"], model.amp, getattr(model, "groups", None))
    sig = {s: a.k * int(s) / PX for s in LEVELS}
    it = iter(dl)
    rows, summ = [], {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "k": a.k,
                      "chunk": a.chunk, "steps": a.steps, "warmup": a.warmup, "by_chains": {}}
    step = 0
    for N in a.chains:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        secs = []
        for j in range(a.warmup + a.steps):
            batch = next(it)
            torch.cuda.synchronize()
            t0 = time.time()
            tm, ok = train_step(model, optim, batch, N, p, sig, min(a.chunk, N), dev)
            if step == 0:                # 第一步（不计时）：各模块是否收到梯度
                summ["grad_reach"] = grad_reach(model)
            step += 1
            upd = optim.step(step)
            torch.cuda.synchronize()
            sec = time.time() - t0
            if j >= a.warmup:
                secs.append(sec)
            rows.append({"chains": N, "i": j, "timed": j >= a.warmup, "sec": round(sec, 4),
                         **{k_: round(v, 4) for k_, v in tm.items()}, "ref_ok": ok, "gnorm": upd.get("gnorm"), "scale": upd.get("scale")})
            print(f"N={N} {j}: {sec:.2f}s {tm}", flush=True)
        tim = [r for r in rows if r["chains"] == N and r["timed"]]
        summ["by_chains"][str(N)] = {
            "sec_median": round(float(np.median(secs)), 3), "sec_mean": round(float(np.mean(secs)), 3),
            "sec_p10": round(float(np.percentile(secs, 10)), 3), "sec_p90": round(float(np.percentile(secs, 90)), 3),
            **{f"{k_}_median": round(float(np.median([r[k_] for r in tim])), 3)
               for k_ in ("t_enc", "t_roll", "t_rew", "t_grad")},
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}
    out = RUN / "extra" / "data"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cost_steps.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "cost.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
