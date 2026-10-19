"""开跑前检查（「【RoMa·RL】开跑前检查：粗级 softmax 的尖锐程度与梯度路径」#168）：只读 zero-shot AnyMatch-RoMa，不训练。

    python runs/E12/code/probe.py --n 200 [--group 16]

1. 尖锐程度：Train 上均匀取 n 对，按首批训练同款扰动（难版范围几何 + 擦除，不加光度）。每对算粗级 gm_cls 的 softmax，
   逐格统计熵、mode 概率；按温度 1 每格抽 group 份锚点（finetune/parts/rloo.py 的策略），统计与 mode 不同的比例、所抽锚点离 mode 的距离；
   把每份样本照推理细化到尺度 1，统计最终 warp 与 mode 那份的差、各份 RANSAC 仿射之间的差、r1 与 ncc 两种 reward 在组内的离散程度。
   格分两类：mode 那份尺度 1 certainty（sigmoid）在格内的均值 > 0.5 的为「可信格」，其余为「其余格」。
2. 一致性：rollout 取 k = mode 时，应与 romatch 原前向的尺度 1 warp 相同（前 5 对，报最大差）。
3. 梯度路径：对 2 对，分别反传 (a) 粗级 log π（rloo 的策略）与 (b) 各尺度 s 的 warp 上固定 σ 高斯的 log π，
   报各模块组的梯度范数，看哪些能到 DINOv2 的 LoRA（LoRA 的 B 初值为 0，所以 A 的梯度恒为 0，看 B）。

产物：runs/E12/extra/data/cells.csv（逐对汇总）、summary.json。
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

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from finetune import config as C                    # noqa: E402
from finetune.data import PairSet                   # noqa: E402
from finetune.geom import HW                        # noqa: E402
from finetune.models import roma                    # noqa: E402
from finetune.parts import aug, rloo                # noqa: E402

CFG = {"model": {"name": "roma", "train_vgg": True, "lora_r": 8},
       "optim": {"lr": 3e-5},
       "aug": {"shift": 128.0, "shift_disk": True, "rot": 20.0, "scale": 0.2, "photo": False, "erase": True},
       "rloo": {}}
GROUPS = {"lora": lambda n: "LoRA" in n, "emb_decoder": lambda n: n.startswith("decoder.embedding_decoder"),
          "gp16": lambda n: n.startswith("decoder.gps"), "proj16": lambda n: n.startswith("decoder.proj.16"),
          "proj8-1": lambda n: n.startswith("decoder.proj.") and not n.startswith("decoder.proj.16"),
          **{f"refiner{s}": (lambda n, s=s: n.startswith(f"decoder.conv_refiner.{s}.")) for s in rloo.SCALES},
          "vgg": lambda n: n.startswith("encoder.cnn")}


def named_params(model):
    out = [(n, p) for n, p in model.model.named_parameters() if p.requires_grad]
    out += [(f"LoRA.{k}.B", l.B) for k, l in model.lora.items()]
    return out


def grad_norms(model):
    g = {k: 0.0 for k in GROUPS}
    for n, p in named_params(model):
        for k, f in GROUPS.items():
            if f(n) and p.grad is not None:
                g[k] += float(p.grad.float().norm() ** 2)
    return {k: round(v ** 0.5, 6) for k, v in g.items()}


def zero(model):
    for _, p in named_params(model):
        p.grad = None


def pct(x, qs=(10, 50, 90)):
    x = np.asarray(x, np.float64)
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs} | {"mean": round(float(x.mean()), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--group", type=int, default=16)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    cfg = C.expand(CFG)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    model = roma.Model(cfg, a.weights_root)
    p = cfg["rloo"]
    ds = PairSet(a.data, "train", model.resize, **model.input, seed=0, **aug.data_kw(cfg["aug"]))
    ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // a.n)][: a.n]
    dev = "cuda"
    rows, cells = [], {"H": [], "pmode": [], "nonmode": [], "d_anchor": [], "conf": []}
    consist, grads, secs = [], [], []
    for i in range(len(ds)):
        it = ds[i]
        i0, i1 = it["image0"][None].to(dev), it["image1"][None].to(dev)
        t0 = time.time()
        with torch.no_grad():
            f1, f2 = rloo.features(model, i0, i1)
            cls, cert = rloo.coarse(model, f1, f2)
            cls = cls.float()
            C_, h, w = cls.shape[1:]
            q = cls[0].softmax(0)
            logq = cls[0].log_softmax(0)
            mode = q.argmax(0)
            k = torch.multinomial(q.reshape(C_, -1).T, a.group, replacement=True).T.reshape(-1, h, w)
            P = cls.softmax(1)
            flow, ce = rloo.rollout(model, f1, f2, P, cert.float(), torch.cat([mode[None], k]), p["chunk"])
            if i < 5:
                ref = model.forward(i0, i1)[1]["flow"].float()
                consist.append(float((ref[0] - flow[0]).abs().max() * (HW / 2)))
            fits = [rloo.fit(flow[g], ce[g], p["num"], p["ransac"]) for g in range(len(flow))]
            valid = rloo.sar_valid(i1[0])
            r1 = torch.stack([rloo.r1_cells(flow[g], A, p["ransac"], (h, w)) for g, (A, _) in enumerate(fits)])
            nc = np.array([rloo.ncc(i0[0], i1[0], valid, A) for A, _ in fits])
        secs.append(time.time() - t0)
        G_, _ = rloo.anchors(C_, cls.device)
        conf = torch.nn.functional.adaptive_avg_pool2d(ce[0:1].sigmoid(), (h, w))[0, 0] > 0.5
        H = -(q * logq).sum(0)
        nonmode = (k != mode).float().mean(0)
        d_anc = ((G_[k] - G_[mode]).norm(dim=-1) * (HW / 2))                   # (G,h,w)，原网格 px
        d_fin = ((flow[1:] - flow[:1]).norm(dim=1) * (HW / 2))                 # (G,H,W)
        for key, v in (("H", H), ("pmode", q.max(0).values), ("nonmode", nonmode), ("d_anchor", d_anc.mean(0)),
                       ("conf", conf.float())):
            cells[key].append(v.reshape(-1).cpu().numpy())
        A0 = fits[0][0]
        corners = np.array([[0, 0], [HW, 0], [0, HW], [HW, HW]], np.float64)

        def cdist(A, B):
            if A is None or B is None:
                return np.nan
            return float(np.linalg.norm(corners @ A[:, :2].T + A[:, 2] - corners @ B[:, :2].T - B[:, 2], axis=1).mean())

        dA = [cdist(A, A0) for A, _ in fits[1:]]
        rows.append({"pair": it["pair"], "H": float(H.mean()), "pmode": float(q.max(0).values.mean()),
                     "nonmode": float(nonmode.mean()), "nonmode_conf": float(nonmode[conf].mean()) if conf.any() else np.nan,
                     "conf_frac": float(conf.float().mean()),
                     "d_anchor_nonmode": float(d_anc[k != mode].mean()) if (k != mode).any() else 0.0,
                     "d_final_med": float(d_fin.median()), "d_final_gt3": float((d_fin > 3).float().mean()),
                     "dA_med": float(np.nanmedian(dA)) if np.isfinite(dA).any() else np.nan,
                     "n_fail": int(sum(A is None for A, _ in fits[1:])),
                     "r1_mode": float(r1[0].mean()), "r1_mean": float(r1[1:].mean()),
                     "r1_sd_cell": float(r1[1:].std(0).mean()), "r1_sd_pair": float(r1[1:].mean((1, 2)).std()),
                     "ncc_mode": float(nc[0]), "ncc_mean": float(nc[1:].mean()), "ncc_sd": float(nc[1:].std())})
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(ds)} {secs[-1]:.2f}s", flush=True)
    # 梯度路径
    for i in range(2):
        it = ds[i]
        i0, i1 = it["image0"][None].to(dev), it["image1"][None].to(dev)
        zero(model)
        f1, f2 = rloo.features(model, i0, i1)
        cls, _ = rloo.coarse(model, f1, f2)
        logp = cls.float().log_softmax(1)
        with torch.no_grad():
            k = torch.multinomial(logp[0].exp().reshape(logp.shape[1], -1).T, 4, replacement=True).T
            k = k.reshape(4, *logp.shape[-2:])
        (-rloo._logp(logp[0], k).mean()).backward()
        grads.append({"pair": it["pair"], "policy": "coarse gm_cls", **grad_norms(model)})
        for s in rloo.SCALES:
            zero(model)
            out = model.forward(i0, i1)
            mu = out[int(s)]["flow"].float()
            sig = 1.0 / (HW / 2)                                                # 1 原网格 px
            act = (mu + sig * torch.randn_like(mu)).detach()
            (-(-((act - mu) ** 2).sum() / (2 * sig ** 2))).backward()
            grads.append({"pair": it["pair"], "policy": f"gaussian scale {s}", **grad_norms(model)})
        zero(model)
    # 一步训练的时间与显存（rloo 成分，reward r1）
    torch.cuda.reset_peak_memory_stats()
    term = rloo.Term(p, model, ds)

    class St:
        def __init__(self, it):
            self.it = it

        def images(self):
            return self.it["image0"][None].to(dev), self.it["image1"][None].to(dev)

    step_sec = []
    for i in range(5):
        t0 = time.time()
        loss, st, _ = term(St(ds[i]))
        loss.backward()
        zero(model)
        torch.cuda.synchronize()
        step_sec.append(time.time() - t0)
    out = RUN / "extra" / "data"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    cat = {k: np.concatenate(v) for k, v in cells.items()}
    cf = cat["conf"] > 0.5
    summ = {"n_pairs": len(rows), "group": a.group, "cells_conf_frac": round(float(cf.mean()), 4),
            "cell": {sub: {k: pct(cat[k][m]) for k in ("H", "pmode", "nonmode", "d_anchor")}
                     for sub, m in (("all", np.ones_like(cf)), ("conf", cf), ("rest", ~cf))},
            "pair": {k: pct([r[k] for r in rows if np.isfinite(r[k])])
                     for k in rows[0] if k != "pair"},
            "consistency_max_px": consist, "grads": grads,
            "probe_sec_per_pair": pct(secs), "train_step_sec": [round(x, 3) for x in step_sec],
            "train_step_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "uniform_entropy": round(float(np.log(4096)), 4)}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
