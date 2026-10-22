"""奖励信号对错的诊断（「【RoMa·RL】诊断：粗级与细化各级的奖励信号有多少是对的」#195）：只读 zero-shot AnyMatch-RoMa，
不训练。

    python runs/E21/code/probe.py --shard 0 --nshard 2 [--n 200]

数据、扰动、真值、参照仿射同 E18（runs/E18/code/probe.py）：Val 有标注的图对均匀取 n 对，SAR 一侧加难版扰动 + 擦除；
真值仿射 A_true = 扰动 ∘ 标注点最小二乘仿射。正常推理一次（粗级取最高分锚点，细化尺度 16…1 照常），尺度 1 的输出按
E15 同款抽 5000 点、RANSAC 3 px 拟合参照仿射 A_ref。只统计真值落点在 SAR 画面内的格子 / 像素。长度一律为原网格 px（512）。

每格 / 像素：T 为真值落点，R 为 A_ref 给出的位置，奖励 r = exp(−½(d/δ)²)，d = |位置 − R|；真值误差 e = |位置 − T|。
每次只在一级加随机性，其余各级照正常推理，只量这一级本身的输出。ΔE[e] 为一步期望更新下期望真值误差的一阶变化：
< 0 记为对，> 0 记为错，= 0（奖励处处下溢为 0）记为无信号；|ΔE[e]| 为强度。

- 粗级（40 × 40 格，δ = 8）：π = softmax(gm_cls / 0.3)；锚点 j 的位置 x_j = rloo.anchor_flow（抽到 j、与 4 个邻居按
  softmax(gm_cls) 加权，同训练）。把每格 4096 个 logits 当独立参数，x_j 视为固定：
  b. ΔE = Σ_j π_j²(r_j − r̄)(e_j − ē)；
  c. 抽 16 次 k_g ~ π，RLOO 优势 adv_g = r_g − 其余 15 次的均值，实际的 logit 变化 Δθ_j = mean_g adv_g(1[k_g = j] − π_j)，
     ΔE = Σ_j π_j(e_j − ē)Δθ_j；
  a. 几何模型（geom.coarse_a）：M = Σπ_j x_j，s = √(Σπ_j|x_j − M|² / 2)。
- 细化尺度 s ∈ {16, 8, 4, 2, 1}（这一级输出网格），μ 为这一级加完位移的输出，σ = k·s，δ = s：
  b. 每像素抽 1024 次 a ~ N(μ, σ²)，u = 样本上 (r − r̄)(a − μ)/σ²，v = 样本上 (e − ē)(a − μ)/σ²，
     ΔE = Σ_{i≠j} u_i·v_j / (n(n − 1))（去掉同一样本自乘的偏差）；
  c. 抽 16 次，Δμ = mean_g adv_g(a_g − μ)/σ²，ΔE = Δμ · ∇_μE[e]（∇_μE[e] 用闭式，geom）；
  a. 几何模型（geom.fine_a）：M = μ，s = σ。三层的比例尺相同（c 的期望 = b 的期望 = a）。

配准成功 / 失败同 E18：E16 正常推理（温度 1 一行的 err0）标注点误差 ≤ 5 px 为成功。
产物：runs/E21/raw/shard<i>.npz（逐配置、组、b/s 分档的计数与强度和；逐像素随机子样本）、shard<i>_pairs.csv。
汇总与出图见 report.py。
"""
from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(HERE))

from finetune import config as C                    # noqa: E402
from finetune.data import PairSet                   # noqa: E402
from finetune.geom import HW, compose               # noqa: E402
from finetune.models import roma                    # noqa: E402
from finetune.parts import aug, rloo                # noqa: E402
from finetune.parts.pseudo import affine_norm, grid  # noqa: E402
import geom                                          # noqa: E402

CFG = {"model": {"name": "roma", "train_vgg": True, "lora_r": 8},
       "optim": {"lr": 3e-5},
       "aug": {"shift": 128.0, "shift_disk": True, "rot": 20.0, "scale": 0.2, "photo": False, "erase": True},
       "rloo": {}}                                  # 同 E16 / E18
FINE = (16, 8, 4, 2, 1)
KS = (0.25, 0.5, 1.0)
CONFIGS = ["coarse"] + [f"s{s}_k{k}" for s in FINE for k in KS]
TEMP, DELTA_C, G, NMC = 0.3, 8.0, 16, 1024
BINS = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16, np.inf])      # b/s 分档
LAYERS = ("a", "b", "c")
# 计数表 acc[配置]：(层 3, 组 2（成功、失败）, 分档, 6)，6 = 像素数、对、错、无信号、对的强度和、错的强度和
PX = HW / 2
E16 = RUN.parent / "E16" / "extra" / "data" / "pairs.csv"
KEEP = 64                                                    # 每对每配置留作直方图的像素数


def labels(ds, pair):
    a = np.loadtxt(ds.data.path(ds.src[pair], pair, "Label"), dtype=np.float64, ndmin=2)
    a[:, [1, 3]] *= -1
    return a[:, :2], a[:, 2:4]


def lstsq_affine(p, q):
    X = np.c_[p, np.ones(len(p))]
    return np.linalg.lstsq(X, q, rcond=None)[0].T


@torch.no_grad()
def refine_levels(model, f1, f2, flow, certainty):
    """同 E18：尺度 16…1 照推理细化，返回各级加完位移、上采样之前的输出 {s: (B,2,h,w)} 与尺度 1 的 (flow, certainty)。"""
    dec = model.model.decoder
    h, w = f1[1].shape[-2:]
    outs = {}
    for s in rloo.SCALES:
        ins = int(s)
        d_flow, d_cert = dec.conv_refiner[s](f1[ins], f2[ins], flow, scale_factor=1, logits=certainty)
        disp = ins * torch.stack((d_flow[:, 0].float() / (dec.refine_init * w),
                                  d_flow[:, 1].float() / (dec.refine_init * h)), dim=1)
        flow, certainty = flow + disp, certainty + d_cert
        outs[ins] = flow
        if s != "1":
            size = f1[ins // 2].shape[-2:]
            flow = F.interpolate(flow, size=size, mode=dec.flow_upsample_mode)
            certainty = F.interpolate(certainty, size=size, mode=dec.flow_upsample_mode)
    return outs, flow, certainty


def amap(A, h, w, dev):
    """原网格 2×3 仿射 → 这一级像素中心的映射位置（px，(h·w, 2)，float64）。"""
    An = torch.from_numpy(affine_norm(A)).to(dev).double()
    return (torch.einsum("ij,hwj->hwi", An[:, :2], grid(h, w, dev).double()) + An[:, 2]).reshape(-1, 2) * PX


def reward(x, R, delta):
    return torch.exp(-0.5 * ((x - R).norm(dim=-1) / delta) ** 2)


def loo(r):
    return r - (r.sum(0, keepdim=True) - r) / (r.shape[0] - 1)


def coarse_level(cls, P, T, R, ins, gen):
    """cls (1,C,h,w)，P = softmax(cls)，T、R (h·w,2) px，ins 掩码 → (geo (N,4), dE (3,N))。"""
    C_, h, w = cls.shape[1:]
    pi = (cls[0].double() / TEMP).softmax(0).reshape(C_, -1)[:, ins]                   # (C,N)
    X = []
    for j0 in range(0, C_, 256):
        j = torch.arange(j0, min(C_, j0 + 256), device=cls.device)
        k = j[:, None, None].expand(-1, h, w)
        X.append(rloo.anchor_flow(P.expand(len(j), -1, -1, -1), k).permute(0, 2, 3, 1).reshape(len(j), -1, 2)[:, ins])
    X = torch.cat(X).double() * PX                                                     # (C,N,2)
    r, e = reward(X, R[None], DELTA_C), (X - T[None]).norm(dim=-1)
    rb, eb = (pi * r).sum(0), (pi * e).sum(0)
    ec = pi * (e - eb)                                                                 # ∂E[e]/∂θ_j
    dEb = (pi * (r - rb) * ec).sum(0)
    M = (pi[..., None] * X).sum(0)
    s = ((pi * ((X - M) ** 2).sum(-1)).sum(0) / 2).sqrt().clamp(min=1e-3)       # 单点分布时防除零
    dEa = geom.coarse_a(M, T, R, s, DELTA_C)
    k = torch.multinomial(pi.T.float(), G, replacement=True, generator=gen).T          # (G,N)
    adv = loo(torch.gather(r, 0, k))
    dEc = (adv * (torch.gather(ec, 0, k) - (pi * ec).sum(0))).mean(0)
    geo = torch.stack(((R - T).norm(dim=-1), (M - T).norm(dim=-1), (M - R).norm(dim=-1), s), 1)
    return geo, torch.stack((dEa, dEb, dEc))


def fine_level(mu, T, R, sig, delta, gen, chunk=4096):
    """mu、T、R (N,2) px → (geo (N,4), dE (3,N))。"""
    N = len(mu)
    dEa = geom.fine_a(mu, T, R, sig, delta)
    dEb = torch.empty(N, dtype=torch.float64, device=mu.device)
    for i in range(0, N, chunk):
        m, t, rr = mu[i:i + chunk], T[i:i + chunk], R[i:i + chunk]
        z = torch.randn(NMC, len(m), 2, device=mu.device, generator=gen, dtype=torch.float64)
        a = m + sig * z
        r, e = reward(a, rr, delta), (a - t).norm(dim=-1)
        u = (r - r.mean(0))[..., None] * z / sig                                     # (a − μ)/σ² = z/σ
        v = (e - e.mean(0))[..., None] * z / sig
        dEb[i:i + chunk] = ((u.sum(0) * v.sum(0)).sum(-1) - (u * v).sum((0, 2))) / (NMC * (NMC - 1))
    z = torch.randn(G, N, 2, device=mu.device, generator=gen, dtype=torch.float64)
    adv = loo(reward(mu + sig * z, R, delta))
    dmu = (adv[..., None] * z / sig).mean(0)
    dT = mu - T
    nu = dT.norm(dim=-1)
    ge = geom.rice_slope(nu, torch.tensor(sig, dtype=torch.float64)) / nu.clamp(min=1e-300)
    dEc = ge * (dmu * dT).sum(-1)
    geo = torch.stack(((R - T).norm(dim=-1), nu, (mu - R).norm(dim=-1), torch.full_like(nu, sig)), 1)
    return geo, torch.stack((dEa, dEb, dEc))


def tally(acc, geo, dE, grp):
    """把一对一个配置的结果累加进 acc (3,2,nbin,6)。"""
    bs = (geo[:, 0] / geo[:, 3]).cpu().numpy()
    bi = np.clip(np.searchsorted(BINS, bs, side="right") - 1, 0, len(BINS) - 2)
    d = dE.cpu().numpy()
    for li in range(3):
        x = d[li]
        cols = (np.ones_like(x), x < 0, x > 0, x == 0, np.where(x < 0, -x, 0), np.where(x > 0, x, 0))
        for ci, c in enumerate(cols):
            acc[li, grp, :, ci] += np.bincount(bi, weights=c.astype(np.float64), minlength=len(BINS) - 1)


def pair_row(pair, ok, cfg, geo, dE):
    row = {"pair": pair, "success": ok, "config": cfg, "n_px": geo.shape[0]}
    g = geo.cpu().numpy()
    for i, n in enumerate(("b", "mt", "mr", "s")):
        row[f"{n}_med"] = float(np.median(g[:, i]))
    for li, ln in enumerate(LAYERS):
        x = dE[li].cpu().numpy()
        tot = np.abs(x).sum()
        row[f"right_{ln}"] = float((x < 0).mean())
        row[f"net_{ln}"] = float(-x.sum() / tot) if tot > 0 else np.nan
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    cfg = C.expand(CFG)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    model = roma.Model(cfg, a.weights_root)
    p = cfg["rloo"]
    ds = PairSet(a.data, "val", model.resize, **model.input, seed=0, **aug.data_kw(cfg["aug"]))
    lab = [q for q in ds.pairs if ds.data.path(ds.src[q], q, "Label").exists()]
    ds.pairs = lab[:: max(1, len(lab) // a.n)][: a.n]
    succ = {r["pair"]: float(r["err0"]) <= 5 for r in csv.DictReader(open(E16, encoding="utf-8"))
            if float(r["temp"]) == 1.0}
    idx = list(range(a.shard, len(ds), a.nshard))
    print(f"val: {len(lab)} 对有标注，取 {len(ds.pairs)} 对，本片 {len(idx)} 对", flush=True)
    dev = "cuda"
    acc = {c: np.zeros((3, 2, len(BINS) - 1, 6)) for c in CONFIGS}
    keep = {c: [] for c in CONFIGS}
    rows, n_ref_fail = [], 0
    for n_done, i in enumerate(idx):
        t0 = time.time()
        it = ds[i]
        pair = it["pair"]
        ok = succ[pair]
        grp = 0 if ok else 1
        T_ = it["T"].numpy().astype(np.float64)
        po, qo = labels(ds, pair)
        A_true = compose(T_, lstsq_affine(po, qo))
        gen = torch.Generator(device=dev).manual_seed(1000 + i)
        rng = np.random.default_rng(i)
        with torch.no_grad():
            i0, i1 = it["image0"][None].to(dev), it["image1"][None].to(dev)
            f1, f2 = rloo.features(model, i0, i1)
            cls, cert = rloo.coarse(model, f1, f2)
            cls = cls.float()
            P = cls.softmax(1)
            outs, flow1, ce1 = refine_levels(model, f1, f2, rloo.anchor_flow(P, P.argmax(1)), cert.float())
            A_ref, _ = rloo.fit(flow1[0], ce1[0], p["num"], p["ransac"])
            if A_ref is None:
                n_ref_fail += 1
                continue
            jobs = [("coarse", cls.shape[-2:], None, None)]
            jobs += [(f"s{s}_k{k}", outs[s].shape[-2:], s, k) for s in FINE for k in KS]
            for name, (h, w), s, k in jobs:
                T, R = amap(A_true, h, w, dev), amap(A_ref, h, w, dev)
                ins = (T.abs() < PX).all(-1)
                T, R = T[ins], R[ins]
                if s is None:
                    geo, dE = coarse_level(cls, P, T, R, ins, gen)
                else:
                    mu = outs[s][0].permute(1, 2, 0).reshape(-1, 2)[ins].double() * PX
                    geo, dE = fine_level(mu, T, R, k * s, float(s), gen)
                tally(acc[name], geo, dE, grp)
                sel = rng.choice(len(geo), min(len(geo), KEEP), replace=False)
                keep[name].append(np.c_[np.full(len(sel), i), np.full(len(sel), grp),
                                        geo[sel].cpu().numpy(), dE[:, sel].T.cpu().numpy()].astype(np.float32))
                rows.append(pair_row(pair, ok, name, geo, dE))
        print(f"{n_done + 1}/{len(idx)} {time.time() - t0:.1f}s", flush=True)

    out = RUN / "raw"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"shard{a.shard}.npz", bins=BINS, configs=np.array(CONFIGS), n_ref_fail=n_ref_fail,
                        n_pairs=len(idx), **{f"acc_{c}": acc[c] for c in CONFIGS},
                        **{f"keep_{c}": np.concatenate(keep[c]) for c in CONFIGS})
    fmt = lambda v: f"{v:.6g}" if isinstance(v, float) else v
    with open(out / f"shard{a.shard}_pairs.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: fmt(v) for k, v in r.items()})
    print("done", flush=True)


if __name__ == "__main__":
    main()
