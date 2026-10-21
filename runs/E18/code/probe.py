"""真值诊断（「【RoMa·RL】诊断：细化各级加高斯后能否抽到更好的点，σ 的取值范围与多链训练开销」#189）：只读 zero-shot
AnyMatch-RoMa，不训练。

    python runs/E18/code/probe.py --n 200 [--group 16]

数据、扰动、真值同 E16（runs/E16/code/probe.py）：Val 有标注的 pair 均匀取 n 对，SAR 一侧加难版几何扰动 + 擦除；
标注点最小二乘拟合仿射 A_lab（光学 → 原 SAR），叠加已知扰动 T 得光学 → 扰动后 SAR 的真值仿射 A_true = T∘A_lab。

正常推理一次：粗级取 mode，跑尺度 16…1 的细化，记下尺度 8、4、2、1 各级的输出 warp（这一级加完位移、上采样之前）。
尺度 1 的输出按 E15 同款抽 5000 点、RANSAC 3 px 拟合参照仿射 A_ref（训练时奖励的参照）。

加噪：每次只在一级 s 上加。这一级加噪后的输出 = 不加噪的输出 + 各向同性高斯（σ = k·s，原网格 px，即 512 网格；
归一化坐标下 σ / 256），其余各级不加、粗级不抽签。这里只量这一级的输出，所以不用往下跑。每对、每级、每个 k 抽 group 次。

逐像素（这一级输出网格上的每个像素，只统计真值落点在 SAR 画面内的）：
- 误差 e：输出位置与像素中心经 A_true 的真值落点之间的距离（原网格 px）；不加噪时为 e0；
- 奖励 r：输出位置与像素中心经 A_ref 的位置相距 d，d ≤ 2s 时 1 − ½(d/s)²，否则 −0.1（δ = s、τ = 2δ）；
- 至少一次更好：min_g e_g < e0；最好一次的改善：e0 − min_g e_g；中位数差：median_g e_g − e0（16 个取第 8、9 个的均值）；
  全都更差：所有 e_g > e0；组内秩相关：r 与 e 在 group 次之间的 Spearman（并列取平均秩；任一方全相同记 nan）。
比例按全部像素精确计数；分布（分位数）用每对每级随机取的至多 --keep 个像素（各 k 共用同一批像素）。

产物：runs/E18/extra/data/summary.json、pairs.csv（逐对 × 级 × k）。
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
from finetune.geom import HW, compose               # noqa: E402
from finetune.models import roma                    # noqa: E402
from finetune.parts import aug, rloo                # noqa: E402
from finetune.parts.pseudo import affine_norm, grid  # noqa: E402

CFG = {"model": {"name": "roma", "train_vgg": True, "lora_r": 8},
       "optim": {"lr": 3e-5},
       "aug": {"shift": 128.0, "shift_disk": True, "rot": 20.0, "scale": 0.2, "photo": False, "erase": True},
       "rloo": {}}                                  # 与 E16 相同（数据与扰动一致）
LEVELS = (8, 4, 2, 1)
PX = HW / 2                                         # 归一化坐标 → 原网格 px


def labels(ds, pair):
    """(光学 N×2, 原 SAR N×2)，同 E16。"""
    a = np.loadtxt(ds.data.path(ds.src[pair], pair, "Label"), dtype=np.float64, ndmin=2)
    a[:, [1, 3]] *= -1
    return a[:, :2], a[:, 2:4]


def lstsq_affine(p, q):
    X = np.c_[p, np.ones(len(p))]
    return np.linalg.lstsq(X, q, rcond=None)[0].T


@torch.no_grad()
def refine_levels(model, f1, f2, flow, certainty):
    """同 rloo.refine（尺度 16…1，不加噪声），另外返回各级加完位移、上采样之前的输出 {s: (B,2,h,w)}。"""
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


def amap_t(A, h, w, dev):
    """原网格 2×3 仿射 → 这一级像素中心的映射位置（归一化坐标，(h,w,2)）。"""
    An = torch.from_numpy(affine_norm(A)).to(dev).float()
    return torch.einsum("ij,hwj->hwi", An[:, :2], grid(h, w, dev)) + An[:, 2]


def avg_rank(x):
    """x (G,N) → 每列的平均秩（并列取平均），(G,N)。"""
    lt = (x[None] < x[:, None]).sum(1).float()       # [i] = #{j: x_j < x_i}
    eq = (x[None] == x[:, None]).sum(1).float()
    return lt + (eq + 1) / 2


def spearman_cols(a, b):
    """a、b (G,N) → 每列的 Spearman (N,)；任一方全相同为 nan。"""
    ra, rb = avg_rank(a), avg_rank(b)
    ra, rb = ra - ra.mean(0), rb - rb.mean(0)
    den = (ra.square().sum(0) * rb.square().sum(0)).sqrt()
    return torch.where(den > 0, (ra * rb).sum(0) / den.clamp(min=1e-12), torch.full_like(den, float("nan")))


def pct(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, np.float64)
    x = x[~np.isnan(x)]
    if not len(x):
        return None
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs} | {"mean": round(float(x.mean()), 4),
                                                                         "n": int(len(x))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--ks", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--keep", type=int, default=4096, help="每对每级留作分布统计的像素数")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    gen = torch.Generator(device="cuda").manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = C.expand(CFG)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    model = roma.Model(cfg, a.weights_root)
    p = cfg["rloo"]
    ds = PairSet(a.data, a.split, model.resize, **model.input, seed=0, **aug.data_kw(cfg["aug"]))
    lab = [q for q in ds.pairs if ds.data.path(ds.src[q], q, "Label").exists()]
    ds.pairs = lab[:: max(1, len(lab) // a.n)][: a.n]
    print(f"{a.split}: {len(lab)} 对有标注，取 {len(ds.pairs)} 对", flush=True)
    dev = "cuda"
    G = a.group

    cnt = {(s, k): {"n": 0, "better_any": 0, "all_worse": 0, "rho_nan": 0} for s in LEVELS for k in a.ks}
    keep = {(s, k): {"gain": [], "dmed": [], "rho": [], "gain_pos": []} for s in LEVELS for k in a.ks}
    base = {s: {"e0": [], "ref_true": [], "n_px": 0, "n_inside": 0} for s in LEVELS}
    rows, secs, n_ref_fail = [], [], 0

    for i in range(len(ds)):
        it = ds[i]
        T = it["T"].numpy().astype(np.float64)
        po, qo = labels(ds, it["pair"])
        A_true = compose(T, lstsq_affine(po, qo))
        i0, i1 = it["image0"][None].to(dev), it["image1"][None].to(dev)
        t0 = time.time()
        with torch.no_grad():
            f1, f2 = rloo.features(model, i0, i1)
            cls, cert = rloo.coarse(model, f1, f2)
            P = cls.float().softmax(1)
            outs, flow1, ce1 = refine_levels(model, f1, f2, rloo.anchor_flow(P, P.argmax(1)), cert.float())
            A_ref, n_inl = rloo.fit(flow1[0], ce1[0], p["num"], p["ransac"])
            n_ref_fail += A_ref is None
            for s in LEVELS:
                mu = outs[s][0].permute(1, 2, 0).float()                         # (h,w,2)
                h, w = mu.shape[:2]
                xt = amap_t(A_true, h, w, dev)
                ins = (xt.abs() < 1).all(-1).reshape(-1)
                mu, xt = mu.reshape(-1, 2)[ins], xt.reshape(-1, 2)[ins]
                xr = amap_t(A_ref, h, w, dev).reshape(-1, 2)[ins] if A_ref is not None else None
                N = mu.shape[0]
                e0 = (mu - xt).norm(dim=-1) * PX                                   # (N,)
                sel = torch.from_numpy(rng.choice(N, min(N, a.keep), replace=False)).to(dev)
                b = base[s]
                b["n_px"] += h * w
                b["n_inside"] += N
                b["e0"].append(e0[sel].cpu().numpy())
                if xr is not None:
                    b["ref_true"].append(((xr - xt).norm(dim=-1) * PX)[sel].cpu().numpy())
                for k in a.ks:
                    sig = k * s / PX
                    x = mu[None] + sig * torch.randn(G, N, 2, device=dev, generator=gen)  # (G,N,2)
                    e = (x - xt[None]).norm(dim=-1) * PX                                    # (G,N)
                    if xr is not None:
                        d = (x - xr[None]).norm(dim=-1) * PX
                        r = torch.where(d <= 2 * s, 1 - 0.5 * (d / s) ** 2, torch.full_like(d, -0.1))
                    else:
                        r = torch.zeros_like(e)
                    es = e.sort(0).values
                    emin = es[0]
                    gain = e0 - emin
                    dmed = 0.5 * (es[(G - 1) // 2] + es[G // 2]) - e0
                    better = emin < e0
                    worse = (e > e0[None]).all(0)
                    rho = spearman_cols(r, e)
                    c = cnt[(s, k)]
                    c["n"] += N
                    c["better_any"] += int(better.sum())
                    c["all_worse"] += int(worse.sum())
                    c["rho_nan"] += int(rho.isnan().sum())
                    kp = keep[(s, k)]
                    kp["gain"].append(gain[sel].cpu().numpy())
                    kp["dmed"].append(dmed[sel].cpu().numpy())
                    kp["rho"].append(rho[sel].cpu().numpy())
                    kp["gain_pos"].append(gain[sel][better[sel]].cpu().numpy())
                    rows.append({"pair": it["pair"], "level": s, "k": k, "sigma_px": k * s, "n_px": N,
                                 "shift": float(np.linalg.norm(T[:, 2] - (np.eye(2) - T[:, :2]) @ [HW / 2, HW / 2])),
                                 "ref_ok": A_ref is not None, "n_inl_ref": n_inl,
                                 "e0_med": float(e0.median()), "frac_better_any": float(better.float().mean()),
                                 "gain_med": float(gain.median()), "dmed_med": float(dmed.median()),
                                 "frac_all_worse": float(worse.float().mean()),
                                 "rho_med": float(rho.nanmedian()) if (~rho.isnan()).any() else np.nan})
        secs.append(time.time() - t0)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(ds)} {secs[-1]:.2f}s", flush=True)

    out = RUN / "extra" / "data"
    out.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: f"{v:.6g}" if isinstance(v, float) else v
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: fmt(v) for k, v in r.items()})

    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    by = {}
    for s in LEVELS:
        b = base[s]
        lv = {"n_px": b["n_px"], "n_inside": b["n_inside"], "e0": pct(cat(b["e0"])),
              "ref_true": pct(cat(b["ref_true"])), "by_k": {}}
        for k in a.ks:
            c, kp = cnt[(s, k)], keep[(s, k)]
            rho = cat(kp["rho"])
            lv["by_k"][str(k)] = {
                "sigma_px": k * s,
                "frac_better_any": round(c["better_any"] / c["n"], 4),
                "frac_all_worse": round(c["all_worse"] / c["n"], 4),
                "frac_rho_nan": round(c["rho_nan"] / c["n"], 4),
                "gain": pct(cat(kp["gain"])), "gain_if_better": pct(cat(kp["gain_pos"])),
                "dmed": pct(cat(kp["dmed"])), "rho": pct(rho),
                "frac_rho_neg": round(float((rho[~np.isnan(rho)] < 0).mean()), 4) if (~np.isnan(rho)).any() else None,
            }
        by[str(s)] = lv
    summ = {"split": a.split, "n_labelled": len(lab), "n_pairs": len(ds), "group": G, "ks": a.ks,
            "n_ref_fail": n_ref_fail, "by_level": by, "sec_per_pair": pct(secs)}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
