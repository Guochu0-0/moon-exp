"""诊断：跨模态结构相似度能否当「整对 reward」——它看得见几个像素的整体偏移吗？（首轮 RL 实验设计，地图 #1）

    python -m baselines.diagnose_reward --pred runs/S1/preds/main/val.jsonl --out <dir> [--workers 16]

对每个有标注的 Val pair：光学、SAR 各算一次特征，把光学特征按基准仿射 warp 到 SAR 网格，
在 ±R px 的整数平移上算相似度曲面，峰值做抛物线亚像素插值。基准仿射两种：
- A_gt：8 个检查点最小二乘拟合（标注地板意义下的「真值」）。看峰值偏离 0 多远：reward 的最优点是否就是真值。
- A_pred：--pred 给的预测。把它平移到峰值处（A' = A − d），看真值误差是否下降：reward 最优策略的增益上限。

相似度（per-pixel 特征在去掉边框的重叠区上）：
- cfog：9 方向梯度通道，空间高斯 + 方向 [1,2,1] 平滑，逐像素 L2 归一化，取平均余弦（Ye et al. CFOG）。
- gradncc：梯度幅值的 NCC。
- ncc：原始灰度 NCC。
输入映射与主表相同（光学 /255，SAR p2p98）。坐标：A 为角点约定（同 workbench），cv2 为像素中心约定。
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from baselines import inputs
from workbench import protocol
from workbench.dataset import Dataset

R = 6          # 平移搜索半径（原网格 px）
MARGIN = 24    # 比较区去掉的边框，须 > R
NORI = 9


def cfog(img):
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    th = np.arange(NORI) * np.pi / NORI
    F = np.stack([np.abs(gx * np.cos(t) + gy * np.sin(t)) for t in th])
    F = np.stack([cv2.GaussianBlur(f, (0, 0), 0.8) for f in F])
    F = 0.25 * np.roll(F, 1, 0) + 0.5 * F + 0.25 * np.roll(F, -1, 0)
    return F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-6)


def gradmag(img):
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)[None]


FEATS = {"cfog": cfog, "gradncc": gradmag, "ncc": lambda x: x.astype(np.float32)[None]}


def warp_feat(F, A, shape):
    """光学特征 F (C,h,w) → SAR 网格：SAR 像素 p 取光学 A^{-1}(p)。A 角点约定 → 中心约定再求逆。"""
    L, t = A[:, :2], A[:, 2]
    Ac = np.c_[L, t + L @ [0.5, 0.5] - 0.5]
    M = cv2.invertAffineTransform(Ac)
    H, W = shape
    Wf = np.stack([cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=np.nan) for f in F])
    return Wf


def surface(Fw, Fs, kind):
    """Fw, Fs: (C,H,W)。返回 (2R+1, 2R+1) 相似度，行 = dy，列 = dx；S[dy,dx] 比较 SAR(p) 与 warped(p + d)。"""
    C, H, W = Fs.shape
    m = MARGIN
    ref = Fs[:, m:H - m, m:W - m]
    out = np.full((2 * R + 1, 2 * R + 1), np.nan)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            mov = Fw[:, m + dy:H - m + dy, m + dx:W - m + dx]
            ok = np.isfinite(mov).all(0)
            if ok.mean() < 0.5:
                continue
            a, b = mov[:, ok], ref[:, ok]
            if kind == "cfog":
                out[dy + R, dx + R] = float((a * b).sum(0).mean())
            else:
                a = a - a.mean(); b = b - b.mean()
                out[dy + R, dx + R] = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
    return out


def peak(S):
    """整数峰 + 各轴抛物线插值。返回 (dx, dy, 峰值 − 距峰 2–3 px 环上均值)。"""
    if not np.isfinite(S).any():
        return np.nan, np.nan, np.nan
    iy, ix = np.unravel_index(np.nanargmax(S), S.shape)

    def sub(v0, v1, v2):
        den = v0 - 2 * v1 + v2
        return 0.0 if not np.isfinite(den) or den >= 0 else 0.5 * (v0 - v2) / den

    dx = ix - R + (sub(S[iy, ix - 1], S[iy, ix], S[iy, ix + 1]) if 0 < ix < 2 * R else 0)
    dy = iy - R + (sub(S[iy - 1, ix], S[iy, ix], S[iy + 1, ix]) if 0 < iy < 2 * R else 0)
    yy, xx = np.mgrid[:2 * R + 1, :2 * R + 1]
    d = np.hypot(yy - iy, xx - ix)
    ring = S[(d >= 2) & (d <= 3)]
    return float(dx), float(dy), float(S[iy, ix] - np.nanmean(ring))


def fit_gt(o, s):
    X = np.c_[o, np.ones(len(o))]
    return np.linalg.lstsq(X, s, rcond=None)[0].T


def one(job):
    root, pair, Ap = job
    ds = Dataset(root)
    opt = inputs.get("optical", "div255")(ds.optical("val", pair))
    sar = inputs.get("sar", "p2p98")(ds.sar("val", pair))
    o, s = ds.checkpoints("val", pair)
    Ag = fit_gt(o, s)
    rec = {"pair": pair, "err_gt_fit": protocol.pair_error(Ag, o, s)}
    for name, fn in FEATS.items():
        Fo, Fs = fn(opt), fn(sar)
        for tag, A in (("gt", Ag), ("pred", Ap)):
            if A is None:
                continue
            dx, dy, sharp = peak(surface(warp_feat(Fo, A, sar.shape), Fs, name))
            rec[f"{name}_{tag}_dx"], rec[f"{name}_{tag}_dy"], rec[f"{name}_{tag}_sharp"] = dx, dy, sharp
            if tag == "pred" and np.isfinite(dx):
                A2 = A.copy(); A2[:, 2] -= [dx, dy]
                rec[f"{name}_refined_err"] = protocol.pair_error(A2, o, s)
    if Ap is not None:
        rec["pred_err"] = protocol.pair_error(Ap, o, s)
        rec["pred_minus_gt_t"] = (Ap[:, 2] - Ag[:, 2]).tolist()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="workbench 记录的 val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    preds = {}
    for line in Path(args.pred).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            preds[d["pair"]] = None if d.get("A") is None else np.asarray(d["A"], float)
    ds = Dataset(args.data)
    pairs = ds.labelled("val")[: args.limit or None]
    with Pool(args.workers) as pool:
        recs = pool.map(one, [(args.data, p, preds.get(p)) for p in pairs], chunksize=4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    summ = {"n": len(recs), "R": R}
    base = np.array([r.get("pred_err", np.inf) for r in recs])
    summ["pred"] = {k: round(protocol.auc(base, t), 4) for k, t in (("auc@3", 3), ("auc@5", 5), ("auc@10", 10))}
    for name in FEATS:
        g = np.array([[r.get(f"{name}_gt_dx", np.nan), r.get(f"{name}_gt_dy", np.nan)] for r in recs])
        dist = np.hypot(*g.T)
        ref = np.array([r.get(f"{name}_refined_err", np.inf) for r in recs])
        summ[name] = {
            "gt_peak_dist_median": round(float(np.nanmedian(dist)), 3),
            "gt_peak_within_1px": round(float(np.nanmean(dist <= 1)), 3),
            "gt_peak_within_2px": round(float(np.nanmean(dist <= 2)), 3),
            "gt_peak_dx_std": round(float(np.nanstd(g[:, 0])), 3),
            "gt_peak_dy_std": round(float(np.nanstd(g[:, 1])), 3),
            "refined": {k: round(protocol.auc(ref, t), 4) for k, t in (("auc@3", 3), ("auc@5", 5), ("auc@10", 10))},
            "refined_better_frac": round(float(np.mean(ref < base)), 3),
        }
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
