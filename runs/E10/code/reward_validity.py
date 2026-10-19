"""reward 有效性检验（「候选 reward 在 Val 上能否按真实误差排序」#164）：逐对造出误差已知的多个输出，算每个候选 reward 与真实误差。
只用 CPU，读 E7 现成的点对（RANSAC 之前、按 conf 截断到 2000 点），不推理、不训练；标注只用来算真实误差。

    python runs/E10/code/reward_validity.py --data <数据集根> --matches <主 checkout>/runs/E7/preds [--limit N] [--workers 8]

坐标为评测口径：原 512 网格、角点原点（点对 npz 已 +0.5，与 A、标注同一约定）。

输出（一个 patch 对的一个「版本」= 一份点对）：
- 合成族：以 zero-shot RoMa（E7/anymatch_roma）的点对为底，只改 SAR 一侧，每族 base + 4 档：
    shift   整体平移 d ∈ {1, 2, 3, 5} px，方向逐对随机
    rot     绕 patch 中心旋转，角度使图像边缘（距中心 256 px）位移 d，正负逐对随机
    scale   绕中心各向同性缩放 1 ± d/256，正负逐对随机
    noise   逐点独立高斯 σ ∈ {0.5, 1, 2, 3} px
    outlier 随机替换比例 f ∈ {10, 20, 30, 50}% 的点，SAR 位置在图内均匀分布
- 策略样本组 s1_1 / s1_2：G = 8 份逐点高斯 σ = 1 / 2 px（RL 策略 S1 的组内样本），不含 base。
- 真实族：real_roma 为 3 个 RoMa 模型（zero-shot、离线伪标签擦除 + 仿射第 2 轮、在 Val+Test 标注上过拟合），
  real_all 再加 5 个 LoFTR 系模型，各自在这一对上的实际点对。
每个版本 RANSAC（3 px，同 moonlib.ransac 的口径）拟合仿射 A，真实误差 = 检查点平均距离（workbench.protocol.pair_error）。

reward（都只用影像和点对，越大越好）：
- r1_inl   RANSAC 内点 +1、外点 −0.1，对全部点取平均（RIPE++ 的内点 / 外点打分）
- r1_grad  分级残差：对 A 的残差 r，max(0, 1 − (r/3)²) 的平均（截断二次核，形式仿 RIPE++ 附录的分级 reward，系数自拟）
- r2_cfog  用 A 把 SAR 拉到光学坐标系，在重叠区上比 CFOG 描述子的 NCC（CFOG：Ye et al., TGRS 2019，参数自拟）
- r2_ncc   同上，比两幅影像灰度的 NCC
- r3_<T>   已知合成扰动一致性：同一版本在 SAR 侧叠加 E7 的已知仿射 T（shift64 / rot10 / scale10）后的点对拟合出 A₂，
           reward = −(T∘A 与 A₂ 在光学网格上的平均距离)。合成族把同一误差在扰动后的坐标系里再施加一次（模型偏差跟着它看到的图走），
           噪声和离群点重新抽样；真实族直接用该模型在扰动图上的点对。r3_mean 为三者平均。
- r4_boot  子集重拟合稳定性：K = 8 个随机半子集各自 RANSAC，reward = −(子集仿射与 A 在光学网格上的平均距离) 的平均
- rand     均匀随机数（对照）
逐点检验（只对 base）：每个点对 A 的残差与它对标注最小二乘仿射的真实残差，算对内 Spearman，并统计 R1 内点里真实残差 < 3 px 的比例。

产物：runs/E10/extra/data/versions.csv.gz（逐对逐版本）、points.csv（逐对逐点检验）。统计与画图见 summarize.py。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                                   # runs/E10
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from moonlib import perturb                          # noqa: E402
from moonlib.inputs import sar_p2p98                 # noqa: E402
from workbench.dataset import Dataset                # noqa: E402
from workbench.protocol import pair_error            # noqa: E402

HW = 512
C = np.array([HW / 2, HW / 2])
THR = 3.0
LEVELS = (1.0, 2.0, 3.0, 5.0)                       # shift / rot / scale：图像边缘位移（px）
NOISE = (0.5, 1.0, 2.0, 3.0)
OUTLIER = (0.1, 0.2, 0.3, 0.5)
G, K = 8, 8
T_VARIANTS = ("shift64", "rot10", "scale10")
BASE = "anymatch_roma"
ROMA = ("anymatch_roma", "erase_geo__r2", "roma_vgg")
LOFTR = ("anymatch_loftr", "P10", "Q4", "Q11p", "loftr_lr5e5")
REWARDS = ("r1_inl", "r1_grad", "r2_cfog", "r2_ncc", *(f"r3_{t}" for t in T_VARIANTS), "r3_mean", "r4_boot", "rand")

_g = np.linspace(0, HW, 16)
GRID = np.stack(np.meshgrid(_g, _g), -1).reshape(-1, 2)


def key(pair: str) -> str:
    return pair.replace("/", "__")


def apply(A, p):
    return p @ A[:, :2].T + A[:, 2]


def compose(T, A):
    """T∘A（都是 2×3）。"""
    return np.c_[T[:, :2] @ A[:, :2], T[:, :2] @ A[:, 2] + T[:, 2]]


def grid_dist(A, B):
    return float(np.linalg.norm(apply(A, GRID) - apply(B, GRID), axis=1).mean())


def fit(src, dst):
    """moonlib.ransac 的口径（confidence 0.99999、10000 次、每次重置种子），坐标直接用角点约定。"""
    if len(src) < 3:
        return None, None
    cv2.setRNGSeed(0)
    A, inl = cv2.estimateAffine2D(src.astype(np.float64), dst.astype(np.float64), method=cv2.RANSAC,
                                  ransacReprojThreshold=THR, maxIters=10000, confidence=0.99999)
    if A is None or inl is None or int(inl.sum()) < 3:
        return None, None
    return A, inl.ravel().astype(bool)


# ---------- 合成误差 ----------

def similarity(theta=0.0, s=1.0, t=(0.0, 0.0)):
    L = s * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.c_[L, C - L @ C + np.asarray(t)]


def corrupt(sar, family, level, prm, rng):
    """在 SAR 一侧施加一档误差。prm：逐对固定的方向 / 符号。"""
    if family == "shift":
        return apply(similarity(t=level * np.array([np.cos(prm["u"]), np.sin(prm["u"])])), sar)
    if family == "rot":
        return apply(similarity(theta=prm["sign"] * level / (HW / 2)), sar)
    if family == "scale":
        return apply(similarity(s=1 + prm["sign"] * level / (HW / 2)), sar)
    if family in ("noise", "s1"):
        return sar + rng.normal(0, level, sar.shape)
    if family == "outlier":
        out = sar.copy()
        idx = rng.choice(len(sar), int(round(level * len(sar))), replace=False)
        out[idx] = rng.uniform(0, HW, (len(idx), 2))
        return out
    raise ValueError(family)


# ---------- CFOG ----------

N_ORI, SIGMA = 9, 2.0


def cfog(img):
    """CFOG（Ye et al., Fast and Robust Matching for Multimodal Remote Sensing Image Registration, TGRS 2019）：
    9 个方向的有向梯度取绝对值（对模态间梯度反向不敏感），空间高斯平滑后沿方向维 [1,2,1] 平滑，逐像素 L2 归一化。"""
    img = img.astype(np.float32)
    gx = cv2.filter2D(img, -1, np.array([[-1, 0, 1]], np.float32))
    gy = cv2.filter2D(img, -1, np.array([[-1], [0], [1]], np.float32))
    th = np.arange(N_ORI) * np.pi / N_ORI
    d = np.abs(np.cos(th) * gx[..., None] + np.sin(th) * gy[..., None])
    d = np.stack([cv2.GaussianBlur(d[..., k], (0, 0), SIGMA) for k in range(N_ORI)], -1)
    d = 0.25 * np.roll(d, 1, -1) + 0.5 * d + 0.25 * np.roll(d, -1, -1)       # 方向是周期的
    return d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-6)


def warp_to_opt(img, A):
    """out(x) = img(A·x)，x 为光学像素；A 是角点约定，换成 cv2 的像素中心约定。"""
    L, t = A[:, :2], A[:, 2]
    M = np.c_[L, t + 0.5 * L.sum(1) - 0.5]
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    w = cv2.warpAffine(img, M, (HW, HW), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = cv2.warpAffine(np.ones((HW, HW), np.float32), M, (HW, HW), flags=flags, borderValue=0)
    m = cv2.erode((m > 0.999).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return w, m


def ncc(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


# ---------- 一对 ----------

class Ctx:
    def __init__(self, a):
        self.ds = Dataset(a.data)
        self.mroot = Path(a.matches)
        self.T = {v: perturb.load(REPO / "runs" / "E7", v, "val") for v in T_VARIANTS}
        self.models = ROMA + LOFTR


CTX: Ctx | None = None


def init(a):
    global CTX
    cv2.setNumThreads(1)
    CTX = Ctx(a)


def load_matches(method, pair):
    with np.load(CTX.mroot / method / "val_matches.npz") as z:
        k = key(pair)
        return z[k][:, :4].astype(np.float64) if k in z.files else None


def rewards(opt, sar, A, inl, opt_f, sar_img, opt_desc, rng, subsets, A2s, Ts):
    r = {}
    res = np.linalg.norm(apply(A, opt) - sar, axis=1)
    r["r1_inl"] = float(np.where(res < THR, 1.0, -0.1).mean())
    r["r1_grad"] = float(np.clip(1 - (res / THR) ** 2, 0, None).mean())
    w, m = warp_to_opt(sar_img, A)
    if m.sum() < 1000:
        r["r2_cfog"] = r["r2_ncc"] = -1.0
    else:
        r["r2_ncc"] = ncc(opt_f[m][:, None], w[m][:, None])
        r["r2_cfog"] = ncc(opt_desc[m], cfog(w)[m])
    for v in T_VARIANTS:
        A2 = A2s.get(v)
        r[f"r3_{v}"] = -grid_dist(compose(Ts[v], A), A2) if A2 is not None else np.nan
    r["r3_mean"] = float(np.mean([r[f"r3_{v}"] for v in T_VARIANTS]))
    d = []
    for idx in subsets(len(opt)):
        As, _ = fit(opt[idx], sar[idx])
        d.append(grid_dist(As, A) if As is not None else 1e3)
    r["r4_boot"] = -float(np.mean(d))
    r["rand"] = float(rng.random())
    return r


def run_pair(pair):
    ds = CTX.ds
    gt = ds.checkpoints("val", pair)
    seed = zlib.crc32(pair.encode())
    rng = np.random.default_rng(seed)
    prm = {"u": rng.uniform(0, 2 * np.pi), "sign": 1.0 if rng.random() < 0.5 else -1.0}
    opt_img = ds.optical("val", pair)
    opt_img = (opt_img if opt_img.ndim == 2 else opt_img[..., 0]).astype(np.float32) / 255.0
    sar_img = sar_p2p98(ds.sar("val", pair))
    opt_desc = cfog(opt_img)
    Ts = {v: CTX.T[v][pair] for v in T_VARIANTS}
    # 标注最小二乘仿射：逐点真实残差用
    go, gs = gt
    Agt = np.linalg.lstsq(np.c_[go, np.ones(len(go))], gs, rcond=None)[0].T

    def subsets(n):
        srng = np.random.default_rng(seed + 1)
        return [srng.choice(n, n // 2, replace=False) for _ in range(K)]

    raw = {m: load_matches(m, pair) for m in CTX.models}
    rawT = {(m, v): load_matches(f"{m}__{v}", pair) for m in CTX.models for v in T_VARIANTS}
    rows, points = [], []

    def evaluate(family, version, level, opt, sar, sarT):
        """sarT：{T 变体: (opt, sar) 扰动图上的点对}。"""
        A, inl = fit(opt, sar)
        row = {"pair": pair, "family": family, "version": version, "level": level, "n": len(opt)}
        if A is None:
            rows.append({**row, "err": np.inf})
            return None
        A2s = {}
        for v, (o2, s2) in sarT.items():
            A2s[v], _ = fit(o2, s2) if o2 is not None else (None, None)
        r = rewards(opt, sar, A, inl, opt_img, sar_img, opt_desc, rng, subsets, A2s, Ts)
        rows.append({**row, "err": pair_error(A, *gt), **r})
        return A

    # 合成族与策略样本组
    o0, s0 = raw[BASE][:, :2], raw[BASE][:, 2:]
    oT = {v: (rawT[(BASE, v)][:, :2], rawT[(BASE, v)][:, 2:]) for v in T_VARIANTS}
    A0 = evaluate("base", "base", 0.0, o0, s0, oT)
    fams = [("shift", LEVELS), ("rot", LEVELS), ("scale", LEVELS), ("noise", NOISE), ("outlier", OUTLIER)]
    for fam, lvls in fams:
        for lv in lvls:
            # 同一误差在扰动图上再施加一次：确定性误差用同一参数，随机误差各自抽样
            s = corrupt(s0, fam, lv, prm, rng)
            sT = {v: (o, corrupt(st, fam, lv, prm, rng)) for v, (o, st) in oT.items()}
            evaluate(fam, f"{fam}{lv:g}", lv, o0, s, sT)
    for sig in (1.0, 2.0):
        for g in range(G):
            s = corrupt(s0, "s1", sig, prm, rng)
            sT = {v: (o, corrupt(st, "s1", sig, prm, rng)) for v, (o, st) in oT.items()}
            evaluate(f"s1_{sig:g}", f"s1_{sig:g}_{g}", sig, o0, s, sT)
    # 真实族
    for m in CTX.models:
        if raw[m] is None or len(raw[m]) < 3:
            continue
        sT = {v: ((rawT[(m, v)][:, :2], rawT[(m, v)][:, 2:]) if rawT[(m, v)] is not None else (None, None))
              for v in T_VARIANTS}
        evaluate("real", m, 0.0, raw[m][:, :2], raw[m][:, 2:], sT)
    # 逐点检验（base）
    if A0 is not None:
        res = np.linalg.norm(apply(A0, o0) - s0, axis=1)
        true = np.linalg.norm(apply(Agt, o0) - s0, axis=1)
        from scipy.stats import spearmanr
        inl = res < THR
        points.append({"pair": pair, "n": len(o0), "spearman": float(spearmanr(res, true)[0]),
                       "inl_frac": float(inl.mean()),
                       "true_ok_in_inl": float((true[inl] < THR).mean()) if inl.any() else np.nan,
                       "true_ok_in_out": float((true[~inl] < THR).mean()) if (~inl).any() else np.nan,
                       "true_ok_all": float((true < THR).mean()),
                       "gt_fit_err": pair_error(Agt, *gt)})
    return rows, points


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--matches", required=True, help="E7 的 preds 目录（含 *_matches.npz）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    pairs = Dataset(a.data).labelled("val")
    if a.limit:
        pairs = pairs[:: max(1, len(pairs) // a.limit)][: a.limit]
    out = RUN / "extra" / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows, points = [], []
    with Pool(a.workers, initializer=init, initargs=(a,)) as pool:
        for i, (r, p) in enumerate(pool.imap(run_pair, pairs, chunksize=2)):
            rows += r
            points += p
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(pairs)}", flush=True)
    cols = ["pair", "family", "version", "level", "n", "err", *REWARDS]
    with gzip.open(out / "versions.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(out / "points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(points[0]))
        w.writeheader()
        for r in points:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(pairs)} 对，{len(rows)} 个版本 → {out}")


if __name__ == "__main__":
    main()
