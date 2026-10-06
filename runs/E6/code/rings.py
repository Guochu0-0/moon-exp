"""按到陨石坑的距离分圈拟合仿射，在标注点上量误差（「在陨石坑附近拟合仿射是否更准」#114）。只用 CPU，读 infer.py 的产物。

    /opt/envs/loftr/bin/python runs/E6/code/rings.py [--models ...] [--limit N] [--workers 16]

做法（坐标为评测口径：原 512 网格、角点原点）：
- 候选对应：LoFTR 为全部匹配（RANSAC 之前），位置取光学侧；RoMa 为光学侧每个像素中心 (c + 0.5)，
  对应由 256 网格上的稠密位移双线性插值得到。
- 分圈：光学侧到陨石坑标注的距离，圈为 RINGS。主口径按「到最近陨石坑的距离」分圈，点归最近的坑；
  副口径照票面按「到各自坑的距离」分圈，一个点可同时在几个坑的圈里（同一次抽样内不重复取）。圈超出图像的部分自然截掉。
- 抽样：每次共 N_FIT 个点，各坑平分；某个坑的点不够，就把差额平分给同一对里还有余量的坑；整对凑不满 N_FIT，这一圈不计这一对。
- 拟合：RANSAC（3 px，cv2.estimateAffine2D，不做细化），再在内点上做普通最小二乘。每圈抽 DRAWS 次。
- 误差：在这一对的全部标注点上量 e = A·o − s。一次拟合失败或平均偏移某一轴超过 SANE 记为「拟合粗错」，不进误差、单独报告比例。
  一个点的误差先在这一圈的有效抽样之间平均（dx²、|e|²、|dx|、|e|），再在全部点上汇总：
  x 向均方误差 = 各点 mean(dx²) 的平均；x 向中位数 = 各点 mean(|dx|) 的中位数；二维同理。
- 对的筛选：本次推理的模型仿射失败、或平均偏移某一轴超过 SANE（同 E3）的对整对不计。主图用每个模型「五圈都有效」的对。
- 参照线（同一批对、同一批点）：模型仿射；全图对应拟合（全部候选，RANSAC + 最小二乘）；全图随机 N_FIT 点拟合（DRAWS 次，
  与各圈同样的点数）；标注点处用局部对应拟合（样本内，同 E3：LoFTR 为 12 px 内匹配位移的中位数、至少 3 个，RoMa 为稠密对应插值，
  离标注 ≥ SANE 的不用，至少 4 点，普通最小二乘）；用标注拟合的最佳仿射（样本内）。
- 对应质量（只描述）：LoFTR 圈内匹配密度（每 1000 px²），RoMa 圈内 certainty 均值，圈内拟合的 RANSAC 内点率。
- 区间：按对自助抽样 1000 次的 95% 区间；与最内圈之差也按同一批对自助抽样。
产物：runs/E6/extra/rings.json、rings_x.png、rings_quality.png、data/rings_<模型>.csv（逐对逐圈）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                                   # runs/E6
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO / "runs" / "E4" / "code"))

from common import HW, MODELS, apply, checkpoints, fit_gt, key  # noqa: E402

RAW = Path(os.environ.get("E6_RAW", RUN / "raw"))
OUT = RUN / "extra"
RINGS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, np.inf)]
RING_NAMES = ["0–16", "16–32", "32–64", "64–128", "≥128"]
VARIANTS = ["nearest", "own"]                       # 主口径：到最近的坑；副口径：到各自的坑
N_FIT, DRAWS, THR, SANE = 40, 20, 3.0, 20.0
R_NB, MIN_NB = 12.0, 3                              # 同 E3：LoFTR 局部对应的邻域半径与最少点数
ORDER = ["loftr_zs", "loftr_q4", "loftr_e1", "roma_zs", "roma_m4", "roma_e1"]
G = 256
PIX = np.stack(np.meshgrid(np.arange(HW) + 0.5, np.arange(HW) + 0.5), -1).reshape(-1, 2)   # 像素中心


# ---------------------------------------------------------------- 候选对应

def bilinear(D, pts):
    """D: G×G×C，网格点在原网格坐标 2j+1；pts: N×2 (x, y)。边缘外取最近网格点。"""
    g = np.clip((pts - 1.0) / 2.0, 0, G - 1)
    x0 = np.minimum(np.floor(g[:, 0]).astype(int), G - 2)
    y0 = np.minimum(np.floor(g[:, 1]).astype(int), G - 2)
    fx, fy = (g[:, 0] - x0)[:, None], (g[:, 1] - y0)[:, None]
    return ((1 - fx) * (1 - fy) * D[y0, x0] + fx * (1 - fy) * D[y0, x0 + 1]
            + (1 - fx) * fy * D[y0 + 1, x0] + fx * fy * D[y0 + 1, x0 + 1])


class Pair:
    """一对的候选对应。src: N×2 光学；dst(idx) → SAR；quality(idx) → RoMa certainty（LoFTR 为 None）。"""

    def __init__(self, family, M=None, D=None, C=None):
        self.family = family
        if family == "loftr":
            self.src, self._dst = M[:, :2].astype(np.float64), M[:, 2:4].astype(np.float64)
        else:
            self.src, self.D, self.C = PIX, D.astype(np.float64), C.astype(np.float64)

    def dst(self, idx):
        if self.family == "loftr":
            return self._dst[idx]
        return self.src[idx] + bilinear(self.D, self.src[idx])

    def cert(self, idx):
        return None if self.family == "loftr" else bilinear(self.C[..., None], self.src[idx])[:, 0]


# ---------------------------------------------------------------- 拟合

def fit(src, dst, seed):
    """RANSAC（3 px，不细化）→ 内点上普通最小二乘。返回 (A 或 None, 内点率)。"""
    import cv2

    if len(src) < 3:
        return None, 0.0
    cv2.setRNGSeed(seed)
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=THR,
                                  maxIters=10000, confidence=0.99999, refineIters=0)
    if A is None or inl is None or int(inl.sum()) < 3:
        return None, 0.0
    m = inl.ravel().astype(bool)
    A = np.linalg.lstsq(np.c_[src[m], np.ones(m.sum())], dst[m], rcond=None)[0].T
    return A, float(m.mean())


def allocate(avail, n, rng):
    """各坑平分 n 个名额，不够的坑把差额平分给还有余量的坑；凑不满返回 None。"""
    avail = np.asarray(avail)
    if avail.sum() < n:
        return None
    q, rem = np.zeros(len(avail), int), n
    while rem > 0:
        el = np.flatnonzero(avail > q)
        share = rem // len(el)
        if share == 0:
            q[rng.choice(el, rem, replace=False)] += 1
            break
        add = np.minimum(share, avail[el] - q[el])
        q[el] += add
        rem -= int(add.sum())
    return q


def members(src, o, variant):
    """每个坑在每一圈的候选下标：list[ring][crater] → idx。"""
    d = np.linalg.norm(src[:, None, :] - o[None], axis=2)            # N×K
    out = []
    if variant == "nearest":
        own, dn = d.argmin(1), d.min(1)
        for lo, hi in RINGS:
            inr = (dn >= lo) & (dn < hi)
            out.append([np.flatnonzero(inr & (own == k)) for k in range(len(o))])
    else:
        for lo, hi in RINGS:
            out.append([np.flatnonzero((d[:, k] >= lo) & (d[:, k] < hi)) for k in range(len(o))])
    return out


def draw(groups, rng):
    """从各坑的候选里按 allocate 的名额抽样，同一次内不重复。凑不满返回 None。"""
    q = allocate([len(g) for g in groups], N_FIT, rng)
    if q is None:
        return None
    picks = []
    for g, k in zip(groups, q):
        if k == 0:
            continue
        cand = g[~np.isin(g, np.concatenate(picks))] if picks else g
        picks.append(rng.choice(cand, min(k, len(cand)), replace=False))
    return np.concatenate(picks)


def score(A, o, s):
    """一次拟合在全部标注点上的误差；粗错返回 None。"""
    if A is None:
        return None
    e = apply(A, o) - s
    return None if np.abs(e.mean(0)).max() > SANE else e


def summarize(errs):
    """一组有效抽样的误差（各为 K×2）→ 逐点在抽样之间平均的 dx²、|e|²、|dx|、|e|（各 K 个）。"""
    E = np.stack(errs)                                                  # D×K×2
    return {"dx2": (E[..., 0] ** 2).mean(0), "e2": (E ** 2).sum(-1).mean(0),
            "adx": np.abs(E[..., 0]).mean(0), "ae": np.linalg.norm(E, axis=-1).mean(0)}


def repeated(P, idx_fn, o, s, seed0):
    """DRAWS 次抽样拟合。idx_fn(rng) → 下标或 None。返回 (逐点汇总 或 None, 粗错率, 内点率均值, 可用与否)。"""
    errs, gross, inl = [], 0, []
    for t in range(DRAWS):
        rng = np.random.default_rng(seed0 + t)
        idx = idx_fn(rng)
        if idx is None:
            return None, np.nan, np.nan, False
        A, r = fit(P.src[idx], P.dst(idx), seed0 + t)
        inl.append(r)
        e = score(A, o, s)
        if e is None:
            gross += 1
        else:
            errs.append(e)
    return (summarize(errs) if errs else None), gross / DRAWS, float(np.mean(inl)), True


def local_in_sample(P, o, s):
    """同 E3：标注点处的局部对应，普通最小二乘拟合（样本内）。"""
    if P.family == "loftr":
        loc = np.full_like(o, np.nan)
        for i, p in enumerate(o):
            nb = np.linalg.norm(P.src - p, axis=1) <= R_NB
            if nb.sum() >= MIN_NB:
                loc[i] = p + np.median(P._dst[nb] - P.src[nb], axis=0)
    else:
        loc = o + bilinear(P.D, o)
    ok = np.isfinite(loc).all(1)
    ok[ok] = np.linalg.norm(loc[ok] - s[ok], axis=1) < SANE
    if ok.sum() < 4:
        return None, int(ok.sum())
    return np.linalg.lstsq(np.c_[o[ok], np.ones(ok.sum())], loc[ok], rcond=None)[0].T, int(ok.sum())


# ---------------------------------------------------------------- 逐对

def one_pair(args):
    import cv2
    cv2.setNumThreads(1)
    model, i, pair, A0 = args
    family = MODELS[model]["family"]
    o, s = checkpoints(pair)
    rec = {"pair": pair, "n_pts": len(o)}
    e0 = score(None if np.isnan(A0).any() else A0, o, s)
    if e0 is None:
        rec["excluded"] = True
        return rec
    rec["excluded"] = False
    if family == "loftr":
        with np.load(RAW / model / "matches.npz") as z:
            P = Pair("loftr", M=z[f"M__{key(pair)}"])
    else:
        D = np.load(RAW / model / "D.npy", mmap_mode="r")[i]
        C = np.load(RAW / model / "C.npy", mmap_mode="r")[i]
        P = Pair("roma", D=np.asarray(D), C=np.asarray(C))
    seed = 1000 * i
    one = lambda e: summarize([e]) if e is not None else None
    ref = {"model": one(e0), "label_fit": one(score(fit_gt(o, s), o, s))}
    Af, _ = fit(P.src, P.dst(np.arange(len(P.src))), seed)
    ref["full_all"] = one(score(Af, o, s))
    Al, rec["n_local"] = local_in_sample(P, o, s)
    ref["local_in_sample"] = one(score(Al, o, s))
    allidx = np.arange(len(P.src))
    r40 = repeated(P, lambda rng: rng.choice(allidx, N_FIT, replace=False) if len(allidx) >= N_FIT else None,
                   o, s, seed + 500)
    ref["full_40"] = r40[0]
    rec["ref"] = ref
    rec["full_40_gross"] = r40[1]
    # 各圈
    area = members(PIX, o, "nearest")                                 # 圈的面积（像素数），用于密度
    rec["rings"] = {}
    for v in VARIANTS:
        mem = members(P.src, o, v)
        rows = []
        for j in range(len(RINGS)):
            res, gross, inl, ok = repeated(P, lambda rng, g=mem[j]: draw(g, rng), o, s, seed + 100 * (j + 1) + (50 if v == "own" else 0))
            row = {"ok": ok, "res": res, "gross": gross, "inlier": inl, "n_cand": int(sum(len(g) for g in mem[j]))}
            if v == "nearest":
                idx = np.concatenate(mem[j])
                a = sum(len(g) for g in area[j])
                row["area"] = int(a)
                row["density"] = 1000.0 * len(idx) / a if a else np.nan
                c = P.cert(idx)
                row["cert"] = float(np.mean(c)) if c is not None and len(idx) else np.nan
            rows.append(row)
        rec["rings"][v] = rows
    return rec


# ---------------------------------------------------------------- 汇总

def pool_points(recs, get):
    """把若干对的逐点汇总拼起来；get(rec) → summarize 的结果。"""
    parts = [get(r) for r in recs]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("dx2", "e2", "adx", "ae")}


def stats(S):
    return {"ms_x": float(S["dx2"].mean()), "med_x": float(np.median(S["adx"])),
            "ms_2d": float(S["e2"].mean()), "med_2d": float(np.median(S["ae"]))}


def boot(recs, gets, n=1000, seed=0):
    """按对自助抽样：gets 为若干个 get，返回每个的 x 向均方误差样本（n 个）。"""
    rng = np.random.default_rng(seed)
    per = [[(g(r)["dx2"].sum(), len(g(r)["dx2"])) for r in recs] for g in gets]
    per = [np.array(p) for p in per]
    out = np.zeros((len(gets), n))
    for b in range(n):
        ix = rng.integers(0, len(recs), len(recs))
        for k, p in enumerate(per):
            out[k, b] = p[ix, 0].sum() / p[ix, 1].sum()
    return out


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def analyse_model(model, recs):
    kept = [r for r in recs if not r["excluded"]]
    res = {"n_pairs": len(recs), "n_excluded": len(recs) - len(kept)}
    for v in VARIANTS:
        full = [r for r in kept if all(x["ok"] and x["res"] is not None for x in r["rings"][v])]
        sub = [r for r in full if all(r["ref"][k] is not None for k in ("full_all", "full_40", "local_in_sample", "label_fit"))]
        R = {"n_pairs_all_rings": len(full), "n_pairs_with_refs": len(sub), "n_points": int(sum(r["n_pts"] for r in full))}
        gets = [lambda r, j=j: r["rings"][v][j]["res"] for j in range(len(RINGS))]
        B = boot(full, gets)
        R["rings"] = []
        for j in range(len(RINGS)):
            st = stats(pool_points(full, gets[j]))
            st["ms_x_ci"] = ci(B[j])
            st["diff_vs_ring0_ms_x"] = float(st["ms_x"] - stats(pool_points(full, gets[0]))["ms_x"])
            st["diff_vs_ring0_ci"] = ci(B[j] - B[0])
            st["gross_rate"] = float(np.mean([r["rings"][v][j]["gross"] for r in full]))
            st["inlier_rate"] = float(np.mean([r["rings"][v][j]["inlier"] for r in full]))
            if v == "nearest":
                st["density_per_1000px2"] = float(np.nanmean([r["rings"][v][j]["density"] for r in full]))
                st["cert_mean"] = float(np.nanmean([r["rings"][v][j]["cert"] for r in full]))
            R["rings"].append(st)
        # 每圈单独看：凡这一圈可用的对都算（不要求五圈都有效）
        R["rings_any"] = []
        for j in range(len(RINGS)):
            av = [r for r in kept if r["rings"][v][j]["ok"] and r["rings"][v][j]["res"] is not None]
            R["rings_any"].append({"n_pairs": len(av), **(stats(pool_points(av, gets[j])) if av else {}),
                                   "n_unfilled": sum(1 for r in kept if not r["rings"][v][j]["ok"])})
        # 参照线：在有全部参照的对上（与各圈同一批点再算一遍，便于直接比）
        R["refs"] = {}
        names = ["model", "full_all", "full_40", "local_in_sample", "label_fit"]
        Bref = boot(sub, [lambda r, k=k: r["ref"][k] for k in names] + gets)
        for n_, k in enumerate(names):
            st = stats(pool_points(sub, lambda r: r["ref"][k]))
            st["ms_x_ci"] = ci(Bref[n_])
            R["refs"][k] = st
        R["rings_on_ref_pairs"] = [{**stats(pool_points(sub, gets[j])), "ms_x_ci": ci(Bref[len(names) + j])}
                                   for j in range(len(RINGS))]
        R["full_40_gross_rate"] = float(np.mean([r["full_40_gross"] for r in sub]))
        res[v] = R
    return res


def write_csv(model, recs):
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    with open(OUT / "data" / f"rings_{model}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "variant", "ring", "ok", "n_cand", "gross", "inlier", "ms_x", "ms_2d"])
        for r in recs:
            if r["excluded"]:
                w.writerow([r["pair"], "", "", "excluded", "", "", "", "", ""])
                continue
            for v in VARIANTS:
                for j, x in enumerate(r["rings"][v]):
                    m = x["res"]
                    w.writerow([r["pair"], v, RING_NAMES[j], int(x["ok"]), x["n_cand"], f"{x['gross']:.3f}",
                                f"{x['inlier']:.3f}", f"{m['dx2'].mean():.4f}" if m else "",
                                f"{m['e2'].mean():.4f}" if m else ""])


# ---------------------------------------------------------------- 图

def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


REF_STYLE = {"model": ("模型仿射", "k", "-"), "full_all": ("全图对应拟合", "tab:gray", "--"),
             "full_40": (f"全图随机 {N_FIT} 点拟合", "tab:purple", ":"),
             "local_in_sample": ("标注点处局部对应拟合（样本内）", "tab:green", "-."),
             "label_fit": ("标注拟合的最佳仿射", "tab:red", (0, (1, 3)))}


def plot(summary):
    plt = setup_plt()
    models = [m for m in ORDER if m in summary["models"]]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    x = np.arange(len(RINGS))
    for ax, m in zip(axs.ravel(), models):
        R = summary["models"][m]
        for v, st, lab in (("nearest", "-o", "按到最近坑的距离分圈"), ("own", "--s", "按到各自坑的距离分圈")):
            y = [r["ms_x"] for r in R[v]["rings"]]
            lo = [r["ms_x_ci"][0] for r in R[v]["rings"]]
            hi = [r["ms_x_ci"][1] for r in R[v]["rings"]]
            ax.plot(x, y, st, color="tab:blue" if v == "nearest" else "tab:cyan", label=lab, ms=4)
            if v == "nearest":
                ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.15, lw=0)
        for k, (lab, c, ls) in REF_STYLE.items():
            ax.axhline(R["nearest"]["refs"][k]["ms_x"], color=c, ls=ls, lw=1.2, label=lab)
        ax.set_title(f"{MODELS[m]['name']}（{R['nearest']['n_pairs_all_rings']} 对）", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(RING_NAMES)
        ax.set_xlabel("拟合点到陨石坑的距离（px）")
    for ax in axs[:, 0]:
        ax.set_ylabel("标注点 x 向均方误差（px²）")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "rings_x.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for m in models:
        R = summary["models"][m]["nearest"]["rings"]
        c = {"zs": "tab:blue", "q4": "tab:orange", "m4": "tab:orange", "e1": "tab:red"}[m.split("_")[1]]
        ls = "-" if m.startswith("loftr") else "--"
        if m.startswith("loftr"):
            axs[0].plot(x, [r["density_per_1000px2"] for r in R], "o" + ls, color=c, label=MODELS[m]["name"], ms=4)
        else:
            axs[1].plot(x, [r["cert_mean"] for r in R], "o" + ls, color=c, label=MODELS[m]["name"], ms=4)
        axs[2].plot(x, [r["inlier_rate"] for r in R], "o" + ls, color=c, label=MODELS[m]["name"], ms=4)
    for ax, t in zip(axs, ["LoFTR 匹配密度（每 1000 px²）", "RoMa certainty 均值", "圈内拟合的 RANSAC 内点率"]):
        ax.set_title(t, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(RING_NAMES)
        ax.set_xlabel("到最近陨石坑的距离（px）")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rings_quality.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"settings": {"rings_px": [[lo, None if np.isinf(hi) else hi] for lo, hi in RINGS], "n_fit": N_FIT,
                            "draws": DRAWS, "ransac_px": THR, "sane_px": SANE, "variants": VARIANTS},
               "models": {}}
    for m in a.models:
        pairs = json.loads((RAW / m / "pairs.json").read_text(encoding="utf-8"))[: a.limit or None]
        A = np.load(RAW / m / "A.npy")
        with Pool(a.workers) as pool:
            recs = pool.map(one_pair, [(m, i, p, A[i]) for i, p in enumerate(pairs)], chunksize=4)
        write_csv(m, recs)
        summary["models"][m] = analyse_model(m, recs)
        print(m, json.dumps({v: [round(r["ms_x"], 2) for r in summary["models"][m][v]["rings"]] for v in VARIANTS}),
              {k: round(v["ms_x"], 2) for k, v in summary["models"][m]["nearest"]["refs"].items()}, flush=True)
    (OUT / "rings.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    plot(summary)


if __name__ == "__main__":
    main()
