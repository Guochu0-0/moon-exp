"""读 infer.py / occlude.py 的产物，算三项统计、画图，写 extra/stats.json 与 extra/*.png。

    /opt/envs/loftr/bin/python runs/E4/code/analyze.py      （工作台环境没有 matplotlib；只用 CPU）

所有重要性图统一成 32×32 格（每格 16 px）、和为 1，分光学、SAR 两个坐标系：
- 主要来源：LoFTR 为 4 个 cross 层「被关注总量」的平均；RoMa 为 certainty。
- 内点分布：RANSAC 内点落在各格的个数。
- 遮挡：每块的仿射变化量减去该对噪声底的均值、截到 ≥ 0；遮后失败的块记 50 px。只有 60 对。SAR 侧按标注拟合仿射搬过去。
重要区域 = 按重要性从高到低取格，累计到一半重要性为止；集中程度 = 这些格占的面积比例（均匀分布为 0.5）。
"""
from __future__ import annotations

import json
import math

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import (COUPLES, HW, LAYERS, MODELS, RUN, Data, DATA, apply, checkpoints, fit_gt, key,  # noqa: E402
                    ref_preds, val_pairs)
from moonlib import inputs  # noqa: E402

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 服务器上没有中文字体，放了一份在 gpfs
try:
    matplotlib.font_manager.fontManager.addfont(FONT)
except (OSError, FileNotFoundError):
    pass
plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
N = 32
CELL = HW // N
FAIL_DELTA = 50.0
OUT = RUN / "extra"
FRAMES = ("opt", "sar")
FR_ZH = {"opt": "光学", "sar": "SAR"}
SRC_ZH = {"primary": "注意力 / certainty", "inliers": "内点分布", "occlusion": "遮挡敏感性"}
PROP_ZH = {"bright": "亮度", "dark": "暗区占比", "texture": "纹理强弱"}
RNG = np.random.default_rng(0)
EN = {"最高": "high", "中位": "mid", "最低": "low"}


# ---------- 统计小工具（不依赖 scipy） ----------

def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    xs = x[o]
    for v in np.unique(xs[1:][np.diff(xs) == 0]):          # 并列取平均秩
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, n = x[ok], y[ok], int(ok.sum())
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    rho = float(np.corrcoef(rank(x), rank(y))[0, 1])
    t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    return {"rho": rho, "p": float(math.erfc(abs(t) / math.sqrt(2))), "n": n}   # 正态近似，n 大时足够


def partial_spearman(x, y, z):
    """控制 z 后 x 与 y 的秩偏相关（由三个 Spearman 系数算出），p 用 n−3 自由度的正态近似。"""
    a, b, c = spearman(x, y), spearman(x, z), spearman(y, z)
    rxy, rxz, ryz, n = a["rho"], b["rho"], c["rho"], a["n"]
    if not (np.isfinite(rxy) and np.isfinite(rxz) and np.isfinite(ryz)) or n < 5:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    r = (rxy - rxz * ryz) / math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    t = r * math.sqrt((n - 3) / max(1e-12, 1 - r ** 2))
    return {"rho": float(r), "p": float(math.erfc(abs(t) / math.sqrt(2))), "n": n}


def auc5(e):
    e = np.asarray(e, float)
    return float(np.clip(1.0 - e / 5.0, 0.0, None).mean())      # 同 workbench.protocol.auc，∞ 记 0


def wilcoxon(d):
    """符号秩检验（正态近似），返回 p。"""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n < 6:
        return float("nan")
    r = rank(np.abs(d))
    w = r[d > 0].sum()
    mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return float(math.erfc(abs(w - mu) / sd / math.sqrt(2)))


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    q = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]), "mean": float(v.mean())}


# ---------- 图像属性 ----------

def to_grid(img512, n=N):
    return cv2.resize(np.asarray(img512, np.float32), (n, n), interpolation=cv2.INTER_AREA)


def properties(pair):
    d = Data(DATA)
    opt = inputs.optical_div255(d.optical("val", pair))
    sar_raw = d.sar("val", pair)
    sar = inputs.sar_p2p98(sar_raw)
    db = inputs.sar_db(sar_raw)

    def tex(x):
        g = cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.0)
        m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        return m / max(m.mean(), 1e-9)

    # 暗区：低于该图自身第 10 百分位的像素。光学 patch 对比度很低，几乎没有真正的阴影（低于中位数 0.7 倍的像素不到万分之一），
    # 所以用相对定义；SAR 按 dB 同样处理。
    return {"opt": {"bright": to_grid(opt), "dark": to_grid(opt < np.percentile(opt, 10)), "texture": to_grid(tex(opt))},
            "sar": {"bright": to_grid(sar), "dark": to_grid(db < np.percentile(db, 10)), "texture": to_grid(tex(sar))}}


def images(pair):
    d = Data(DATA)
    return {"opt": inputs.optical_div255(d.optical("val", pair)), "sar": inputs.sar_p2p98(d.sar("val", pair))}


# ---------- 重要性图 ----------

def norm(m):
    m = np.clip(np.asarray(m, np.float64), 0, None)
    s = m.sum()
    return m / s if s > 0 else None


def hist(pts, n=N):
    h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=n, range=[[0, HW], [0, HW]])
    return h


def warp_to_sar(m_opt, G):
    """光学坐标系的格图 → SAR 坐标系（按标注拟合的仿射），在 512 网格上做再降回 N×N。"""
    big = cv2.resize(np.asarray(m_opt, np.float32), (HW, HW), interpolation=cv2.INTER_NEAREST)
    M = np.c_[G[:, :2], G[:, :2] @ [0.5, 0.5] + G[:, 2] - 0.5]          # 像素下标约定
    return to_grid(cv2.warpAffine(big, M, (HW, HW), flags=cv2.INTER_LINEAR, borderValue=0))


def load_model(name, pairs):
    z = np.load(RUN / "raw" / name / "val.npz")
    fam = MODELS[name]["family"]
    out = {"A": {}, "primary": {f: {} for f in FRAMES}, "inliers": {f: {} for f in FRAMES}, "layers": {}}
    cross = [i for i, t in enumerate(LAYERS) if t == "cross"]
    for p in pairs:
        k = key(p)
        A = z[f"A__{k}"]
        out["A"][p] = None if np.isnan(A).any() else A
        imp = z[f"imp__{k}"].astype(np.float32)
        inl = z[f"inl__{k}"]
        for fi, f in enumerate(FRAMES):
            if fam == "loftr":
                out["primary"][f][p] = norm(to_grid(imp[cross, fi].mean(0)))
            else:
                out["primary"][f][p] = norm(to_grid(imp[fi]))
            out["inliers"][f][p] = norm(hist(inl[:, 2 * fi:2 * fi + 2])) if len(inl) >= 3 else None
        if fam == "loftr":
            out["layers"][p] = [[norm(to_grid(imp[l, fi])) for fi in range(2)] for l in range(len(LAYERS))]
    return out


def load_occlusion(name):
    res, info = {fr: {} for fr in FRAMES}, {}
    for f in sorted((RUN / "raw" / name).glob("occl*.npz")):     # 整份 occl.npz，或分份 occl_<k>of<n>.npz
        _occl_file(np.load(f), res, info)
    return res, info


def _occl_file(z, res, info):
    for i, p in enumerate(z["pairs"].tolist()):
        if i >= int(z["done"]) or np.isnan(z["A0"][i]).any():
            continue
        d = z["delta"][i].astype(np.float64)
        fails = np.isnan(d)
        d[fails] = FAIL_DELTA
        floor = float(np.nanmean(z["noise"][i]))
        m16 = np.clip(d - floor, 0, None)
        m = np.kron(m16, np.ones((N // 16, N // 16)))
        res["opt"][p] = norm(m)
        res["sar"][p] = norm(warp_to_sar(m, z["G"][i]))
        info[p] = {"noise_floor": floor, "max_delta": float(np.nanmax(z["delta"][i])) if (~fails).any() else None,
                   "n_fail_blocks": int(fails.sum()), "median_delta": float(np.nanmedian(z["delta"][i])),
                   "noise_floor_ransac": float(np.nanmean(z["noise_ransac"][i])),
                   "median_delta_ransac": float(np.nanmedian(z["delta_ransac"][i])),
                   "max_delta_ransac": float(np.nanmax(z["delta_ransac"][i])),
                   "share_blocks_above_noise": float(np.mean(d > 2 * floor))}


def region(m):
    """累计到一半重要性的格（布尔 N×N）与其面积比例。"""
    f = m.ravel()
    o = np.argsort(-f, kind="mergesort")
    k = int(np.searchsorted(np.cumsum(f[o]), 0.5) + 1)
    r = np.zeros(f.size, bool)
    r[o[:k]] = True
    return r.reshape(m.shape), k / f.size


def pair_error(A, o, s):
    return math.inf if A is None else float(np.linalg.norm(apply(A, o) - s, axis=1).mean())


# ---------- 主流程 ----------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = val_pairs()
    names = [m for m in MODELS if (RUN / "raw" / m / "val.npz").exists()]
    print("models:", names, flush=True)
    cps = {p: checkpoints(p) for p in pairs}
    G = {p: fit_gt(*cps[p]) for p in pairs}
    props = {p: properties(p) for p in pairs}
    print("properties done", flush=True)
    D = {m: load_model(m, pairs) for m in names}
    for m in names:
        D[m]["occlusion"], D[m]["occl_info"] = load_occlusion(m)
    err = {m: {p: pair_error(D[m]["A"][p], *cps[p]) for p in pairs} for m in names}

    # 到最近标注点的距离（格中心，光学坐标系）
    c = (np.arange(N) + 0.5) * CELL
    cx, cy = np.meshgrid(c, c)
    dist = {p: np.min(np.hypot(cx[..., None] - cps[p][0][:, 0], cy[..., None] - cps[p][0][:, 1]), axis=-1)
            for p in pairs}

    stats = {"models": {m: MODELS[m]["name"] for m in names}, "n_pairs": len(pairs), "concentration": {},
             "properties": {}, "distance": {}, "agreement": {}, "layers": {}, "occlusion": {}, "sanity": {}}
    sources = ("primary", "inliers", "occlusion")

    # 0. 复现核对、遮挡噪声底
    for m in names:
        rows = [json.loads(l) for l in (RUN / "raw" / m / "val.jsonl").read_text(encoding="utf-8").splitlines()]
        dv = np.array([r["diff_vs_ref_px"] for r in rows], float)
        ref = ref_preds(m)
        stats["sanity"][m] = {"auc5_rerun": auc5([err[m][p] for p in pairs]),
                              "auc5_recorded": auc5([pair_error(ref.get(p), *cps[p]) for p in pairs]),
                              "diff_vs_ref_px": summ(dv), "share_below_0.1px": float(np.mean(dv[np.isfinite(dv)] < 0.1)),
                              "fails": int(sum(r["fail"] for r in rows)), "ref_fails": int(sum(r["ref_fail"] for r in rows)),
                              "sec_per_pair": float(np.mean([r["sec"] for r in rows]))}
        info = D[m]["occl_info"]
        if info:
            stats["occlusion"][m] = {k: summ([v[k] for v in info.values() if v[k] is not None])
                                     for k in ("noise_floor", "max_delta", "median_delta", "n_fail_blocks", "noise_floor_ransac",
                                               "median_delta_ransac", "max_delta_ransac", "share_blocks_above_noise")}

    # 1a. 集中程度
    for m in names:
        for s in sources:
            for f in FRAMES:
                v = [region(x)[1] for x in D[m][s][f].values() if x is not None]
                stats["concentration"][f"{m}/{s}/{f}"] = summ(v)
        if MODELS[m]["family"] == "loftr":
            stats["layers"][m] = {f"{l}_{t}_{f}": summ([region(D[m]["layers"][p][l][fi])[1] for p in pairs
                                                        if D[m]["layers"][p][l][fi] is not None])
                                  for l, t in enumerate(LAYERS) for fi, f in enumerate(FRAMES)}

    # 1b. 重要区域与其余区域的图像属性
    contrast = {}
    for m in names:
        for s in sources:
            for f in FRAMES:
                for pr in PROP_ZH:
                    v = []
                    for p, x in D[m][s][f].items():
                        if x is None:
                            continue
                        r, _ = region(x)
                        P = props[p][f][pr]
                        a, b = P[r].mean(), P[~r].mean()
                        v.append(math.log(a / b) if pr == "texture" else a - b)
                    v = np.array(v)
                    contrast[(m, s, f, pr)] = v
                    stats["properties"][f"{m}/{s}/{f}/{pr}"] = {**summ(v), "share_positive": float(np.mean(v > 0)) if len(v) else None,
                                                                "wilcoxon_p": wilcoxon(v)}

    # 2. 依赖区域离标注点越远，误差是否越大
    dwx = {}
    for m in names:
        for s in sources:
            ps = [p for p, x in D[m][s]["opt"].items() if x is not None]
            dw = np.array([(D[m][s]["opt"][p] * dist[p]).sum() for p in ps])
            du = np.array([dist[p].mean() for p in ps])
            e = np.array([err[m][p] for p in ps])
            dwx[(m, s)] = (ps, dw, du, e)
            stats["distance"][f"{m}/{s}"] = {"weighted": summ(dw), "uniform": summ(du), "ratio": summ(dw / du),
                                             "rho_weighted_vs_error": spearman(dw, e),
                                             "rho_uniform_vs_error": spearman(du, e),
                                             "rho_weighted_vs_error_given_uniform": partial_spearman(dw, e, du),
                                             "rho_excess_vs_error": spearman(dw - du, e),
                                             "rho_ratio_vs_error": spearman(dw / du, e)}

    # 3. LoFTR 与 RoMa 是否看同样的地方
    agree = {}
    for a, b in COUPLES:
        if a not in D or b not in D:
            continue
        for s in sources:
            for f in FRAMES:
                ps = [p for p in pairs if D[a][s][f].get(p) is not None and D[b][s][f].get(p) is not None]
                if len(ps) < 6:
                    continue
                corr = lambda x, y: spearman(x.ravel(), y.ravel())["rho"]
                same = np.array([corr(D[a][s][f][p], D[b][s][f][p]) for p in ps])
                other = []
                for i, p in enumerate(ps):
                    for j in RNG.choice([j for j in range(len(ps)) if j != i], min(5, len(ps) - 1), replace=False):
                        other.append(corr(D[a][s][f][p], D[b][s][f][ps[j]]))
                other = np.array(other)
                agree[(a, b, s, f)] = (ps, same, other)
                stats["agreement"][f"{a}~{b}/{s}/{f}"] = {"same_pair": summ(same), "different_pair": summ(other),
                                                         "share_same_above_diff_median": float(np.mean(same > np.nanmedian(other)))}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("stats written", flush=True)

    figures(names, D, contrast, dwx, agree, props, cps, err, pairs)


# ---------- 图 ----------

def box(ax, data, labels, title):
    data = [np.asarray(d)[np.isfinite(d)] for d in data]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=30)


def figures(names, D, contrast, dwx, agree, props, cps, err, pairs):
    zh = [MODELS[m]["name"] for m in names]
    sources = ("primary", "inliers", "occlusion")

    # 图 1：集中程度
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, s in zip(axs, sources):
        box(ax, [[region(x)[1] for x in D[m][s]["opt"].values() if x is not None] for m in names], zh,
            f"{SRC_ZH[s]}（光学坐标系）")
        ax.axhline(0.5, color="gray", lw=0.6, ls="--")
    axs[0].set_ylabel("占一半重要性的面积比例")
    fig.tight_layout()
    fig.savefig(OUT / "concentration.png", dpi=130)
    plt.close(fig)

    # 图 2：重要区域与其余区域的属性差（注意力 / certainty 与遮挡）
    for s in ("primary", "occlusion"):
        fig, axs = plt.subplots(2, 3, figsize=(14, 7))
        for i, f in enumerate(FRAMES):
            for j, pr in enumerate(PROP_ZH):
                box(axs[i, j], [contrast[(m, s, f, pr)] for m in names], zh,
                    f"{FR_ZH[f]}：{PROP_ZH[pr]}（重要区域 − 其余{'，对数比' if pr == 'texture' else ''}）")
                axs[i, j].axhline(0, color="gray", lw=0.6, ls="--")
        fig.suptitle(f"重要性来源：{SRC_ZH[s]}")
        fig.tight_layout()
        fig.savefig(OUT / f"properties_{s}.png", dpi=130)
        plt.close(fig)

    # 图 3：加权距离与误差
    fig, axs = plt.subplots(len(sources), len(names), figsize=(3 * len(names), 2.8 * len(sources)), squeeze=False)
    for i, s in enumerate(sources):
        for j, m in enumerate(names):
            ax = axs[i, j]
            ps, dw, du, e = dwx[(m, s)]
            ok = np.isfinite(e)
            if ok.sum() < 6:
                ax.axis("off")
                continue
            ax.scatter(dw[ok], e[ok], s=4, alpha=0.4)
            q = np.quantile(dw[ok], np.linspace(0, 1, 6))
            mids, meds = [], []
            for lo, hi in zip(q[:-1], q[1:]):
                sel = ok & (dw >= lo) & (dw <= hi)
                mids.append(dw[sel].mean())
                meds.append(np.mean(e[sel]))
            ax.plot(mids, meds, "r-o", ms=3)
            ax.set_yscale("log")
            r0, r1 = spearman(dw, e), partial_spearman(dw, e, du)
            ax.set_title(f"{MODELS[m]['name']}｜{SRC_ZH[s]}\nρ={r0['rho']:.2f}，控制均匀距离后 {r1['rho']:.2f}（n={r0['n']}）",
                         fontsize=8)
            if j == 0:
                ax.set_ylabel("该对误差 (px)")
            if i == len(sources) - 1:
                ax.set_xlabel("重要性加权的到最近标注点距离 (px)")
    fig.tight_layout()
    fig.savefig(OUT / "distance_error.png", dpi=130)
    plt.close(fig)

    # 图 4：LoFTR 与 RoMa 的一致程度
    fig, axs = plt.subplots(len(sources), len(COUPLES), figsize=(4 * len(COUPLES), 2.8 * len(sources)), squeeze=False)
    for i, s in enumerate(sources):
        for j, (a, b) in enumerate(COUPLES):
            ax = axs[i, j]
            if (a, b, s, "opt") not in agree:
                ax.axis("off")
                continue
            _, same, other = agree[(a, b, s, "opt")]
            bins = np.linspace(-0.6, 1, 33)
            ax.hist(other[np.isfinite(other)], bins, density=True, alpha=0.5, label="不同对")
            ax.hist(same[np.isfinite(same)], bins, density=True, alpha=0.5, label="同一对")
            ax.set_title(f"{MODELS[a]['name']} vs {MODELS[b]['name']}｜{SRC_ZH[s]}", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.supxlabel("两张重要性图的秩相关（光学坐标系，32×32 格）")
    fig.tight_layout()
    fig.savefig(OUT / "agreement.png", dpi=130)
    plt.close(fig)

    examples(names, D, props, cps, err, pairs, dwx, agree)


def overlay(ax, img, m, pts=None, title=""):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    if m is not None:
        big = cv2.resize(np.asarray(m, np.float32), (HW, HW), interpolation=cv2.INTER_LINEAR)
        ax.imshow(big, cmap="inferno", alpha=0.5, extent=(0, HW, HW, 0))
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], s=14, facecolors="none", edgecolors="cyan", lw=0.8)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def pick3(score: dict):
    ps = sorted((v, p) for p, v in score.items() if np.isfinite(v))
    if not ps:
        return []
    return [("最高", ps[-1][1]), ("中位", ps[len(ps) // 2][1]), ("最低", ps[0][1])]


def example_grid(pair, names, D, props, cps, err, src, path, head):
    fig, axs = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.6), squeeze=False)
    img = images(pair)
    for j, m in enumerate(names):
        for i, f in enumerate(FRAMES):
            x = D[m][src][f].get(pair)
            reg = f"，一半重要性占 {region(x)[1]:.0%} 面积" if x is not None else ""
            overlay(axs[i, j], img[f], x, cps[pair][i],
                    f"{MODELS[m]['name']}（{FR_ZH[f]}）\n误差 {err[m][pair]:.1f} px{reg}" if i == 0 else f"{FR_ZH[f]}{reg}")
    fig.suptitle(f"{head}｜{SRC_ZH[src]}；青色圈为标注点", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def examples(names, D, props, cps, err, pairs, dwx, agree):
    ex = OUT / "examples"
    ex.mkdir(exist_ok=True)
    picked = {}
    # 第 1 项：主要来源的集中程度（6 个模型平均）
    sc = {p: np.mean([region(D[m]["primary"]["opt"][p])[1] for m in names if D[m]["primary"]["opt"].get(p) is not None])
          for p in pairs}
    for lab, p in pick3(sc):
        f = f"concentration_{EN[lab]}.png"
        example_grid(p, names, D, props, cps, err, "primary", ex / f, f"集中程度{lab}（面积比例均值 {sc[p]:.2f}）：{p}")
        picked[f] = p
    # 第 2 项：加权距离相对均匀分布的比值（6 个模型平均）
    rat = {}
    for m in names:
        ps, dw, du, _ = dwx[(m, "primary")]
        for p, a, b in zip(ps, dw, du):
            rat.setdefault(p, []).append(a / b)
    sc = {p: float(np.mean(v)) for p, v in rat.items()}
    for lab, p in pick3(sc):
        f = f"distance_{EN[lab]}.png"
        example_grid(p, names, D, props, cps, err, "primary", ex / f,
                     f"加权距离 / 均匀分布距离{lab}（{sc[p]:.2f}）：{p}")
        picked[f] = p
    # 第 3 项：LoFTR 与 RoMa 内点分布的相关（三组配对平均）
    acc = {}
    for (a, b, s, fr), (ps, same, _) in agree.items():
        if s == "inliers" and fr == "opt":
            for p, v in zip(ps, same):
                acc.setdefault(p, []).append(v)
    sc = {p: float(np.mean(v)) for p, v in acc.items()}
    for lab, p in pick3(sc):
        f = f"agreement_{EN[lab]}.png"
        example_grid(p, names, D, props, cps, err, "inliers", ex / f, f"LoFTR 与 RoMa 内点分布相关{lab}（{sc[p]:.2f}）：{p}")
        picked[f] = p
    (ex / "picked.json").write_text(json.dumps(picked, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
