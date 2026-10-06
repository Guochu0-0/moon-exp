"""两种训练怎样改变模型依赖的区域：重做 train_change.py 的实验二。只读已有结果，只用 CPU。

    python -m workbench.launch begin runs/E5 main -- /opt/envs/loftr/bin/python runs/E5/code/regions.py
    /opt/envs/loftr/bin/python runs/E5/code/regions.py
    python -m workbench.launch end runs/E5 main ok
    （服务器上跑：E4 的原始推理只在 gpfs 主 checkout）

来源（都在光学坐标系）：LoFTR 用 RANSAC 内点与遮挡敏感性（60 对）；RoMa 用 certainty 与内点。LoFTR 的注意力图
在同一对与不同对之间几乎一样相似（E4、E5），主要是与图像无关的共同布局，这里不用；RoMa 的遮挡图被噪声主导（E4），不用。
原始推理读主 checkout 的 runs/E4/raw/<模型>/，读法与定义沿用 E4 的 analyze.py。

- 三个量（32×32 格，E4 的定义）：集中程度、纹理偏好（重要区域与其余区域纹理强度之比的对数）、距离比。
  内点的这三个量随内点个数变（个数越多越铺开），所以每对把同一网络三个模型的内点都随机抽到三者中最少的个数，
  抽 DRAWS 次取平均（点数对齐）；不抽的原值一并记下。
- 图之间的比较在 16×16 格上用 Pearson 相关（遮挡本来就是 16×16 块；内点在 32×32 上太稀）。
  - 训练前后的相似度：内点把 zero-shot 的内点随机分成不重叠的两半 A、B，训练后的模型抽同样个数，
    相似度 = corr(A, 训练后)，上限 = corr(A, B)（同一个模型、同样点数下能有多像），对照 = 与另外 5 对的训练后相关。
  - 两种训练是否改动同一些格：训练后的图对 zero-shot 的图回归取残差，比较两份残差在同一对上的相关与不同对之间的相关。
    内点：两种训练分别对 zero-shot 的不重叠两半回归（各自的噪声不共用），斜率用另一半作工具变量估计（不被噪声压低），
    训练后的图用全部内点；重复 SPLITS 次取平均。certainty 没有抽样噪声，直接回归。遮挡只有一份 zero-shot 图，直接回归。
  - 残差的可靠程度：训练后的内点也分成两半，各自得到残差，两份残差的相关按 Spearman–Brown 折算成全量；
    两份残差相关的上限约为两者可靠程度乘积的平方根。
  - 跨网络对照：LoFTR 的内点残差与 RoMa 的内点残差在同一对上的相关。
- 与误差的关系：各量的逐对变化与误差（E4 那次推理的仿射，取对数）变化的秩相关；在 E5 的 291 对公共点上，
  与「模型仿射与局部对应拟合的仿射之差」（x 向均方误差之差）变化的秩相关。

产物：runs/E5/extra/regions.json、regions_similarity.png、regions_shared_change.png。
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                                   # runs/E5
REPO = RUN.parents[1]
E3 = REPO / "runs" / "E3"
MAIN = Path(os.environ.get("MAIN", "/remote-home/xufang/YGC/moon-exp"))
sys.path.insert(0, str(REPO / "runs" / "E4" / "code"))

import analyze as e4  # noqa: E402  （E4 的读法、统计与字体设置）
from common import apply, checkpoints, key, val_pairs  # noqa: E402

plt = e4.plt
OUT = RUN / "extra"
RAW = MAIN / "runs" / "E4" / "raw"                 # 原始推理不进 git，只在主 checkout
N, CELL = e4.N, e4.CELL
G = 16                                              # 图之间比较用的格数
DRAWS, SPLITS, NDIFF = 20, 10, 5
MIN_SPLIT = 40                                      # 分两半至少要的内点数（每半 ≥ 20）
SANE = 20.0
NETS = {"loftr": ("loftr_zs", "loftr_q4", "loftr_e1"), "roma": ("roma_zs", "roma_m4", "roma_e1")}
NAME = {"loftr_zs": "LoFTR zero-shot", "loftr_q4": "LoFTR 无标注训练", "loftr_e1": "LoFTR 标注过拟合",
        "roma_zs": "RoMa zero-shot", "roma_m4": "RoMa 无标注训练", "roma_e1": "RoMa 标注过拟合"}
SOURCES = {"loftr": ("inliers", "occlusion"), "roma": ("certainty", "inliers")}
SRC_ZH = {"inliers": "内点", "occlusion": "遮挡", "certainty": "certainty"}
LOCAL = {"loftr": slice(6, 8), "roma": slice(9, 11)}   # 同 E3：LoFTR 用邻近匹配，RoMa 用稠密对应
QS = ("concentration", "texture", "distance_ratio")
RNG = np.random.default_rng(0)


def pc(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def ols_res(y, x):
    b = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    return (y - y.mean()) - b * (x - x.mean())


def iv_res(y, x, z):
    """y 对 x 回归的残差，斜率用 z（x 的独立重测）作工具变量估计：x 的测量噪声不压低斜率。"""
    b = np.cov(y, z)[0, 1] / np.cov(x, z)[0, 1]
    return (y - y.mean()) - b * (x - x.mean())


def hist(pts, n):
    h = e4.hist(pts, n).ravel()
    return h / h.sum()


def paired(before, after):
    b, a = np.asarray(before, float), np.asarray(after, float)
    ok = np.isfinite(a) & np.isfinite(b)
    b, a = b[ok], a[ok]
    return {"n": int(ok.sum()), "median_before": float(np.median(b)), "median_after": float(np.median(a)),
            "median_change": float(np.median(a - b)), "share_increase": float(np.mean(a > b)),
            "wilcoxon_p": e4.wilcoxon(a - b)}


def others(i, n, k=NDIFF):
    return RNG.choice([j for j in range(n) if j != i], min(k, n - 1), replace=False)


# ======================================================================== 读数据

def load():
    pairs = val_pairs()
    cps = {p: checkpoints(p) for p in pairs}
    D = {}
    for net, models in NETS.items():
        for m in models:
            z = np.load(RAW / m / "val.npz")
            d = {"A": {}, "inl": {}}
            if net == "roma":
                d["cert"] = {}
            for p in pairs:
                k = key(p)
                A = z[f"A__{k}"]
                d["A"][p] = None if np.isnan(A).any() else A
                d["inl"][p] = z[f"inl__{k}"][:, :2].astype(np.float64)
                if net == "roma":
                    d["cert"][p] = z[f"imp__{k}"][0].astype(np.float32)        # 光学侧 certainty，64×64
            if net == "loftr":
                d["occl"] = load_occl(m)
            D[m] = d
    tex = {p: e4.properties(p)["opt"]["texture"] for p in pairs}
    c = (np.arange(N) + 0.5) * CELL
    cx, cy = np.meshgrid(c, c)
    dist = {p: np.min(np.hypot(cx[..., None] - cps[p][0][:, 0], cy[..., None] - cps[p][0][:, 1]), axis=-1) for p in pairs}
    return pairs, cps, D, tex, dist


def load_occl(m):
    """遮挡图，16×16：每块的仿射变化减去该对噪声底、截到 ≥ 0，遮后失败记 50 px（同 E4）。"""
    out = {}
    for f in sorted((RAW / m).glob("occl*.npz")):
        z = np.load(f)
        for i, p in enumerate(z["pairs"].tolist()):
            if i >= int(z["done"]) or np.isnan(z["A0"][i]).any():
                continue
            d = z["delta"][i].astype(np.float64)
            d[np.isnan(d)] = e4.FAIL_DELTA
            m16 = np.clip(d - float(np.nanmean(z["noise"][i])), 0, None)
            if m16.sum() > 0:
                out[p] = m16 / m16.sum()
    return out


def grid(D, m, src, p, n):
    """一张图（展平，和为 1）；内点用全部点。没有时返回 None。"""
    d = D[m]
    if src == "inliers":
        pts = d["inl"][p]
        return hist(pts, n) if len(pts) >= 3 else None
    if src == "certainty":
        return e4.norm(e4.to_grid(d["cert"][p], n)).ravel()
    x = d["occl"].get(p)
    if x is None:
        return None
    return x.ravel() if n == G else e4.norm(np.kron(x, np.ones((n // G, n // G)))).ravel()


# ======================================================================== 三个量

def metrics(m32, tex, dist):
    m = m32.reshape(N, N)
    r, conc = e4.region(m)
    return conc, math.log(tex[r].mean() / tex[~r].mean()), float((m * dist).sum() / dist.mean())


def all_metrics(pairs, D, tex, dist):
    """M[(m, src, kind)][p] = (集中程度, 纹理偏好, 距离比)；kind 为 "all"（全部点 / 原图）或 "matched"（内点点数对齐）。"""
    M = {}
    for net, models in NETS.items():
        for src in SOURCES[net]:
            for m in models:
                M[(m, src, "all")] = {}
                for p in pairs:
                    g = grid(D, m, src, p, N)
                    if g is not None:
                        M[(m, src, "all")][p] = metrics(g, tex[p], dist[p])
            if src != "inliers":
                continue
            for m in models:
                M[(m, src, "matched")] = {}
            for p in pairs:
                k = min(len(D[m]["inl"][p]) for m in models)
                if k < 3:
                    continue
                for m in models:
                    pts = D[m]["inl"][p]
                    v = [metrics(hist(pts[RNG.choice(len(pts), k, replace=False)], N), tex[p], dist[p])
                         for _ in range(DRAWS if len(pts) > k else 1)]
                    M[(m, src, "matched")][p] = tuple(np.mean(v, 0))
    return M


# ======================================================================== 训练前后的相似度

def similarity(pairs, D):
    S = {}
    for net, (zs, *trained) in NETS.items():
        for src in SOURCES[net]:
            for t in trained:
                if src == "inliers":
                    ps = [p for p in pairs if min(len(D[m]["inl"][p]) for m in (zs, t)) >= MIN_SPLIT]
                    same, ceil, diff = np.zeros(len(ps)), np.zeros(len(ps)), []
                    for _ in range(SPLITS):
                        A, B, T = [], [], []
                        for p in ps:
                            z, q = D[zs]["inl"][p], D[t]["inl"][p]
                            h = min(len(z), len(q)) // 2
                            i = RNG.permutation(len(z))
                            A.append(hist(z[i[:h]], G))
                            B.append(hist(z[i[h:2 * h]], G))
                            T.append(hist(q[RNG.choice(len(q), h, replace=False)], G))
                        same += [pc(a, b) for a, b in zip(A, T)]
                        ceil += [pc(a, b) for a, b in zip(A, B)]
                        diff += [pc(A[i], T[j]) for i in range(len(ps)) for j in others(i, len(ps))]
                    S[(t, src)] = {"n_pairs": len(ps), "same": e4.summ(same / SPLITS), "ceiling": e4.summ(ceil / SPLITS),
                                   "different": e4.summ(diff)}
                else:
                    ps = [p for p in pairs if grid(D, zs, src, p, G) is not None and grid(D, t, src, p, G) is not None]
                    Z = [grid(D, zs, src, p, G) for p in ps]
                    T = [grid(D, t, src, p, G) for p in ps]
                    same = [pc(a, b) for a, b in zip(Z, T)]
                    diff = [pc(Z[i], T[j]) for i in range(len(ps)) for j in others(i, len(ps))]
                    S[(t, src)] = {"n_pairs": len(ps), "same": e4.summ(same), "ceiling": None, "different": e4.summ(diff)}
    return S


# ======================================================================== 两种训练是否改动同一些格

def residuals(pairs, D):
    """R[(m, src)][p] = SPLITS 份残差（内点）或一份（certainty、遮挡）；Rel[(m, "inliers")][p] = 残差的可靠程度。"""
    R, Rel = {}, {}
    for net, (zs, un, lab) in NETS.items():
        for src in SOURCES[net]:
            for t in (un, lab):
                R[(t, src)] = {}
            if src != "inliers":
                for p in pairs:
                    Z = grid(D, zs, src, p, G)
                    for t in (un, lab):
                        T = grid(D, t, src, p, G)
                        if Z is not None and T is not None:
                            R[(t, src)][p] = [ols_res(T, Z)]
                continue
            for t in (un, lab):
                Rel[(t, src)] = {}
            for p in pairs:
                if min(len(D[m]["inl"][p]) for m in (zs, un, lab)) < MIN_SPLIT:
                    continue
                z = D[zs]["inl"][p]
                full = {t: hist(D[t]["inl"][p], G) for t in (un, lab)}
                rs, rel = {un: [], lab: []}, {un: [], lab: []}
                for _ in range(SPLITS):
                    i = RNG.permutation(len(z))
                    h = len(z) // 2
                    A, B = hist(z[i[:h]], G), hist(z[i[h:2 * h]], G)
                    rs[un].append(iv_res(full[un], A, B))       # 无标注训练对 A 回归，标注过拟合对 B 回归
                    rs[lab].append(iv_res(full[lab], B, A))
                    for t in (un, lab):
                        q = D[t]["inl"][p]
                        j = RNG.permutation(len(q))
                        k = len(q) // 2
                        r = pc(iv_res(hist(q[j[:k]], G), A, B), iv_res(hist(q[j[k:2 * k]], G), B, A))
                        rel[t].append(2 * r / (1 + r))           # Spearman–Brown：两半 → 全量
                for t in (un, lab):
                    R[(t, src)][p] = rs[t]
                    Rel[(t, src)][p] = float(np.mean(rel[t]))
    return R, Rel


def res_corr(R, a, b):
    """同一对上两份残差的相关（多份时逐份配对后平均）与不同对之间的相关。"""
    ps = sorted(set(R[a]) & set(R[b]))
    same = np.array([np.mean([pc(x, y) for x, y in zip(R[a][p], R[b][p])]) for p in ps])
    diff = np.array([pc(R[a][ps[i]][0], R[b][ps[j]][0]) for i in range(len(ps)) for j in others(i, len(ps))])
    return {"n_pairs": len(ps), "same": e4.summ(same), "different": e4.summ(diff),
            "share_same_above_diff_median": float(np.mean(same > np.median(diff)))}


def shared_change(R, Rel):
    out = {}
    for net, (zs, un, lab) in NETS.items():
        for src in SOURCES[net]:
            out[f"{net}/{src}/无标注~标注"] = res_corr(R, (un, src), (lab, src)) | {"models": [un, lab]}
    for a, b in (("loftr_q4", "roma_m4"), ("loftr_e1", "roma_e1"), ("loftr_q4", "roma_e1"), ("loftr_e1", "roma_m4")):
        out[f"跨网络/inliers/{a}~{b}"] = res_corr(R, (a, "inliers"), (b, "inliers")) | {"models": [a, b]}
    rel = {m: e4.summ(list(v.values())) for (m, _), v in Rel.items()}
    for k, v in out.items():
        a, b = v["models"]
        v["ceiling"] = math.sqrt(rel[a]["median"] * rel[b]["median"]) if "/inliers/" in k else None
    return out, rel


# ======================================================================== 与误差的关系

def read_points(m):
    by = {}
    with open(E3 / "extra" / "data" / f"points_{m}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["pair"], []).append([float(r[c]) if r[c] not in ("", "nan") else np.nan
                                                 for c in ("ox", "oy", "sx", "sy", "ax", "ay", "lx", "ly", "ln", "dx", "dy", "dc")])
    return {p: np.array(v) for p, v in by.items()}


def gaps(D):
    """E5 的 291 对公共点（筛选同 E3）上，模型仿射（E4 那次推理）与局部对应拟合的仿射的 x 向均方误差之差，px²。"""
    keys = [m for ms in NETS.values() for m in ms]
    pts = {m: read_points(m) for m in keys}
    common = sorted(set.intersection(*(set(v) for v in pts.values())))
    fam = lambda m: m.split("_")[0]

    def local_ok(v, m):
        e = v[:, LOCAL[fam(m)]] - v[:, 2:4]
        ok = np.isfinite(e).all(1)
        ok[ok] = np.linalg.norm(e[ok], axis=1) < SANE
        return ok

    out = {m: {} for m in keys}
    for p in common:
        keep = np.all([local_ok(pts[m][p], m) for m in keys], 0)
        if keep.sum() < 4:
            continue
        if any(not np.isfinite(pts[m][p][:, 4:6]).all() or
               np.abs((pts[m][p][keep, 4:6] - pts[m][p][keep, 2:4]).mean(0)).max() >= SANE for m in keys):
            continue
        for m in keys:
            v = pts[m][p][keep]
            o, s, l = v[:, 0:2], v[:, 2:4], v[:, LOCAL[fam(m)]]
            f_ = apply(np.linalg.lstsq(np.c_[o, np.ones(len(o))], l, rcond=None)[0].T, o)
            A = D[m]["A"][p]
            out[m][p] = np.nan if A is None else float(np.mean((apply(A, o)[:, 0] - s[:, 0]) ** 2) -
                                                       np.mean((f_[:, 0] - s[:, 0]) ** 2))
    return out


def relation(pairs, cps, D, M, gap):
    err = {m: {p: e4.pair_error(D[m]["A"][p], *cps[p]) for p in pairs} for m in D}
    lg = lambda e: math.log(e) if np.isfinite(e) and e > 0 else np.nan
    out = {}
    for net, (zs, *trained) in NETS.items():
        for src in SOURCES[net]:
            kind = "matched" if src == "inliers" else "all"
            for t in trained:
                a, b = M[(zs, src, kind)], M[(t, src, kind)]
                ps = [p for p in pairs if p in a and p in b]
                de = np.array([lg(err[t][p]) - lg(err[zs][p]) for p in ps])
                dg = np.array([gap[t].get(p, np.nan) - gap[zs].get(p, np.nan) for p in ps])
                out[f"{t}/{src}"] = {q: {"vs_log_error_change": e4.spearman([b[p][i] - a[p][i] for p in ps], de),
                                         "vs_gap_change": e4.spearman([b[p][i] - a[p][i] for p in ps], dg)}
                                     for i, q in enumerate(QS)}
    gsum = {}
    for m, v in gap.items():
        x = np.array(list(v.values()), float)
        gsum[m] = {"n": int(np.isfinite(x).sum()), "mean_gap_px2": float(np.nanmean(x))}
    return out, gsum


# ======================================================================== 图

def fig_similarity(S):
    rows = [(t, s) for net, (zs, *tr) in NETS.items() for s in SOURCES[net] for t in tr]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(rows))
    seen = set()
    for i, r in enumerate(rows):
        for off, k, col, lab in ((-0.27, "same", "#3182bd", "同一对：zero-shot 与训练后"),
                                 (0.0, "ceiling", "#9ecae1", "上限：zero-shot 内点的两半之间"),
                                 (0.27, "different", "#bdbdbd", "对照：zero-shot 与另一对的训练后")):
            v = S[r][k]
            if v is None:
                continue
            ax.bar(x[i] + off, v["median"], 0.25, color=col, label=None if k in seen else lab)
            seen.add(k)
            ax.plot([x[i] + off] * 2, [v["q25"], v["q75"]], c="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{NAME[t]}\n{SRC_ZH[s]}" for t, s in rows], fontsize=8)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_ylabel("两张图的相关（16×16 格）")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "regions_similarity.png", dpi=120)
    plt.close(fig)


def fig_shared(C):
    rows = [("loftr/inliers/无标注~标注", "LoFTR 两种训练\n内点"), ("loftr/occlusion/无标注~标注", "LoFTR 两种训练\n遮挡"),
            ("roma/certainty/无标注~标注", "RoMa 两种训练\ncertainty"), ("roma/inliers/无标注~标注", "RoMa 两种训练\n内点"),
            ("跨网络/inliers/loftr_q4~roma_m4", "LoFTR 与 RoMa\n无标注训练，内点"),
            ("跨网络/inliers/loftr_e1~roma_e1", "LoFTR 与 RoMa\n标注过拟合，内点")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(rows))
    for i, (k, _) in enumerate(rows):
        v = C[k]
        for off, kk, col, lab in ((-0.2, "same", "#3182bd", "同一对"), (0.2, "different", "#bdbdbd", "不同对")):
            ax.bar(x[i] + off, v[kk]["median"], 0.38, color=col, label=lab if i == 0 else None)
            ax.plot([x[i] + off] * 2, [v[kk]["q25"], v[kk]["q75"]], c="k", lw=1)
        if v.get("ceiling"):
            ax.plot([x[i] - 0.4, x[i] + 0.4], [v["ceiling"]] * 2, c="#e6550d", lw=1.5, ls="--",
                    label="上限（两份残差可靠程度乘积的平方根）" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rows], fontsize=8)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_ylabel("训练带来的变化（残差）之间的相关")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "regions_shared_change.png", dpi=120)
    plt.close(fig)


# ======================================================================== 主流程

def clean(o):
    if isinstance(o, dict):
        return {("/".join(k) if isinstance(k, tuple) else str(k)): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    pairs, cps, D, tex, dist = load()
    print("loaded", flush=True)
    M = all_metrics(pairs, D, tex, dist)
    print("metrics", flush=True)
    res = {"n_pairs": len(pairs), "inlier_count": {m: e4.summ([len(v) for v in D[m]["inl"].values()]) for m in D},
           "n_occlusion_pairs": {m: len(D[m]["occl"]) for m in NETS["loftr"]}, "change": {}}
    for net, (zs, *trained) in NETS.items():
        for src in SOURCES[net]:
            for kind in (("matched", "all") if src == "inliers" else ("all",)):
                for t in trained:
                    a, b = M[(zs, src, kind)], M[(t, src, kind)]
                    ps = [p for p in pairs if p in a and p in b]
                    res["change"][f"{t}/{src}/{kind}"] = {
                        q: paired([a[p][i] for p in ps], [b[p][i] for p in ps]) for i, q in enumerate(QS)}
    S = similarity(pairs, D)
    res["similarity"] = S
    print("similarity", flush=True)
    R, Rel = residuals(pairs, D)
    C, rel = shared_change(R, Rel)
    res["shared_change"], res["residual_reliability"] = C, rel
    print("shared change", flush=True)
    gap = gaps(D)
    res["relation"], res["gap"] = relation(pairs, cps, D, M, gap)
    OUT.mkdir(exist_ok=True)
    (OUT / "regions.json").write_text(json.dumps(clean(res), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    fig_similarity(S)
    fig_shared(C)
    print("done", flush=True)


if __name__ == "__main__":
    main()
