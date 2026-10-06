"""训练改变了什么（「【误差分析】训练改变了什么」#102）：zero-shot 与训练后模型在同一批 Val 对上的逐对比较。只读已有结果，只用 CPU。

    /opt/envs/loftr/bin/python runs/E5/code/train_change.py      （服务器上跑：E4 的原始重要性图只在 gpfs 主 checkout）

四组比较：LoFTR zero-shot → 无标注训练 / 标注过拟合；RoMa zero-shot → 无标注训练 / 标注过拟合。
- 实验一（误差）读 E3 的逐点数据 runs/E3/extra/data/points_<模型>.csv（评测记录的仿射、标注、局部对应）。
  一对的误差 d = A·o − s；平均偏移 b = mean(d)；均方误差 mean|d|² = |b|² + mean|d − b|²（平均偏移 + 平移以外）。
  三段：同 E3，在 6 个模型都有局部对应的公共点集上，比较模型仿射、局部对应拟合的仿射、局部对应本身的 x 向均方误差。
- 实验二（看的地方）读 E4 的原始推理 runs/E4/raw/<模型>/（注意力 / certainty、内点、遮挡），用 E4 的读法与定义
  （32×32 格、和为 1、重要区域、集中程度、纹理对数比、加权距离）。误差按 E4 那次推理的仿射计算。
  实验二已由 regions.py 取代（来源换成 LoFTR 的内点与遮挡、RoMa 的 certainty 与内点，内点点数对齐，加跨网络对照）；
  Notes 不再引用这里的 stats.json["exp2"] 与 map_similarity.png。
产物：runs/E5/extra/stats.json 与图。
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
E4_CODE = REPO / "runs" / "E4" / "code"
MAIN = Path(os.environ.get("MAIN", "/remote-home/xufang/YGC/moon-exp"))
OUT = RUN / "extra"
sys.path.insert(0, str(E4_CODE))

import analyze as e4  # noqa: E402  （E4 的读法、统计与字体设置）
from common import MODELS, apply, checkpoints, val_pairs  # noqa: E402

e4.RUN = MAIN / "runs" / "E4"                         # 原始推理不进 git，只在主 checkout
plt = e4.plt
spearman, wilcoxon, region = e4.spearman, e4.wilcoxon, e4.region

SANE = 20.0                                         # 同 E3：平均偏移某一轴超过它视为粗错
NAME = {"loftr_zs": "LoFTR zero-shot", "loftr_q4": "LoFTR 无标注训练", "loftr_e1": "LoFTR 标注过拟合",
        "roma_zs": "RoMa zero-shot", "roma_m4": "RoMa 无标注训练", "roma_e1": "RoMa 标注过拟合"}
TRANS = [("loftr_zs", "loftr_q4"), ("loftr_zs", "loftr_e1"), ("roma_zs", "roma_m4"), ("roma_zs", "roma_e1")]
LOCAL = {"loftr": slice(6, 8), "roma": slice(9, 11)}   # 同 E3：LoFTR 用邻近匹配，RoMa 用稠密对应
RNG = np.random.default_rng(0)


def fam(m):
    return m.split("_")[0]


def tname(t):
    return f"{NAME[t[1]]}"


def summ(v):
    return e4.summ(v)


def paired(before, after):
    """配对比较：前后中位数、after > before 的比例、符号秩检验 p。"""
    b, a = np.asarray(before, float), np.asarray(after, float)
    ok = np.isfinite(a) & np.isfinite(b)
    b, a = b[ok], a[ok]
    return {"n": int(ok.sum()), "median_before": float(np.median(b)) if len(b) else None,
            "median_after": float(np.median(a)) if len(a) else None,
            "median_change": float(np.median(a - b)) if len(b) else None,
            "share_increase": float(np.mean(a > b)) if len(b) else None, "wilcoxon_p": wilcoxon(a - b)}


# ======================================================================== 实验一：误差

def read_points(m):
    by = {}
    with open(E3 / "extra" / "data" / f"points_{m}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["pair"], []).append([float(r[c]) if r[c] not in ("", "nan") else np.nan
                                                 for c in ("ox", "oy", "sx", "sy", "ax", "ay", "lx", "ly", "ln", "dx", "dy", "dc")])
    return {p: np.array(v) for p, v in by.items()}


def pair_terms(v):
    """一对：平均偏移 b、各轴的平均偏移平方与平移以外的均方，误差（平均距离）；失败或粗错返回 None。"""
    a, s = v[:, 4:6], v[:, 2:4]
    if not np.isfinite(a).all():
        return None
    d = a - s
    b = d.mean(0)
    if np.abs(b).max() >= SANE:
        return None
    r = d - b
    return {"b": b, "off2": b ** 2, "rem2": (r ** 2).mean(0), "err": float(np.linalg.norm(d, axis=1).mean()),
            "rem": float(np.linalg.norm(r, axis=1).mean())}


def resid(y, x):
    """y 对 x 做一元线性回归后的残差。"""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return y - np.polyval(np.polyfit(x, y, 1), x)


def lsq_affine(o, s):
    return np.linalg.lstsq(np.c_[o, np.ones(len(o))], s, rcond=None)[0].T


def local_ok(v, m):
    e = v[:, LOCAL[fam(m)]] - v[:, 2:4]
    ok = np.isfinite(e).all(1)
    ok[ok] = np.linalg.norm(e[ok], axis=1) < SANE
    return ok


def exp1():
    keys = list(NAME)
    pts = {m: read_points(m) for m in keys}
    pairs = sorted(set.intersection(*(set(v) for v in pts.values())))
    T = {m: {p: pair_terms(pts[m][p]) for p in pairs} for m in keys}
    res = {"n_pairs": len(pairs), "bad": {m: sum(T[m][p] is None for p in pairs) for m in keys}, "trans": {}}

    for z, t in TRANS:
        ok = [p for p in pairs if T[z][p] is not None and T[t][p] is not None]
        g = lambda m, k: np.array([T[m][p][k] for p in ok])
        r = {"n_pairs": len(ok),
             "fixed": sum(T[z][p] is None and T[t][p] is not None for p in pairs),
             "broken": sum(T[z][p] is not None and T[t][p] is None for p in pairs)}
        # 均方误差（逐对取均值，等权）及其四个组成的变化
        comp = {}
        for k, lab in (("off2", "offset"), ("rem2", "rest")):
            for ax, c in (("x", 0), ("y", 1)):
                comp[f"{lab}_{ax}"] = {"before": float(g(z, k)[:, c].mean()), "after": float(g(t, k)[:, c].mean())}
        tot_b = sum(v["before"] for v in comp.values())
        tot_a = sum(v["after"] for v in comp.values())
        for v in comp.values():
            v["change"] = v["after"] - v["before"]
            v["share_of_drop"] = v["change"] / (tot_a - tot_b)
        r["ms"] = {"before": tot_b, "after": tot_a, "components": comp}
        r["err"] = paired(g(z, "err"), g(t, "err"))
        r["rem"] = paired(g(z, "rem"), g(t, "rem"))
        # 平均偏移：逐对前后相关、回归斜率、同号率
        for ax, c in (("x", 0), ("y", 1)):
            bz, bt = g(z, "b")[:, c], g(t, "b")[:, c]
            slope = float(np.polyfit(bz, bt, 1)[0])
            bs = []
            for _ in range(1000):
                i = RNG.integers(0, len(bz), len(bz))
                bs.append(np.polyfit(bz[i], bt[i], 1)[0])
            both = (np.abs(bz) > 1) & (np.abs(bt) > 1)
            r[f"offset_{ax}"] = {"pearson": float(np.corrcoef(bz, bt)[0, 1]), "slope": slope,
                                 "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                 "std_before": float(bz.std()), "std_after": float(bt.std()),
                                 "same_sign_both_gt1": float(np.mean(np.sign(bz[both]) == np.sign(bt[both]))),
                                 "n_both_gt1": int(both.sum()),
                                 "abs_paired": paired(np.abs(bz), np.abs(bt))}
        r["_bx"] = (g(z, "b")[:, 0], g(t, "b")[:, 0])
        r["_pairs"] = ok
        res["trans"][f"{z}>{t}"] = r

    # 两个训练后的模型是否朝同一方向变：各自对自己的 zero-shot 回归、取残差（训练带来的、zero-shot 预测不了的部分），
    # 再算两份残差的相关。直接拿「训练后 − zero-shot」相关会因共用同一个 zero-shot 项而虚高。
    # 对照：LoFTR 与 RoMa 两个无标注训练（起点不同），反映与训练方式无关、只因这对图像本身而共有的变化。
    res["same_change"] = {}
    for lab_, (z1, t1), (z2, t2) in (("loftr", ("loftr_zs", "loftr_q4"), ("loftr_zs", "loftr_e1")),
                                     ("roma", ("roma_zs", "roma_m4"), ("roma_zs", "roma_e1")),
                                     ("loftr~roma_unlabelled", ("loftr_zs", "loftr_q4"), ("roma_zs", "roma_m4")),
                                     ("loftr~roma_labelled", ("loftr_zs", "loftr_e1"), ("roma_zs", "roma_e1"))):
        ok = [p for p in pairs if all(T[m][p] is not None for m in (z1, t1, z2, t2))]
        # 误差取对数（偏态）；平均偏移取原值
        v = lambda m, k: np.array([T[m][p]["b"][0] if k == "bx" else math.log(T[m][p][k]) for p in ok])
        r = {"n_pairs": len(ok)}
        for k in ("bx", "err"):
            r1, r2 = resid(v(t1, k), v(z1, k)), resid(v(t2, k), v(z2, k))
            r[f"{k}_resid_pearson"] = float(np.corrcoef(r1, r2)[0, 1])
            r[f"{k}_after_pearson"] = float(np.corrcoef(v(t1, k), v(t2, k))[0, 1])
            r[f"{k}_before_pearson"] = float(np.corrcoef(v(z1, k), v(z2, k))[0, 1])
        res["same_change"][lab_] = r

    # 三段：6 个模型都有局部对应的公共点集，x 向均方误差。筛选同 E3：至少 4 个公共点，
    # 每个模型的仿射有效、且在这些点上的平均偏移两轴都小于 SANE
    rows = {m: [] for m in keys}
    for p in pairs:
        keep = np.all([local_ok(pts[m][p], m) for m in keys], 0)
        if keep.sum() < 4:
            continue
        if any(not np.isfinite(pts[m][p][:, 4:6]).all() or
               np.abs((pts[m][p][keep, 4:6] - pts[m][p][keep, 2:4]).mean(0)).max() >= SANE for m in keys):
            continue
        for m in keys:
            v = pts[m][p][keep]
            o, s, a, l = v[:, 0:2], v[:, 2:4], v[:, 4:6], v[:, LOCAL[fam(m)]]
            f_ = apply(lsq_affine(o, l), o)
            g_ = apply(lsq_affine(o, s), o)
            rows[m].append({"a": a[:, 0] - s[:, 0], "f": f_[:, 0] - s[:, 0], "g": g_[:, 0] - s[:, 0], "l": l[:, 0] - s[:, 0]})
    seg = {"n_pairs": len(rows[keys[0]]), "n_points": int(sum(len(r["a"]) for r in rows[keys[0]])), "models": {}, "trans": {}}
    for m in keys:
        seg["models"][m] = {k: float(np.mean(np.concatenate([r[k] for r in rows[m]]) ** 2)) for k in ("a", "f", "g", "l")}
    for z, t in TRANS:
        tr = {}
        for k in ("a", "f", "l"):
            pz = np.array([np.mean(r[k] ** 2) for r in rows[z]])
            pt = np.array([np.mean(r[k] ** 2) for r in rows[t]])
            tr[k] = {"before": seg["models"][z][k], "after": seg["models"][t][k],
                     "change": seg["models"][t][k] - seg["models"][z][k], "pair_paired": paired(pz, pt)}
        seg["trans"][f"{z}>{t}"] = tr
    res["segments"] = seg
    return res


def fig_offset(res):
    fig, axs = plt.subplots(1, 4, figsize=(16, 5.2), sharex=True, sharey=True)
    for a, (z, t) in zip(axs, TRANS):
        r = res["trans"][f"{z}>{t}"]
        bz, bt = r["_bx"]
        a.scatter(bz, bt, s=5, alpha=0.4)
        lim = 12
        a.plot([-lim, lim], [-lim, lim], c="0.5", lw=0.8, ls="--")
        xs = np.array([-lim, lim])
        a.plot(xs, r["offset_x"]["slope"] * xs + np.polyval(np.polyfit(bz, bt, 1), 0), c="C3", lw=1)
        a.axhline(0, c="k", lw=0.4)
        a.axvline(0, c="k", lw=0.4)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        o = r["offset_x"]
        a.set_title(f"{NAME[t]}（{r['n_pairs']} 对）\n相关 {o['pearson']:.2f}，斜率 {o['slope']:.2f}", fontsize=10)
        a.set_xlabel(f"{NAME[z]} 的 x 向平均偏移（px）", fontsize=9)
    axs[0].set_ylabel("训练后的 x 向平均偏移（px）", fontsize=9)
    fig.suptitle("同一对图像在训练前后的 x 向平均偏移（虚线：不变；红线：最小二乘拟合）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "offset_before_after_x.png", dpi=120)
    plt.close(fig)


# ======================================================================== 实验二：看的地方

def exp2():
    pairs = val_pairs()
    cps = {p: checkpoints(p) for p in pairs}
    names = list(NAME)
    D = {m: e4.load_model(m, pairs) for m in names}
    for m in names:
        D[m]["occlusion"], _ = e4.load_occlusion(m)
    print("loaded", flush=True)
    props = {p: e4.properties(p) for p in pairs}
    print("properties done", flush=True)
    err = {m: {p: e4.pair_error(D[m]["A"][p], *cps[p]) for p in pairs} for m in names}
    N, CELL = e4.N, e4.CELL
    c = (np.arange(N) + 0.5) * CELL
    cx, cy = np.meshgrid(c, c)
    dist = {p: np.min(np.hypot(cx[..., None] - cps[p][0][:, 0], cy[..., None] - cps[p][0][:, 1]), axis=-1) for p in pairs}
    corr = lambda x, y: spearman(x.ravel(), y.ravel())["rho"]

    def tex(m, s, p):
        x = D[m][s]["opt"].get(p)
        if x is None:
            return np.nan
        r, _ = region(x)
        P = props[p]["opt"]["texture"]
        return math.log(P[r].mean() / P[~r].mean())

    res = {"n_pairs": len(pairs), "trans": {}, "agreement": {}, "same_change": {}}
    sim = {}
    for z, t in TRANS:
        for s in ("primary", "inliers", "occlusion"):
            if s == "occlusion" and fam(z) == "roma":
                continue                                    # RoMa 遮挡图被噪声主导（E4），不比
            ps = [p for p in pairs if D[z][s]["opt"].get(p) is not None and D[t][s]["opt"].get(p) is not None]
            same = np.array([corr(D[z][s]["opt"][p], D[t][s]["opt"][p]) for p in ps])
            other = np.array([corr(D[z][s]["opt"][p], D[t][s]["opt"][ps[j]]) for i, p in enumerate(ps)
                              for j in RNG.choice([j for j in range(len(ps)) if j != i], min(5, len(ps) - 1), replace=False)])
            sim[(z, t, s)] = (same, other)
            conc = paired([region(D[z][s]["opt"][p])[1] for p in ps], [region(D[t][s]["opt"][p])[1] for p in ps])
            texture = paired([tex(z, s, p) for p in ps], [tex(t, s, p) for p in ps])
            ratio = lambda m: np.array([(D[m][s]["opt"][p] * dist[p]).sum() / dist[p].mean() for p in ps])
            rz, rt = ratio(z), ratio(t)
            ez, et = np.array([err[z][p] for p in ps]), np.array([err[t][p] for p in ps])
            fin = np.isfinite(ez) & np.isfinite(et)
            res["trans"][f"{z}>{t}/{s}"] = {
                "n_pairs": len(ps), "similarity_same_pair": summ(same), "similarity_different_pair": summ(other),
                "concentration": conc, "texture_logratio": texture, "distance_ratio": paired(rz, rt),
                "rho_ratio_change_vs_err_change": spearman((rt - rz)[fin], (et - ez)[fin]),
                "rho_similarity_vs_err_change": spearman(same[fin], (et - ez)[fin])}
    # LoFTR 与 RoMa 在同一对上的一致程度：训练前后配对
    for s in ("primary", "inliers"):
        base = None
        for a, b in (("loftr_zs", "roma_zs"), ("loftr_q4", "roma_m4"), ("loftr_e1", "roma_e1")):
            ps = [p for p in pairs if D[a][s]["opt"].get(p) is not None and D[b][s]["opt"].get(p) is not None]
            v = {p: corr(D[a][s]["opt"][p], D[b][s]["opt"][p]) for p in ps}
            if base is None:
                base = v
                continue
            common = [p for p in ps if p in base]
            res["agreement"][f"{a}~{b}/{s}"] = paired([base[p] for p in common], [v[p] for p in common])
    # 有标注与无标注训练让重要性图朝同一方向变吗：每张训练后的图（取秩）对同一对的 zero-shot 图（取秩）回归取残差，
    # 即训练带来的、zero-shot 预测不了的部分；同一对上两份残差的相关，对照不同对。
    rk = lambda m: e4.rank(m.ravel())
    for f, un, lab in (("loftr", "loftr_q4", "loftr_e1"), ("roma", "roma_m4", "roma_e1")):
        z = f"{f}_zs"
        for s in ("primary", "inliers"):
            ps = [p for p in pairs if all(D[m][s]["opt"].get(p) is not None for m in (z, un, lab))]
            ch = {(m, p): resid(rk(D[m][s]["opt"][p]), rk(D[z][s]["opt"][p])) for m in (un, lab) for p in ps}
            pc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
            same = np.array([pc(ch[(un, p)], ch[(lab, p)]) for p in ps])
            other = np.array([pc(ch[(un, p)], ch[(lab, ps[j])]) for i, p in enumerate(ps)
                              for j in RNG.choice([j for j in range(len(ps)) if j != i], 5, replace=False)])
            res["same_change"][f"{f}/{s}"] = {"n_pairs": len(ps), "same_pair": summ(same), "different_pair": summ(other),
                                              "share_same_above_diff_median": float(np.mean(same > np.median(other)))}
    return res, sim


def fig_similarity(sim):
    SRC = {"primary": "注意力 / certainty", "inliers": "内点分布", "occlusion": "遮挡敏感性"}
    keys = list(sim)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(keys))
    ax.boxplot([sim[k][0][np.isfinite(sim[k][0])] for k in keys], positions=x - 0.18, widths=0.3, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor="#9ecae1"))
    ax.boxplot([sim[k][1][np.isfinite(sim[k][1])] for k in keys], positions=x + 0.18, widths=0.3, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor="#d9d9d9"))
    src = lambda z, s: ("注意力" if fam(z) == "loftr" else "certainty") if s == "primary" else SRC[s]
    ax.set_xticks(x, [f"{NAME[t]}\n{src(z, s)}" for z, t, s in keys], fontsize=8)
    ax.axhline(0, c="k", lw=0.4)
    ax.set_ylabel("训练前后两张重要性图的秩相关")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc="#9ecae1"), plt.Rectangle((0, 0), 1, 1, fc="#d9d9d9")],
              ["同一对：zero-shot 与训练后", "不同对：zero-shot 的这一对与训练后的另一对"], fontsize=8, loc="lower left")
    ax.set_title("训练后的模型看的地方与 zero-shot 有多像（光学坐标系，32×32 格）", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "map_similarity.png", dpi=120)
    plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r1 = exp1()
    fig_offset(r1)
    print("exp1 done", flush=True)
    r2, sim = exp2()
    fig_similarity(sim)
    stats = {"models": NAME, "exp1": clean(r1), "exp2": clean(r2)}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
