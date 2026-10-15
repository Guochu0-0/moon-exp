"""局部对应拟合仿射的优势是不是样本内效果：留一个坑与逐对可视化（#139）。只用 CPU，读 E6 的全图对应。

    E6_RAW=/remote-home/xufang/YGC/moon-exp/runs/E6/raw /opt/envs/loftr/bin/python runs/E8/code/loo.py [--limit N]

E6 的全图对应在 gpfs 主 checkout 的 runs/E6/raw/（不进 git），由 E6_RAW 指过去。坐标为评测口径（原 512 网格、角点原点）。

一、逐对可视化（写 preds/<模型>/val.jsonl）：
- 方法 = E3 样本内的「局部对应拟合的仿射」，口径同 E3（E6 rings.local_in_sample）：标注点处模型自己的局部对应
  （LoFTR 为 12 px 内匹配位移的中位数、至少 3 个；RoMa 为稠密对应插值），离标注 ≥ 20 px 的不用，至少 4 点，普通最小二乘。
  每个模型用自己能取到局部对应的全部点。不够 4 点的对写失败 few_local。
- 点对写这些局部对应（光学标注点 → 局部对应），供工作台上看。

二、留一个坑（写 extra/）：
- 对的筛选同 E6：模型仿射失败、或平均偏移某一轴超过 20 px 的对整对不计。
- 留下坑 k：候选 = 离其他任一坑 < R_LOO px、且离坑 k ≥ R_LOO px 的全部候选对应（E6 的全图对应：LoFTR 为全部匹配，
  RoMa 为每个像素中心）；不额外排除邻近的坑。至少 MIN_CRATERS 个其他坑各贡献至少 MIN_PER 个候选，否则这个坑不计。
  全部候选直接最小二乘（不做 RANSAC，同 E6 --fit lsq），只在坑 k 上量误差 e = A·o_k − s_k。
  |e| 某一轴超过 20 px 记为「拟合粗错」，单独报告比例，不进误差。
- 同一批点上的对照：样本内（同 E3，在这一对全部标注点上拟合）、模型仿射、全部坑周围拟合（离任一坑 < R_LOO px 的
  全部候选，含被评测的坑，一对拟合一次；与留一同样的候选来源和外点，只差有没有用到被评测的坑）。
- 主指标：x 向均方误差（各点 dx² 的平均）；另报 x 向中位数（各点 |dx| 的中位数）。区间：按对自助抽样 1000 次的 95% 区间，
  差值也按同一批对自助抽样。主口径为各模型自己的点；副口径为 6 个模型都有的点。
产物：preds/<模型>/val.jsonl；extra/loo.json、loo_x.png、data/loo_<模型>.csv（逐点）。
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
RUN = HERE.parent                                   # runs/E8
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "runs" / "E6" / "code"))

import rings  # noqa: E402  （顺带把 runs/E4/code 放进 sys.path）
from common import MODELS, apply, checkpoints, key  # noqa: E402
from rings import ORDER, RAW, SANE, Pair, local_in_sample, score  # noqa: E402

from workbench.launch import begin, end  # noqa: E402
from workbench.records import PredWriter  # noqa: E402

OUT = RUN / "extra"
R_LOO, MIN_CRATERS, MIN_PER = 16.0, 3, 3
LINES = ["model", "in_sample", "all_craters", "loo"]
LINE_NAMES = {"model": "模型仿射", "in_sample": "标注点处局部对应拟合（样本内，同 E3）",
              "all_craters": "全部坑周围拟合（含被评测的坑）", "loo": "留一个坑：其他坑周围拟合"}


def lsq(src, dst):
    return np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0].T


def local_points(P, o, s):
    """E3 口径的局部对应（与 local_in_sample 相同的取法），返回 (光学点, 局部对应) 两个 M×2。"""
    if P.family == "loftr":
        loc = np.full_like(o, np.nan)
        for i, p in enumerate(o):
            nb = np.linalg.norm(P.src - p, axis=1) <= rings.R_NB
            if nb.sum() >= rings.MIN_NB:
                loc[i] = p + np.median(P._dst[nb] - P.src[nb], axis=0)
    else:
        loc = o + rings.bilinear(P.D, o)
    ok = np.isfinite(loc).all(1)
    ok[ok] = np.linalg.norm(loc[ok] - s[ok], axis=1) < SANE
    return o[ok], loc[ok]


def load_pair(model, i, pair):
    if MODELS[model]["family"] == "loftr":
        with np.load(RAW / model / "matches.npz") as z:
            return Pair("loftr", M=z[f"M__{key(pair)}"])
    D = np.load(RAW / model / "D.npy", mmap_mode="r")[i]
    C = np.load(RAW / model / "C.npy", mmap_mode="r")[i]
    return Pair("roma", D=np.asarray(D), C=np.asarray(C))


def one_pair(args):
    model, i, pair, A0 = args
    o, s = checkpoints(pair)
    P = load_pair(model, i, pair)
    A_in, n_local = local_in_sample(P, o, s)
    po, pl = local_points(P, o, s)
    rec = {"pair": pair, "A_in": A_in, "n_local": n_local, "local": (po, pl), "points": []}
    e0 = score(None if np.isnan(A0).any() else A0, o, s)
    rec["excluded"] = e0 is None
    if e0 is None:
        return rec
    e_in = score(A_in, o, s)
    d = np.linalg.norm(P.src[:, None, :] - o[None], axis=2)              # N×K
    near = d < R_LOO
    m_all = near.any(1)
    e_all = score(lsq(P.src[m_all], P.dst(np.flatnonzero(m_all))), o, s) if m_all.sum() >= 3 else None
    for k in range(len(o)):
        far_k = ~near[:, k]
        others = [j for j in range(len(o)) if j != k and (near[:, j] & far_k).sum() >= MIN_PER]
        row = {"k": k, "n_contrib": len(others), "n_cand": 0, "gross": False,
               "model": e0[k], "in_sample": None if e_in is None else e_in[k],
               "all_craters": None if e_all is None else e_all[k], "loo": None}
        if len(others) >= MIN_CRATERS:
            m = near[:, others].any(1) & far_k
            idx = np.flatnonzero(m)
            row["n_cand"] = int(len(idx))
            e = apply(lsq(P.src[idx], P.dst(idx)), o[k:k + 1])[0] - s[k]
            if np.abs(e).max() > SANE:
                row["gross"] = True
            else:
                row["loo"] = e
        rec["points"].append(row)
    return rec


# ---------------------------------------------------------------- 汇总

def usable(r):
    """进主比较的点：留一可用且非粗错、样本内与全部坑拟合都有。"""
    return all(r[k] is not None for k in LINES)


def per_pair(recs, keep):
    """每对：每条线的 (dx² 和, |dx| 列表, 点数)。keep(rec, row) 决定点是否进入。"""
    out = []
    for rec in recs:
        rows = [x for x in rec["points"] if keep(rec, x)]
        if rows:
            out.append({k: np.array([x[k][0] for x in rows]) for k in LINES})
    return out


def stats(pp, n_boot=1000, seed=0):
    res = {"n_pairs": len(pp), "n_points": int(sum(len(p["loo"]) for p in pp))}
    if not pp:
        return res
    S = np.array([[(p[k] ** 2).sum() for k in LINES] for p in pp])     # 对×线
    n = np.array([len(p["loo"]) for p in pp], float)
    rng = np.random.default_rng(seed)
    B = np.zeros((n_boot, len(LINES)))
    for b in range(n_boot):
        ix = rng.integers(0, len(pp), len(pp))
        B[b] = S[ix].sum(0) / n[ix].sum()
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    for j, k in enumerate(LINES):
        allx = np.concatenate([p[k] for p in pp])
        res[k] = {"ms_x": float(S[:, j].sum() / n.sum()), "ms_x_ci": ci(B[:, j]),
                  "med_x": float(np.median(np.abs(allx)))}
    for a, b in (("loo", "model"), ("loo", "in_sample"), ("loo", "all_craters"), ("in_sample", "model")):
        ja, jb = LINES.index(a), LINES.index(b)
        res[f"diff_{a}_minus_{b}"] = {"ms_x": res[a]["ms_x"] - res[b]["ms_x"], "ci": ci(B[:, ja] - B[:, jb])}
    res["loo_drop_vs_model"] = 1 - res["loo"]["ms_x"] / res["model"]["ms_x"]
    res["in_sample_drop_vs_model"] = 1 - res["in_sample"]["ms_x"] / res["model"]["ms_x"]
    return res


def coverage(recs):
    kept = [r for r in recs if not r["excluded"]]
    pts = [x for r in kept for x in r["points"]]
    tried = [x for x in pts if x["n_contrib"] >= MIN_CRATERS]
    return {"n_pairs": len(recs), "n_pairs_excluded": len(recs) - len(kept), "n_points": len(pts),
            "n_points_loo_tried": len(tried), "loo_gross_rate": float(np.mean([x["gross"] for x in tried])) if tried else None,
            "n_points_in_sample": sum(x["in_sample"] is not None for x in pts),
            "median_contrib_craters": float(np.median([x["n_contrib"] for x in tried])) if tried else None,
            "median_cand": float(np.median([x["n_cand"] for x in tried])) if tried else None}


def write_csv(model, recs):
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    with open(OUT / "data" / f"loo_{model}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "k", "n_contrib", "n_cand", "gross"] + [f"{k}_{a}" for k in LINES for a in ("dx", "dy")])
        for r in recs:
            for x in r["points"]:
                vals = []
                for k in LINES:
                    vals += ["", ""] if x[k] is None else [f"{x[k][0]:.4f}", f"{x[k][1]:.4f}"]
                w.writerow([r["pair"], x["k"], x["n_contrib"], x["n_cand"], int(x["gross"])] + vals)


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"model": "k", "in_sample": "tab:green", "all_craters": "tab:gray", "loo": "tab:blue"}
    models = [m for m in ORDER if m in summary["models"]]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.2
    for j, k in enumerate(LINES):
        y = [summary["models"][m]["own"][k]["ms_x"] for m in models]
        lo = [y_ - summary["models"][m]["own"][k]["ms_x_ci"][0] for y_, m in zip(y, models)]
        hi = [summary["models"][m]["own"][k]["ms_x_ci"][1] - y_ for y_, m in zip(y, models)]
        ax.bar(np.arange(len(models)) + (j - 1.5) * w, y, w, yerr=[lo, hi], color=colors[k], alpha=0.8,
               label=LINE_NAMES[k], capsize=2)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([f"{MODELS[m]['name']}\n（{summary['models'][m]['own']['n_points']} 点）" for m in models], fontsize=9)
    ax.set_ylabel("标注点 x 向均方误差（px²）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "loo_x.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cmd = ["runs/E8/code/loo.py"] + sys.argv[1:]
    for m in a.models:
        begin(RUN, m, cmd, entry="runs/E8/code/loo.py", e6_raw=str(RAW), model=MODELS[m])
    status = "fail"
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        summary = {"settings": {"r_loo_px": R_LOO, "min_craters": MIN_CRATERS, "min_per_crater": MIN_PER,
                                "fit": "lsq", "sane_px": SANE, "lines": LINE_NAMES}, "models": {}}
        allrecs = {}
        for m in a.models:
            pairs = json.loads((RAW / m / "pairs.json").read_text(encoding="utf-8"))[: a.limit or None]
            A = np.load(RAW / m / "A.npy")
            with Pool(a.workers) as pool:
                recs = pool.map(one_pair, [(m, i, p, A[i]) for i, p in enumerate(pairs)], chunksize=4)
            allrecs[m] = recs
            with PredWriter(RUN, m, "val", repo=REPO) as w:
                for r in recs:
                    po, pl = r["local"]
                    M = np.c_[po, pl] - 0.5                    # matcher 约定（整数 = 像素中心），写入端 +0.5
                    if r["A_in"] is None:
                        w.write(r["pair"], None, fail="few_local", matches=M, n_local=r["n_local"])
                    else:
                        w.write(r["pair"], r["A_in"], matches=M, n_local=r["n_local"])
            write_csv(m, recs)
            summary["models"][m] = {"coverage": coverage(recs),
                                    "own": stats(per_pair([r for r in recs if not r["excluded"]], lambda rec, x: usable(x)))}
            s_ = summary["models"][m]["own"]
            print(m, s_["n_points"], {k: round(s_[k]["ms_x"], 2) for k in LINES}, flush=True)
        # 副口径：6 个模型都有的点
        common = None
        for m, recs in allrecs.items():
            ks = {(r["pair"], x["k"]) for r in recs if not r["excluded"] for x in r["points"] if usable(x)}
            common = ks if common is None else common & ks
        for m, recs in allrecs.items():
            summary["models"][m]["common"] = stats(per_pair([r for r in recs if not r["excluded"]],
                                                            lambda rec, x: (rec["pair"], x["k"]) in common))
        (OUT / "loo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        plot(summary)
        status = "ok"
    finally:
        for m in a.models:
            end(RUN, m, status)


if __name__ == "__main__":
    main()
