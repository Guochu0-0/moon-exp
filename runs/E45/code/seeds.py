"""E45 / E41 门槛 0.8 的三个种子（#264）：表与图。读 runs/E45、runs/E41 的 sweep/<方法>/S/metrics.json（逐存档 Val，含逐对误差）。

    python runs/E45/code/seeds.py

种子 0 是 E45 main 与 E41 main__screen0.8，种子 1、2 是 main__seed<s> 与 main__screen0.8__seed<s>。写 extra/data/seeds_*.csv、
extra/seeds.png 并打印：
1. seeds_diff.csv：每个种子 E45 减同种子 E41 的 AUC@5，按训练集遍数（每 1977 次更新）分四段的平均差、全部存档的平均差、
   最后存档（第 7907 次更新）的差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 summary.py）；
   pooled 行为三个种子合起来：各段为三个种子平均差的平均，最后存档为三个种子差的平均，区间在每次重抽中对三个种子用同一批图对。
2. seeds_spread.csv：同一配置不同种子之间：各段平均与最后存档 AUC@5 的逐种子值、极差、标准差（ddof=1）；
   最后存档种子 s 减种子 0 的差与配对自助区间。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
from workbench.protocol import auc  # noqa: E402

E41 = REPO / "runs" / "E41"
SEEDS = (0, 1, 2)
METH = {"E45": (R, {0: "main", 1: "main__seed1", 2: "main__seed2"}),
        "E41": (E41, {0: "main__screen0.8", 1: "main__screen0.8__seed1", 2: "main__screen0.8__seed2"})}
NAME = {"E45": "加等变约束（E45）", "E41": "不加（E41 门槛 0.8）"}
PASS, LAST, NB = 1977, 7907, 2000
OUT = R / "extra" / "data"


def seg_of(s):
    return min(4, (s - 1) // PASS + 1)


def load(d, m):
    met = d / "sweep" / m / "S" / "metrics.json"
    if not met.exists():
        return {}
    out = {}
    for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items():
        if k.startswith("step") and "val" in v and int(k[4:]) > 0:
            v = v["val"]
            out[int(k[4:])] = {"auc5": v["summary"]["auc@5"], "pairs": v.get("pairs"),
                               "err": np.array([np.inf if x is None else x for x in v["errors"]], float)}
    return out


def boot_auc(e, idx):
    return np.array([auc(e[i], 5.0) for i in idx])


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"== {name}")
    for r in rows:
        print(" | ".join(f"{k}={r[k]}" for k in r))


def segmeans(v):
    return [float(np.mean([v[s]["auc5"] for s in v if seg_of(s) == g])) if any(seg_of(s) == g for s in v) else np.nan
            for g in range(1, 5)]


def main():
    val = {k: {sd: load(d, ms[sd]) for sd in SEEDS} for k, (d, ms) in METH.items()}
    have = [sd for sd in SEEDS if LAST in val["E45"][sd] and LAST in val["E41"][sd]]
    n = len(val["E45"][0][LAST]["err"])
    for k in METH:                          # 逐对误差的顺序须一致，配对自助才成立
        for sd in have:
            assert val[k][sd][LAST]["pairs"] == val["E45"][0][LAST]["pairs"], (k, sd)
    idx = np.random.default_rng(0).integers(0, n, (NB, n))
    bt = {(k, sd): boot_auc(val[k][sd][LAST]["err"], idx) for k in METH for sd in have}

    rows, segd, lastd = [], [], []
    for sd in have:
        a, b = val["E45"][sd], val["E41"][sd]
        common = sorted(set(a) & set(b))
        dif = {s: a[s]["auc5"] - b[s]["auc5"] for s in common}
        sm = [float(np.mean([dif[s] for s in common if seg_of(s) == g])) for g in range(1, 5)]
        d_last = a[LAST]["auc5"] - b[LAST]["auc5"]
        ci = np.percentile(bt["E45", sd] - bt["E41", sd], [2.5, 97.5])
        segd.append(sm)
        lastd.append(d_last)
        rows.append({"seed": sd, "n_ckpt": len(common), **{f"seg{g + 1}": round(sm[g], 4) for g in range(4)},
                     "all": round(float(np.mean(list(dif.values()))), 4), "n_lower": sum(x < 0 for x in dif.values()),
                     "last_e45": round(a[LAST]["auc5"], 4), "last_e41": round(b[LAST]["auc5"], 4),
                     "last_diff": round(d_last, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4)})
    if len(have) > 1:
        pm = np.mean(segd, 0)
        pb = np.mean([bt["E45", sd] - bt["E41", sd] for sd in have], 0)
        ci = np.percentile(pb, [2.5, 97.5])
        rows.append({"seed": "pooled", "n_ckpt": "", **{f"seg{g + 1}": round(pm[g], 4) for g in range(4)},
                     "all": round(float(np.mean([r["all"] for r in rows])), 4), "n_lower": "",
                     "last_e45": round(float(np.mean([val["E45"][sd][LAST]["auc5"] for sd in have])), 4),
                     "last_e41": round(float(np.mean([val["E41"][sd][LAST]["auc5"] for sd in have])), 4),
                     "last_diff": round(float(np.mean(lastd)), 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4)})
    write("seeds_diff.csv", rows)

    sp = []
    for k in METH:
        segs = {sd: segmeans(val[k][sd]) for sd in have}
        for g in range(5):
            xs = [segs[sd][g] if g < 4 else val[k][sd][LAST]["auc5"] for sd in have]
            r = {"run": k, "what": f"seg{g + 1}" if g < 4 else "last", **{f"seed{sd}": round(x, 4) for sd, x in zip(have, xs)},
                 "range": round(max(xs) - min(xs), 4), "sd": round(float(np.std(xs, ddof=1)), 4) if len(xs) > 1 else ""}
            if g == 4:
                for sd in have[1:]:
                    c = np.percentile(bt[k, sd] - bt[k, 0], [2.5, 97.5])
                    r[f"s{sd}_minus_s0"] = f"{xs[have.index(sd)] - xs[0]:+.4f} ({c[0]:+.4f}, {c[1]:+.4f})"
            sp.append(r)
    write("seeds_spread.csv", sp)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font = Path("/remote-home/xufang/YGC/fonts/simhei.ttf")
    if font.exists():
        matplotlib.font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ls = {0: "-", 1: "--", 2: ":"}
    for k, c in (("E45", "C3"), ("E41", "0.35")):
        for sd in have:
            v = val[k][sd]
            ax[0].plot(sorted(v), [v[s]["auc5"] for s in sorted(v)], ls[sd], color=c, label=f"{NAME[k]} 种子 {sd}")
    for sd in have:
        a, b = val["E45"][sd], val["E41"][sd]
        cm = sorted(set(a) & set(b))
        ax[1].plot(cm, [a[s]["auc5"] - b[s]["auc5"] for s in cm], ls[sd], color=f"C{sd}", label=f"种子 {sd}")
    ax[1].axhline(0, color="k", lw=0.6)
    for a_ in ax:
        for x in (PASS, 2 * PASS, 3 * PASS):
            a_.axvline(x, color="0.8", lw=0.6)
        a_.set_xlabel("更新次数")
        a_.legend(fontsize=8)
    ax[0].set_ylabel("Val AUC@5")
    ax[1].set_ylabel("AUC@5 之差（E45 减同种子 E41）")
    fig.tight_layout()
    fig.savefig(R / "extra" / "seeds.png", dpi=130)


if __name__ == "__main__":
    main()
