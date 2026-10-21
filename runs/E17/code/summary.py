"""E17 的汇总与图（读 raw/ 下 probe.py 的产物，写 extra/data/ 与 extra/rings.png）。只用 CPU。

    /opt/envs/loftr/bin/python runs/E17/code/summary.py      （工作台环境没有 matplotlib）

- 第 1 项：每个（数据, 模型）第 0…10 圈算对的格子比例 → extra/data/rings.csv、extra/rings.png（月球实线、MegaDepth 虚线）。
- 第 2 项：真值一个候选间距内的概率之和、最近候选位置排名的分位数，以及排第 1、前 5、前 20 的比例 → summary.json。
- 第 3 项：每个模型不替换 / 替换成真值时的 Val 汇总（workbench.protocol.summarize：AUC@5 与自助法 95% CI、误差中位数），
  以及标注点拟合残差 → summary.json；逐对误差复制到 extra/data/gtref_<模型>.csv。
缺哪份 raw 就跳过哪份。
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN.parents[1]))
from workbench import protocol  # noqa: E402

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 服务器上没有中文字体，放了一份在 gpfs（同 E4）
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RAW, OUT = RUN / "raw", RUN / "extra" / "data"
RINGS = range(11)
MOON = ["zs", "loo_t05", "loo_t03", "none_t03", "rns_r2"]
MEGA = ["zs", "roma_outdoor"]
LABEL = {"zs": "未训练的 AnyMatch-RoMa", "loo_t05": "强化学习：留一基线、温度 0.5",
         "loo_t03": "强化学习：留一基线、温度 0.3", "none_t03": "强化学习：直接权重、温度 0.3",
         "rns_r2": "伪标签自训练第 2 轮", "roma_outdoor": "RoMa 原版室外权重"}
COLOR = {"zs": "k", "loo_t05": "tab:blue", "loo_t03": "tab:cyan", "none_t03": "tab:purple", "rns_r2": "tab:red",
         "roma_outdoor": "tab:green"}


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def pct(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, np.float64)
    return {f"p{q}": r4(np.percentile(x, q)) for q in qs} | {"mean": r4(x.mean())}


def coarse(kind, m):
    p = RAW / f"coarse_{kind}_{m}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    ring, pn, rank = z["ring"], z["p_near"], z["rank"].astype(np.int64)
    return {"n_pairs": meta["n_pairs"], "cells": meta["cells"], "n_cells": int(len(ring)),
            "ring_frac": [r4((ring <= n).mean()) for n in RINGS],
            "p_near": pct(pn), "rank": pct(rank),
            "rank_top1": r4((rank == 1).mean()), "rank_top5": r4((rank <= 5).mean()),
            "rank_top20": r4((rank <= 20).mean())}


def gtref(m):
    p = RAW / f"gtref_{m}.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    shutil.copy(p, OUT / p.name)
    col = lambda k: np.array([float(r[k]) for r in rows], np.float64)
    out = {"n_pairs": len(rows), "label_fit_res": pct(col("label_fit_res"))}
    for kind in ("normal", "gt"):
        s = protocol.summarize(col(f"err_{kind}"))
        out[kind] = {k: (r4(v) if not isinstance(v, list) else [r4(x) for x in v]) for k, v in s.items()
                     if k in ("n", "fail_rate", "median", "auc@5", "auc@5_ci", "auc@10", "sr@5")}
    d = col("err_gt") - col("err_normal")
    out["frac_gt_better"] = r4((d < 0).mean())
    out["diff_gt_minus_normal"] = pct(d[np.isfinite(d)])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {"coarse": {"moon": {}, "mega": {}}, "gtref": {}}
    for kind, models in (("moon", MOON), ("mega", MEGA)):
        for m in models:
            c = coarse(kind, m)
            if c:
                summ["coarse"][kind][m] = c
    for m in MOON:
        g = gtref(m)
        if g:
            summ["gtref"][m] = g
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")

    with open(OUT / "rings.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["data", "model", *[f"ring{n}" for n in RINGS]])
        for kind, cs in summ["coarse"].items():
            for m, c in cs.items():
                wr.writerow([kind, m, *c["ring_frac"]])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for kind, ls, name in (("moon", "-", "月球"), ("mega", "--", "MegaDepth")):
        for m, c in summ["coarse"][kind].items():
            ax.plot(list(RINGS), c["ring_frac"], ls, marker="o", ms=3, color=COLOR[m], label=f"{name}：{LABEL[m]}")
    ax.set_xlabel("圈数（候选位置间距）")
    ax.set_ylabel("算对的格子比例")
    ax.set_xticks(list(RINGS))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RUN / "extra" / "rings.png", dpi=150)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
