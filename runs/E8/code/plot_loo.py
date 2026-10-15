"""本地画图：读 extra/loo.json，重画 extra/loo_x.png（服务器上没有中文字体）。

    python runs/E8/code/plot_loo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "extra"
ORDER = ["loftr_zs", "loftr_q4", "loftr_e1", "roma_zs", "roma_m4", "roma_e1"]
NAMES = {"loftr_zs": "LoFTR zero-shot", "loftr_q4": "LoFTR 无标注训练", "loftr_e1": "LoFTR 标注过拟合",
         "roma_zs": "RoMa zero-shot", "roma_m4": "RoMa 无标注训练", "roma_e1": "RoMa 标注过拟合"}
LINES = ["model", "in_sample", "all_craters", "loo"]
COLORS = {"model": "k", "in_sample": "tab:green", "all_craters": "tab:gray", "loo": "tab:blue"}


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    S = json.loads((OUT / "loo.json").read_text(encoding="utf-8"))
    names = S["settings"]["lines"]
    models = [m for m in ORDER if m in S["models"]]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    w = 0.2
    for j, k in enumerate(LINES):
        st = [S["models"][m]["own"][k] for m in models]
        y = np.array([s["ms_x"] for s in st])
        err = np.array([[v - s["ms_x_ci"][0], s["ms_x_ci"][1] - v] for v, s in zip(y, st)]).T
        ax.bar(np.arange(len(models)) + (j - 1.5) * w, y, w, yerr=err, color=COLORS[k], alpha=0.8,
               label=names[k], capsize=2)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([f"{NAMES[m]}\n（{S['models'][m]['own']['n_points']} 点）" for m in models], fontsize=9)
    ax.set_ylabel("标注点 x 向均方误差（px²）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "loo_x.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
