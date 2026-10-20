"""E14 的图：各组粗级格子的熵分布（读 extra/data/summary.json，写 extra/entropy.png）。

    python runs/E14/code/plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path(__file__).resolve().parents[1]
FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 服务器上没有中文字体，放了一份在 gpfs（同 E4）
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LINES = [("moon", "月球光学对 SAR，未训练，无扰动", "C3", "-"),
         ("moon_hard", "月球光学对 SAR，未训练，加扰动（E12）", "C3", "--"),
         ("mega", "MegaDepth 自然图，未训练", "C0", "-"),
         ("optopt_hard", "月球光学对光学，未训练，加同款几何扰动", "C2", "-"),
         ("rns_r2", "月球光学对 SAR，伪标签自训练第 2 轮", "C1", "-")]


def main():
    s = json.loads((RUN / "extra/data/summary.json").read_text(encoding="utf-8"))
    bins = np.array(s["H_bins"])
    mid = np.exp((bins[:-1] + bins[1:]) / 2)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for key, label, c, ls in LINES:
        h = np.array(s["sets"][key]["H_hist"]["all"], float)
        med = s["sets"][key]["cell"]["all"]["H"]["p50"]
        ax.plot(mid, h / h.sum(), color=c, ls=ls, lw=1.6, label=f"{label}（中位 {np.exp(med):.0f}）")
    ax.set_xscale("log")
    ax.set_xlim(1, 4096)
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000, 4096], ["1", "3", "10", "30", "100", "300", "1000", "4096"])
    ax.set_xlabel("每个格子的概率相当于平摊在几个候选位置上（e 的熵次方）")
    ax.set_ylabel("格子占比")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RUN / "extra/entropy.png", dpi=150)


if __name__ == "__main__":
    main()
