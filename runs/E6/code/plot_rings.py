"""画 E6 的图：只读 runs/E6/extra/rings.json（rings.py 的产物），不依赖 cv2 与 E4 代码，在有中文字体的机器上跑。

    python runs/E6/code/plot_rings.py [--fit lsq]

产物：runs/E6/extra/rings_x.png（主图）、rings_quality.png（各圈对应质量）；--fit lsq 时读 rings_lsq.json，只画 rings_lsq_x.png。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "extra"
ORDER = ["loftr_zs", "loftr_q4", "loftr_e1", "roma_zs", "roma_m4", "roma_e1"]
NAMES = {"loftr_zs": "LoFTR zero-shot", "loftr_q4": "LoFTR 无标注训练", "loftr_e1": "LoFTR 标注过拟合",
         "roma_zs": "RoMa zero-shot", "roma_m4": "RoMa 自训练", "roma_e1": "RoMa 标注过拟合"}
RING_NAMES = ["0–16", "16–32", "32–64", "64–128", "≥128"]
REF_STYLE = {"model": ("模型仿射", "k", "-"), "full_all": ("全图全部对应拟合", "tab:gray", "--"),
             "full_40": ("全图随机 40 点拟合", "tab:purple", ":"),
             "local_in_sample": ("标注点处局部对应拟合（样本内）", "tab:green", "-."),
             "label_fit": ("标注拟合的最佳仿射", "tab:red", (0, (1, 3)))}


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", choices=["ransac", "lsq"], default="ransac")
    sfx = "" if ap.parse_args().fit == "ransac" else "_lsq"
    plt = setup_plt()
    S = json.loads((OUT / f"rings{sfx}.json").read_text(encoding="utf-8"))
    models = [m for m in ORDER if m in S["models"]]
    x = np.arange(len(RING_NAMES))

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, m in zip(axs.ravel(), models):
        R = S["models"][m]["nearest"]
        y = [r["ms_x"] for r in R["rings_on_ref_pairs"]]
        lo = [r["ms_x_ci"][0] for r in R["rings_on_ref_pairs"]]
        hi = [r["ms_x_ci"][1] for r in R["rings_on_ref_pairs"]]
        ax.plot(x, y, "-o", color="tab:blue", ms=4, label="各圈 40 点拟合，按到最近坑的距离分圈（95% 区间）")
        ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.15, lw=0)
        y2 = [r["ms_x"] for r in S["models"][m]["own"]["rings_on_ref_pairs"]]
        ax.plot(x, y2, "--s", color="tab:cyan", ms=3, label="各圈 40 点拟合，按到各自坑的距离分圈")
        for k, (lab, c, ls) in REF_STYLE.items():
            ax.axhline(R["refs"][k]["ms_x"], color=c, ls=ls, lw=1.2, label=lab)
        ax.set_title(f"{NAMES[m]}（{R['n_pairs_with_refs']} 对）", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(RING_NAMES)
        ax.set_xlabel("拟合点到陨石坑的距离（px）")
    for ax in axs[:, 0]:
        ax.set_ylabel("标注点 x 向均方误差（px²）")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / f"rings{sfx}_x.png", dpi=120)
    plt.close(fig)
    if sfx:
        return

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    color = {"zs": "tab:blue", "q4": "tab:orange", "m4": "tab:orange", "e1": "tab:red"}
    for m in models:
        R = S["models"][m]["nearest"]["rings"]
        c, ls = color[m.split("_")[1]], "-" if m.startswith("loftr") else "--"
        if m.startswith("loftr"):
            axs[0].plot(x, [r["density_per_1000px2"] for r in R], "o" + ls, color=c, label=NAMES[m], ms=4)
        else:
            axs[1].plot(x, [r["cert_mean"] for r in R], "o" + ls, color=c, label=NAMES[m], ms=4)
        axs[2].plot(x, [r["inlier_rate"] for r in R], "o" + ls, color=c, label=NAMES[m], ms=4)
    for ax, t in zip(axs, ["LoFTR 匹配密度（每 1000 px² 的匹配数）", "RoMa certainty 均值", "各圈 40 点的 RANSAC 内点率"]):
        ax.set_title(t, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(RING_NAMES)
        ax.set_xlabel("到最近陨石坑的距离（px）")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axs[2].set_ylim(0.1, 0.95)
    axs[2].legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "rings_quality.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
