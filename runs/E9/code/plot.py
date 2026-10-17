"""E9 的图（#157）：读 extra/data/summary.csv 与 subset.csv，任何装了 matplotlib 的环境都能跑。

    python runs/E9/code/plot.py

图 1 extra/auc5_vs_level.png：三个轴各一栏，Val AUC@5 随档位（E7 的旧模型淡色细线、E9 重训的模型粗线）。
图 2 extra/retention.png：三个轴各一栏，各档实测 AUC@5 减去同一检查点子集上原始档的 AUC@5（0 = 扰动后没有变差）。
配色同 E7：颜色 = 路线（zero-shot 灰、离线伪标签蓝、在线自监督绿），线型 = 骨干（LoFTR 实线圆点、RoMa 虚线方块）。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "extra"
AXES = {"shift": ("平移（px）", [0, 12, 64, 128]), "rot": ("旋转（度）", [0, 3, 10, 20]), "scale": ("缩放（%）", [0, 3, 10, 20])}
# (实验, 方法) → (图例, 颜色, 线型, 标记, 是否重训)
MODELS = {("E7", "anymatch_loftr"): ("LoFTR zero-shot", "#8c8c8c", "-", "o", False),
          ("E7", "anymatch_roma"): ("RoMa zero-shot", "#8c8c8c", "--", "s", False),
          ("E7", "P10"): ("LoFTR 离线伪标签（旧，P10）", "#1f77b4", "-", "o", False),
          ("E7", "Q4"): ("LoFTR 在线自监督（旧，Q4）", "#2ca02c", "-", "o", False),
          ("E7", "erase_geo__r2"): ("RoMa 离线伪标签（旧，N）", "#1f77b4", "--", "s", False),
          ("E9", "lns_r4"): ("LoFTR 离线伪标签（重训）", "#1f77b4", "-", "o", True),
          ("E9", "q4g"): ("LoFTR 在线自监督（重训）", "#2ca02c", "-", "o", True),
          ("E9", "rns_r2"): ("RoMa 离线伪标签（重训）", "#1f77b4", "--", "s", True)}


def level(a, lv):
    return "orig" if lv == 0 else f"{a}{lv}"


def style(new, c):
    return dict(color=c, lw=2.4 if new else 1.2, alpha=1.0 if new else 0.45, ms=5 if new else 3.5)


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    with open(OUT / "data" / "summary.csv", encoding="utf-8") as f:
        got = {(r["exp"], r["method"], r["level"]): float(r["auc@5"]) for r in csv.DictReader(f) if r["split"] == "val"}
    with open(OUT / "data" / "subset.csv", encoding="utf-8") as f:
        sub = {(r["exp"], r["method"], r["level"]): float(r["orig_on_subset_auc5"])
               for r in csv.DictReader(f) if r["split"] == "val"}
    models = [k for k in MODELS if all((*k, level(a, lv)) in got for a, (_, ls) in AXES.items() for lv in ls)]

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (a, (lab, ls)) in zip(axs, AXES.items()):
        for k in models:
            name, c, st, mk, new = MODELS[k]
            ax.plot(ls, [got[(*k, level(a, lv))] for lv in ls], st, marker=mk, label=name, **style(new, c))
        ax.set_xticks(ls)
        ax.set_xlabel(lab)
        ax.set_ylim(0, 0.37)
        ax.grid(alpha=.3)
    axs[0].set_ylabel("Val AUC@5")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT / "auc5_vs_level.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (a, (lab, ls)) in zip(axs, AXES.items()):
        ax.axhline(0, color="black", lw=.8)
        for k in models:
            name, c, st, mk, new = MODELS[k]
            ys = [0.0] + [got[(*k, level(a, lv))] - sub[(*k, level(a, lv))] for lv in ls[1:]]
            ax.plot(ls, ys, st, marker=mk, label=name, **style(new, c))
        ax.set_xticks(ls)
        ax.set_xlabel(lab)
        ax.grid(alpha=.3)
    axs[0].set_ylabel("Val AUC@5 − 同子集原始档")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT / "retention.png", dpi=130)
    plt.close(fig)
    print("models:", [MODELS[k][0] for k in models])


if __name__ == "__main__":
    main()
