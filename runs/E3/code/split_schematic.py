"""仿射误差三段分解的示意图（不含数据）：python runs/E3/code/split_schematic.py → extra/split_schematic.png
横轴是标注点上的均方误差，各误差是轴上的位置，相邻位置之间的距离就是一段误差。"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "extra/split_schematic.png"


def bracket(ax, x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y - 0.12, y, y, y - 0.12], color=color, lw=1.8)
    ax.text((x0 + x1) / 2, y + 0.18, text, color=color, fontsize=10.5, ha="center", va="bottom")


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    best, localfit, model, local = 3.2, 4.4, 9.0, 1.9
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.annotate("", (10.3, 0), (-0.2, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.text(10.3, -0.3, "均方误差", fontsize=10, ha="right", va="top")

    for x, c, t in ((0, "#2ca02c", "0\n标注"), (best, "#7f7f7f", "最佳仿射"),
                    (localfit, "#ff7f0e", "局部对应\n拟合的仿射"), (model, "#d62728", "模型仿射")):
        ax.plot([x, x], [-0.12, 0.12], color=c, lw=3)
        ax.text(x, -0.3, t, color=c, fontsize=10.5, ha="center", va="top")

    bracket(ax, 0, best, 0.45, "第三段：任何仿射都去不掉", "#7f7f7f")
    bracket(ax, best, localfit, 0.45, "第二段：\n局部对应不准", "#ff7f0e")
    bracket(ax, localfit, model, 0.45, "第一段：仿射在别处拟合（含交叉项）", "#d62728")

    # 局部对应不是仿射，位置不受约束
    ax.plot(local, -1.25, "D", color="#17becf", ms=8)
    ax.annotate("", (0, -1.25), (local, -1.25), arrowprops=dict(arrowstyle="<->", color="#17becf", lw=1.2, ls="--"))
    ax.text(local + 0.25, -1.25, "局部对应：不是仿射，位置不受约束，可以比最佳仿射还小", color="#17becf",
            fontsize=10, ha="left", va="center")

    ax.text(-0.2, 1.75, "固定的关系：最佳仿射在三个仿射里最小；模型仿射与局部对应拟合的仿射谁大并不固定",
            fontsize=9.5, color="#555555", ha="left")
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-1.6, 2.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
