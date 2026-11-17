"""在已有 matplotlib 的环境中绘制已完成的 Val 曲线；不重新计算评价指标。"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]


def main():
    with open(R / "extra/data/val_curve.csv") as f:
        curve = list(csv.DictReader(f))
    report = json.loads((R / "extra/data/summary.json").read_text())
    steps = [int(r["step"]) for r in curve]
    values = [float(r["auc@5"]) for r in curve]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(steps, values, marker=".", label="SCENES adaptation, seed 0")
    for node, label in (("B0m", "Zero-shot"), ("E41", "E41 last")):
        value = next(r["summary"]["auc@5"] for r in report["results"] if r["node"] == node and r["split"] == "val")
        ax.axhline(value, linestyle="--", linewidth=1, label=label)
    ax.scatter([steps[-1]], [values[-1]], color="black", zorder=3, label="Official last checkpoint")
    ax.set(xlabel="Optimizer updates", ylabel="Original Val AUC@5")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(R / "extra/val_curve.png", dpi=160)
    plt.close(fig)
    (R / "extra/data/plot_env.json").write_text(json.dumps({"python": sys.version, "matplotlib": matplotlib.__version__}, indent=1) + "\n")


if __name__ == "__main__":
    main()
