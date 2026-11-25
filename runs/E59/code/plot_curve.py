"""Render E59's validation curve in the training environment."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = [int(row["updates"]) for row in rows]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), layout="constrained")
    for ax, key, title in zip(axes, ("auc_at_5", "failure_rate"), ("AUC@5", "Registration failure rate")):
        ax.plot(steps, [float(row[key]) for row in rows], marker="o", markersize=3)
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Optimizer updates (0 = published weights)")
    fig.savefig(path.parents[1] / "val_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
