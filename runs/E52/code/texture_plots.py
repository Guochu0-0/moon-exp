"""Render E52 low-texture examples and the 3 px excess over uniform random."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines.data import Data
from moonlib.inputs import optical_div255, sar_p2p98
from .compare_e51 import DETECTORS
from .repeatability import overlap_points


METHOD_LABELS = {"superpoint_original": "SuperPoint", "redfeat": "ReDFeat"}
METHOD_COLORS = {"superpoint_original": "#f4a33c", "redfeat": "#00c8ae"}
GROUP_COLORS = ("#384f80", "#779c87", "#bf6844")


def examples(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["roi"]].append(row)
    return [(min(group, key=lambda r: r["texture_score"]), max(group, key=lambda r: r["texture_score"]))
            for group in grouped.values()]


def setup_grid(nrows, title, subtitle, headers):
    fig, axes = plt.subplots(nrows, len(headers), figsize=(12.2, 2.6 * nrows + 1.0), squeeze=False)
    fig.subplots_adjust(left=.197, right=.988, top=.923, bottom=.016, wspace=.035, hspace=.10)
    fig.text(.028, .983, title, ha="left", va="top", fontsize=22, fontweight="bold", color="#172432")
    fig.text(.028, .954, subtitle, ha="left", va="top", fontsize=11.5, color="#445361")
    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=11, fontweight="semibold", color="#172432", pad=5)
    return fig, axes


def draw_image(ax, image, limits):
    ax.imshow(image, cmap="gray", vmin=limits[0], vmax=limits[1], interpolation="nearest")
    ax.set(xlim=(-.5, 511.5), ylim=(511.5, -.5))
    ax.set_axis_off()


def row_label(ax, text):
    ax.text(-.075, .5, text, transform=ax.transAxes, ha="right", va="center",
            fontsize=10, linespacing=1.5, color="#344655")


def example_figures(results, data, points, output):
    picked = examples(results)
    images = {row["pair"]: (optical_div255(data.optical("val", row["pair"])),
                            sar_p2p98(data.sar("val", row["pair"]))) for pair in picked for row in pair}
    optical_range = np.percentile(np.concatenate([pair[0].ravel() for pair in images.values()]), [2, 98])
    limits = (optical_range, (0, 1))
    fig, axes = setup_grid(len(picked), "Lunar optical texture examples",
                           "Matched by ROI. Optical display ranges are shared. Scores are optical gradients × 1,000.",
                           ["Low texture · Optical", "Low texture · SAR",
                            "Higher texture · Optical", "Higher texture · SAR"])
    for axs, (low, high) in zip(axes, picked):
        low_id, high_id = low["pair"].split("/")[1], high["pair"].split("/")[1]
        row_label(axs[0], f"{low['roi']}\n{low_id} · {low['texture_score'] * 1000:.2f}"
                          f"\n{high_id} · {high['texture_score'] * 1000:.2f}")
        for ax, image, limit in zip(axs, images[low["pair"]] + images[high["pair"]], limits * 2):
            draw_image(ax, image, limit)
    fig.savefig(output / "moon_texture_examples.jpg", dpi=155, pil_kwargs={"quality": 90})
    plt.close(fig)

    fig, axes = setup_grid(len(picked), "Detections on low-texture lunar patches",
                           "Up to 300 points per image in the shared field of view. RR is measured at 3 px.",
                           [f"{METHOD_LABELS[method]} · {side}" for method in DETECTORS for side in ("Optical", "SAR")])
    with ExitStack() as stack:
        archives = {m: stack.enter_context(np.load(points / f"texture_{m}.npz")) for m in DETECTORS}
        for axs, (low, _) in zip(axes, picked):
            pair = low["pair"]
            roi, patch = pair.split("/")
            scores = [100 * low["detectors"][m]["thresholds"][1]["repeatability"] for m in DETECTORS]
            row_label(axs[0], f"{roi}\n{patch}\nRR {scores[0]:.1f}% / {scores[1]:.1f}%")
            for method, ax_pair in zip(DETECTORS, axs.reshape(-1, 2)):
                reference = np.asarray(low["detectors"][method]["reference_affine"])
                linear, shift = reference[:, :2].T, reference[:, 2]
                optical, sar, _, _ = overlap_points(low, archives[method], linear, shift)
                optical = (optical - shift) @ np.linalg.inv(linear)
                for ax, image, limit, xy in zip(ax_pair, images[pair], limits, (optical, sar)):
                    draw_image(ax, image, limit)
                    ax.scatter(xy[:, 0] - .5, xy[:, 1] - .5, s=4, alpha=.85,
                               c=METHOD_COLORS[method], edgecolors="#263541", linewidths=.15)
    fig.savefig(output / "moon_low_texture_detections.jpg", dpi=155, pil_kwargs={"quality": 90})
    plt.close(fig)


def comparison_figure(results, output):
    groups = ([r for r in results["moon"] if r["texture_group"] == "low"],
              [r for r in results["moon"] if r["texture_group"] == "higher"], results["whu_urban"])
    labels = ("Moon · low texture", "Moon · higher texture", "Earth · WHU urban")
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.205, right=.978, top=.755, bottom=.235, wspace=.16)
    fig.text(.035, .963, "Keypoint repeatability above uniform random", ha="left", va="top",
             fontsize=19, fontweight="bold", color="#172432")
    fig.text(.035, .865, "Up to 300 points per side · 3 px tolerance · Dots are patch pairs; diamonds show the mean",
             ha="left", va="top", fontsize=10.5, color="#445361")
    rng = np.random.default_rng(9)
    for ax, method in zip(axes, DETECTORS):
        for y, (rows, color) in enumerate(zip(groups, GROUP_COLORS)):
            scores = [row["detectors"][method] for row in rows]
            values = np.array([score["thresholds"][1]["repeatability"] - score["uniform_random"][1]["mean"]
                               for score in scores]) * 100
            ax.scatter(values, y + rng.uniform(-.17, .17, len(values)), color=color, s=22, alpha=.57, linewidths=0)
            ax.scatter(np.mean(values), y, c=color, marker="D", s=63, edgecolors="white", linewidths=.8, zorder=4)
        ax.axvline(0, lw=1.0, ls="--", color="#697c8e")
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight="bold")
        ax.set_yticks(range(len(groups)), [f"{label} (n={len(rows)})" for rows, label in zip(groups, labels)])
        ax.set_ylim(len(groups) - .55, -.45)
        ax.grid(axis="x", color="#d8dee4", alpha=.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("Repeatability minus random mean (percentage points)", y=.08, fontsize=11)
    fig.savefig(output / "texture_repeatability.png", dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("points", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    example_figures(results["moon"], Data(args.dataset), args.points, args.output)
    comparison_figure(results, args.output)


if __name__ == "__main__":
    main()
