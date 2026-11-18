"""Render both detectors on the first fixed patch of each scene and ROI."""
import json
import sys
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines.tif import read_tif
from moonlib import inputs
from runs.E51.code.extract_moon import ROOT, RUN, SIZE
from runs.E51.code.extract_superpoint import read_whu
from workbench import launch

METHODS = (
    ("redfeat", "ReDFeat", "#46edce"),
    ("superpoint_original", "Original SuperPoint", "#ffb548"),
)
MODALITIES = ("optical", "sar")
DATA = RUN / "extra" / "data"


def examples(domain, group):
    rows = json.loads((DATA / f"{domain}_manifest.json").read_text(encoding="utf-8"))
    first_by_group = {}
    for row in rows:
        first_by_group.setdefault(row[group], row)
    if len(first_by_group) != 4:
        raise ValueError(f"Expected four groups in the fixed {domain} manifest")
    return list(first_by_group.values())


def render(domain, dataset_name, rows, read_image):
    figure, axes = plt.subplots(len(rows), 4, figsize=(13.8, 13.2), dpi=150)
    figure.subplots_adjust(left=.105, right=.992, top=.927, bottom=.012, wspace=.034, hspace=.038)
    figure.text(figure.subplotpars.left, .972, dataset_name, fontsize=22, weight="bold", color="#17242a", va="center")
    try:
        with ExitStack() as stack:
            archives = {
                name: stack.enter_context(np.load(DATA / f"{domain}_{name}.npz", allow_pickle=False))
                for name, _, _ in METHODS
            }
            for i, item in enumerate(rows):
                pair = item["pair"]
                images = {m: read_image(item, m) for m in MODALITIES}
                axes[i, 0].text(-.055, .5, pair.replace("/", "\n"), transform=axes[i, 0].transAxes,
                                ha="right", va="center", fontsize=10, color="#334349")
                for j, (method, label, color) in enumerate(METHODS):
                    for k, modality in enumerate(MODALITIES):
                        ax = axes[i, 2 * j + k]
                        points = archives[method][pair.replace("/", "__") + "__" + modality]
                        image = images[modality]
                        if image.shape != (SIZE, SIZE) or points.ndim != 2 or points.shape[1] != 3:
                            raise ValueError(f"Invalid patch or keypoints for {pair}/{modality}/{method}")
                        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                        ax.scatter(points[:, 0], points[:, 1], s=2.0, color=color, alpha=.8, linewidths=0)
                        ax.set(xlim=(-.5, SIZE - .5), ylim=(SIZE - .5, -.5))
                        ax.set_axis_off()
                        if i == 0:
                            mod = "Optical" if modality == "optical" else "SAR"
                            ax.set_title(f"{label} · {mod}", fontsize=12, color="#17242a", pad=5)
                        ax.text(.97, .035, f"{len(points):,} points", transform=ax.transAxes, ha="right",
                                va="bottom", color="white", fontsize=10,
                                bbox={"facecolor": "#111d23", "alpha": .83, "edgecolor": "none", "pad": 4})
        target = RUN / "extra" / f"{domain}_detections.png"
        temporary = target.with_suffix(".tmp")
        with BytesIO() as buffer:
            figure.savefig(buffer, format="png", facecolor="white", pil_kwargs={"optimize": True})
            temporary.write_bytes(buffer.getbuffer())
        temporary.replace(target)
        print(target.relative_to(RUN), [row["pair"] for row in rows], flush=True)
    finally:
        plt.close(figure)


def main():
    cfg = json.loads((ROOT / "configs/baselines/splg.json").read_text(encoding="utf-8"))
    mapping = {m: inputs.get(m, cfg["input"][m]) for m in MODALITIES}

    def read_moon(item, modality):
        return mapping[modality](read_tif(Path(item["paths"][modality])))

    launch.begin(RUN, "plot_detections", [sys.executable, "-m", "runs.E51.code.plot_detections"])
    try:
        render("moon", "Moon · Val", examples("moon", "roi"), read_moon)
        render("whu", "Earth · WHU-OPT-SAR", examples("whu", "scene"), read_whu)
    except BaseException:
        launch.end(RUN, "plot_detections", "failed")
        raise
    launch.end(RUN, "plot_detections", "ok", matplotlib=matplotlib.__version__)


if __name__ == "__main__":
    main()
