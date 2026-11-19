"""Select new annotated Moon Val patch pairs by optical texture within each ROI."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from baselines.data import Data
from moonlib.inputs import optical_div255
from .repeatability import E51_DATA


SIZE = 512
BLOCK = 32
PER_GROUP = 4


def texture_score(image):
    smooth = gaussian_filter(image, sigma=2)
    dy, dx = np.gradient(smooth)
    magnitude = np.hypot(dx, dy)
    blocks = magnitude.reshape(SIZE // BLOCK, BLOCK, SIZE // BLOCK, BLOCK)
    return float(np.median(blocks.mean(axis=(1, 3))))


def select(data, labels):
    previous = json.loads((E51_DATA / "moon_manifest.json").read_text(encoding="utf-8"))
    excluded = {row["pair"] for row in previous}
    groups = defaultdict(list)
    for label in sorted((labels / "Val").glob("ROI_*/*.txt")):
        pair = f"{label.parent.name}/{label.stem}"
        if pair in excluded:
            continue
        image = optical_div255(data.optical("val", pair))
        if image.shape != (SIZE, SIZE) or not data.path("val", pair, "SAR").is_file():
            raise ValueError(f"Missing or invalid patch pair {pair}")
        groups[label.parent.name].append((texture_score(image), pair))

    rows = []
    for roi, group in sorted(groups.items()):
        ranked = sorted(group)
        quartile = len(ranked) // 4
        if quartile < PER_GROUP:
            raise ValueError(f"Insufficient eligible patches in {roi}")
        for category, start in (("low", 0), ("higher", len(ranked) - quartile)):
            for index in range(PER_GROUP):
                rank = start + (2 * index + 1) * quartile // (2 * PER_GROUP)
                score, pair = ranked[rank]
                rows.append({"roi": roi, "pair": pair, "texture_group": category,
                             "texture_score": score, "rank_in_roi": rank + 1,
                             "eligible_in_roi": len(ranked), "window_xywh": [0, 0, SIZE, SIZE]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = select(Data(args.dataset), args.labels)
    if not rows:
        parser.error("No Val patches found in the evaluation label archive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    screened = {row["roi"]: row["eligible_in_roi"] for row in rows}
    print(f"Screened {sum(screened.values())} patches, selected {len(rows)}")


if __name__ == "__main__":
    main()
