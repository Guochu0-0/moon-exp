"""Measure point repeatability on the lunar examples plotted in E51.

From moon-exp: python -m runs.E52.code.repeatability ../moon-dataset/data/ground-truth
"""

import argparse
import json
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parents[1]
E51_DATA = RUN.parent / "E51" / "extra" / "data"
TOP_K = 300
RADII_PX = (1, 3, 5)
METHODS = ("superpoint_original", "spsg", "minima_splg", "redfeat", "rift2", "comir")


def affine(source, target):
    design = np.column_stack((source, np.ones(len(source))))
    coeff, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("Evaluation ground truth cannot determine an affine transform")
    return coeff[:2], coeff[2]


def inside(points, size):
    return ((points >= 0) & (points < size)).all(axis=1)


def top_points(points, scores, mask, top_k):
    indices = np.flatnonzero(mask)
    order = np.argsort(-scores[indices], kind="stable")
    return points[indices[order[:top_k]]]


def point_repeatability(optical, sar):
    from scipy.spatial import cKDTree

    if len(optical) + len(sar) == 0:
        return []
    optical_nearest = cKDTree(sar).query(optical)[0]
    sar_nearest = cKDTree(optical).query(sar)[0]
    rows = []
    for radius in RADII_PX:
        optical_hits = int(np.count_nonzero(optical_nearest <= radius))
        sar_hits = int(np.count_nonzero(sar_nearest <= radius))
        rows.append({
            "radius_px": radius,
            "optical_hits": optical_hits,
            "sar_hits": sar_hits,
            "repeatability": (optical_hits + sar_hits) / (len(optical) + len(sar)),
        })
    return rows


def overlap_points(row, detections, linear, shift, top_k=TOP_K):
    pair = row["pair"]
    key = pair.replace("/", "__")
    optical = detections[f"{key}__optical"]
    sar = detections[f"{key}__sar"]
    # Detection coordinates use pixel centers, with (0, 0) at the first center.
    xy_optical, xy_sar = optical[:, :2] + 0.5, sar[:, :2] + 0.5
    size = row["window_xywh"][2:]
    mapped_optical = xy_optical @ linear + shift
    inverse_sar = (xy_sar - shift) @ np.linalg.inv(linear)
    optical_overlap = inside(xy_optical, size) & inside(mapped_optical, size)
    sar_overlap = inside(xy_sar, size) & inside(inverse_sar, size)
    optical_top = top_points(mapped_optical, optical[:, 2], optical_overlap, top_k)
    sar_top = top_points(xy_sar, sar[:, 2], sar_overlap, top_k)
    return optical_top, sar_top, int(optical_overlap.sum()), int(sar_overlap.sum())


def evaluate_geometry(row, detections, linear, shift, *, top_k=TOP_K):
    optical_top, sar_top, optical_count, sar_count = overlap_points(row, detections, linear, shift, top_k)
    return {
        "optical_overlap_points": optical_count,
        "sar_overlap_points": sar_count,
        "optical_top_points": len(optical_top),
        "sar_top_points": len(sar_top),
        "thresholds": point_repeatability(optical_top, sar_top),
    }


def evaluate(row, labels, detections, *, top_k=TOP_K):
    # The label archive stores y upwards.
    points = np.loadtxt(labels, dtype=np.float64, ndmin=2)[:, :4]
    points[:, [1, 3]] *= -1
    source, target = points[:, :2], points[:, 2:]
    linear, shift = affine(source, target)

    holdout_errors = []
    for index in range(len(source)):
        keep = np.arange(len(source)) != index
        holdout_linear, holdout_shift = affine(source[keep], target[keep])
        predicted = source[index] @ holdout_linear + holdout_shift
        holdout_errors.append(float(np.linalg.norm(predicted - target[index])))

    return {
        "pair": row["pair"],
        "evaluation_points": len(source),
        "reference_affine": np.vstack((linear, shift)).T.tolist(),
        "reference_leave_one_out_mean_px": float(np.mean(holdout_errors)),
        "reference_leave_one_out_errors_px": holdout_errors,
        **evaluate_geometry(row, detections, linear, shift, top_k=top_k),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path, help="moon-dataset/data/ground-truth")
    args = parser.parse_args()
    manifest = json.loads((E51_DATA / "moon_manifest.json").read_text(encoding="utf-8"))
    if not (args.ground_truth / "Val").is_dir():
        parser.error("ground_truth must contain the Val label archive")

    # E51's lunar figure shows the first manifest entry from each ROI.
    plotted = {}
    for row in manifest:
        plotted.setdefault(row["roi"], row)

    results = []
    skipped = []
    with np.load(E51_DATA / "moon_superpoint_original.npz") as detections:
        for row in plotted.values():
            labels = args.ground_truth / "Val" / f"{row['pair']}.txt"
            if labels.is_file():
                results.append(evaluate(row, labels, detections))
            else:
                skipped.append({"pair": row["pair"], "reason": "no_evaluation_ground_truth"})
    if not results:
        parser.error("None of the plotted patch pairs have evaluation ground truth")

    output = {"top_k_per_image": TOP_K, "results": results, "skipped": skipped}
    destination = RUN / "extra" / "repeatability.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
