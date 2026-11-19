"""Compare both detectors and uniform random points across the archived E51 patches.

From moon-exp: python -m runs.E52.code.compare_e51 ../moon-dataset/data/ground-truth
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .random_baseline import SEED, TRIALS, simulate
from .repeatability import E51_DATA, RADII_PX, RUN, TOP_K, evaluate, evaluate_geometry


DETECTORS = ("superpoint_original", "redfeat")
IDENTITY = [[1, 0, 0], [0, 1, 0]]


def random_reference(size, affine, counts, cache, rng):
    key = (tuple(size), tuple(map(tuple, affine)), counts)
    if key not in cache:
        row = {
            "reference_affine": affine,
            "optical_top_points": counts[0],
            "sar_top_points": counts[1],
        }
        scores = simulate(row, size, rng)
        lower, upper = np.quantile(scores, [0.025, 0.975], axis=0)
        cache[key] = [
            {"radius_px": radius, "mean": float(mean), "central_95_percent": [float(lo), float(hi)]}
            for radius, mean, lo, hi in zip(RADII_PX, scores.mean(axis=0), lower, upper)
        ]
    return cache[key]


def evaluate_domain(domain, group, labels_root, cache, rng, *, detectors, top_k):
    manifest = json.loads((E51_DATA / f"{domain}_manifest.json").read_text(encoding="utf-8"))
    plotted = {}
    for row in manifest:
        plotted.setdefault(row[group], row["pair"])

    results = []
    with ExitStack() as stack:
        archives = {
            detector: stack.enter_context(np.load(E51_DATA / f"{domain}_{detector}.npz"))
            for detector in detectors
        }
        for row in manifest:
            pair = row["pair"]
            size = row["window_xywh"][2:]
            result = {"pair": pair, "in_e51_figure": domain == "whu_urban" or pair in plotted.values()}
            labels = labels_root / "Val" / f"{pair}.txt"
            if domain == "moon" and not labels.is_file():
                result["geometry"] = "missing_evaluation_ground_truth"
                result[f"uniform_full_patch_{top_k}"] = random_reference(
                    size, IDENTITY, (top_k, top_k), cache, rng
                )
                results.append(result)
                continue

            result["geometry"] = "evaluation_points_affine" if domain == "moon" else "published_pixel_grid"
            result["detectors"] = {}
            for detector, archive in archives.items():
                if domain == "moon":
                    measurements = evaluate(row, labels, archive, top_k=top_k)
                    affine = measurements["reference_affine"]
                    result["reference_leave_one_out_mean_px"] = measurements["reference_leave_one_out_mean_px"]
                else:
                    measurements = evaluate_geometry(row, archive, np.eye(2), np.zeros(2), top_k=top_k)
                    affine = IDENTITY
                counts = (measurements["optical_top_points"], measurements["sar_top_points"])
                result["detectors"][detector] = {
                    "optical_top_points": counts[0],
                    "sar_top_points": counts[1],
                    "thresholds": measurements["thresholds"],
                    "uniform_random": random_reference(size, affine, counts, cache, rng),
                }
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--detector", choices=DETECTORS, help="Evaluate only the selected detector")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Maximum keypoints per side in the overlap")
    args = parser.parse_args()
    if not (args.ground_truth / "Val").is_dir():
        parser.error("ground_truth must contain the Val label archive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    cache = {}
    rng = np.random.default_rng(SEED)
    batches = {"moon": "roi", "whu": "scene", "whu_urban": "group"}
    detectors = (args.detector,) if args.detector else DETECTORS
    output = {
        "top_k_per_image": args.top_k,
        "uniform_trials": TRIALS,
        "uniform_seed": SEED,
        "batches": {
            domain: evaluate_domain(domain, group, args.ground_truth, cache, rng,
                                    detectors=detectors, top_k=args.top_k)
            for domain, group in batches.items()
        },
    }
    suffix = "" if args.detector is None and args.top_k == TOP_K else f"_{args.detector or 'both'}_{args.top_k}"
    path = RUN / "extra" / f"e51_comparison{suffix}.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for domain, rows in output["batches"].items():
        print(f"{domain}: {sum('detectors' in row for row in rows)}/{len(rows)} measured")


if __name__ == "__main__":
    main()
