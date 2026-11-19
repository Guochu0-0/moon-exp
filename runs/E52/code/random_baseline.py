"""Simulate independent uniform detections on the E52 patch pairs.

From moon-exp: python -m runs.E52.code.random_baseline
"""

import json

import numpy as np

from .repeatability import E51_DATA, RADII_PX, RUN, inside, point_repeatability


TRIALS = 5000
SEED = 20261119


def sample_overlap(rng, count, size, inverse, shift):
    """Sample uniformly in the common footprint, in SAR pixel coordinates."""
    points = np.empty((count, 2))
    filled = 0
    while filled < count:
        candidates = rng.uniform(0, size, size=(count - filled, 2))
        shared = candidates[inside((candidates - shift) @ inverse, size)]
        points[filled:filled + len(shared)] = shared
        filled += len(shared)
    return points


def simulate(row, size, rng):
    affine = np.asarray(row["reference_affine"])
    inverse = np.linalg.inv(affine[:, :2].T)
    shift = affine[:, 2]
    scores = np.empty((TRIALS, len(RADII_PX)))
    for trial in range(TRIALS):
        optical = sample_overlap(rng, row["optical_top_points"], size, inverse, shift)
        sar = sample_overlap(rng, row["sar_top_points"], size, inverse, shift)
        scores[trial] = [threshold["repeatability"] for threshold in point_repeatability(optical, sar)]
    return scores


def summarize(scores, observed):
    lower, upper = np.quantile(scores, [0.025, 0.975], axis=0)
    return [
        {
            "radius_px": radius,
            "superpoint": float(actual),
            "uniform_mean": float(mean),
            "uniform_central_95_percent": [float(lo), float(hi)],
            "uniform_trials_at_or_above_superpoint": int(np.count_nonzero(scores[:, index] >= actual)),
        }
        for index, (radius, actual, mean, lo, hi) in enumerate(
            zip(RADII_PX, observed, scores.mean(axis=0), lower, upper)
        )
    ]


def main():
    extra = RUN / "extra"
    measured = json.loads((extra / "repeatability.json").read_text(encoding="utf-8"))["results"]
    manifest = json.loads((E51_DATA / "moon_manifest.json").read_text(encoding="utf-8"))
    sizes = {row["pair"]: row["window_xywh"][2:] for row in manifest}
    rng = np.random.default_rng(SEED)
    observed = []
    simulated = []
    results = []
    for row in measured:
        actual = [threshold["repeatability"] for threshold in row["thresholds"]]
        trials = simulate(row, sizes[row["pair"]], rng)
        observed.append(actual)
        simulated.append(trials)
        results.append({"pair": row["pair"], "thresholds": summarize(trials, actual)})

    output = {
        "seed": SEED,
        "trials_per_pair": TRIALS,
        "results": results,
        "mean_across_pairs": summarize(np.mean(simulated, axis=0), np.mean(observed, axis=0)),
    }
    (extra / "random_baseline.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
