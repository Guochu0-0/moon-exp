"""Check the Earth results for sensitivity to point budget and a small translation."""

import json

import numpy as np
from scipy.spatial import cKDTree

from .random_baseline import SEED
from .repeatability import E51_DATA, RUN


BUDGETS = (300, 1024, 2048, 4096)
RADII = (3, 5)
TRIALS = 1000
SHIFT_RADIUS = 8
SIZE = 512


def nearest_distances(optical, sar, optical_tree, sar_tree, shift):
    return np.concatenate((sar_tree.query(optical + shift)[0], optical_tree.query(sar - shift)[0]))


def rates(optical, sar, radii=RADII):
    distances = nearest_distances(optical, sar, cKDTree(optical), cKDTree(sar), 0)
    return [float(np.mean(distances <= radius)) for radius in radii]


def heldout_translation(optical, sar):
    # Fit only within one half of a patch and score the other half. A guard
    # band prevents a shift from moving points across the split.
    halves = [
        (optical[optical[:, 0] < SIZE / 2 - SHIFT_RADIUS], sar[sar[:, 0] < SIZE / 2 - SHIFT_RADIUS]),
        (optical[optical[:, 0] >= SIZE / 2 + SHIFT_RADIUS], sar[sar[:, 0] >= SIZE / 2 + SHIFT_RADIUS]),
    ]
    if any(not len(points) for half in halves for points in half):
        return []
    scores = []
    shifts = [(x, y) for x in range(-SHIFT_RADIUS, SHIFT_RADIUS + 1)
              for y in range(-SHIFT_RADIUS, SHIFT_RADIUS + 1)]
    shifts.sort(key=lambda shift: shift[0] ** 2 + shift[1] ** 2)
    for a, b in halves:
        a_tree, b_tree = cKDTree(a), cKDTree(b)
        scores.append([
            float(np.mean(nearest_distances(a, b, a_tree, b_tree, np.array(shift)) <= 3))
            for shift in shifts
        ])
    results = []
    for index, side in enumerate(("left", "right")):
        best = int(np.argmax(scores[index]))
        results.append({
            "fit_side": side,
            "shift_xy_px": shifts[best],
            "heldout_identity": scores[1 - index][0],
            "heldout_shifted": scores[1 - index][best],
        })
    return results


def scores_for_pair(archive, pair):
    key = pair.replace("/", "__")
    optical, sar = [archive[f"{key}__{side}"] for side in ("optical", "sar")]
    optical, sar = [points[np.argsort(-points[:, 2], kind="stable"), :2] for points in (optical, sar)]
    if min(len(optical), len(sar)) < BUDGETS[-1]:
        raise ValueError(f"{pair} has fewer than {BUDGETS[-1]} points on one side")
    return {
        "pair": pair,
        "repeatability": {budget: rates(optical[:budget], sar[:budget]) for budget in BUDGETS},
        "heldout_translation_top300_at_3px": heldout_translation(optical[:300], sar[:300]),
    }


def uniform_reference(budgets, radii):
    """Mean repeatability of independent uniform points on the full patch."""
    rng = np.random.default_rng(SEED)
    return {
        budget: np.mean([rates(*rng.uniform(0, SIZE, size=(2, budget, 2)), radii=radii)
                         for _ in range(TRIALS)], axis=0).tolist()
        for budget in budgets
    }


def main():
    groups = {}
    for group in ("whu", "whu_urban"):
        manifest = json.loads((E51_DATA / f"{group}_manifest.json").read_text(encoding="utf-8"))
        if any(row["window_xywh"][2:] != [SIZE, SIZE] for row in manifest):
            raise ValueError(f"{group} must contain {SIZE} × {SIZE} patches")
        with np.load(E51_DATA / f"{group}_redfeat.npz") as archive:
            groups[group] = [scores_for_pair(archive, row["pair"]) for row in manifest]
    random_means = uniform_reference(BUDGETS, RADII)
    output = {"radii_px": RADII, "uniform_trials": TRIALS, "uniform_seed": SEED,
              "uniform_means": random_means, "groups": groups}
    (RUN / "extra" / "e51_earth_checks.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
