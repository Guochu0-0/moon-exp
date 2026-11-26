"""无标注 Val 的逐对 RANSAC 内点率（E61 口径）。"""
from __future__ import annotations

import statistics

from moonlib.data_version import INLIER_SELECTION

FIT_FAILURES = ("few_matches", "few_inliers")


def inlier_rate(pred) -> float | None:
    if pred is None or (pred.A is None and pred.fail not in FIT_FAILURES):
        return None
    n = pred.extra.get("n_matches")
    k = pred.extra.get("n_inliers", 0) if pred.A is None else pred.extra.get("n_inliers")
    if type(n) is not int or type(k) is not int or not 0 <= k <= n or (pred.A is not None and k < 3):
        return None
    return k / n if n else 0.0


def evaluate_val(pairs, preds) -> dict:
    values = [inlier_rate(preds.get(p)) for p in pairs]
    missing = sum(p not in preds or preds[p].fail == "not_run" for p in pairs)
    unavailable = values.count(None)
    unknown = set(preds).difference(pairs)
    complete = bool(pairs) and not unavailable and not unknown
    valid = [v for v in values if v is not None]
    return {
        "metric": INLIER_SELECTION,
        "summary": {
            "n": len(pairs),
            "inlier_mean": statistics.mean(valid) if complete else None,
            "inlier_median": statistics.median(valid) if complete else None,
            "fit_failures": sum(p in preds and preds[p].fail in FIT_FAILURES for p in pairs),
            "zero_matches": sum(p in preds and preds[p].extra.get("n_matches") == 0 for p in pairs),
            "execution_errors": unavailable - missing,
        },
        "coverage": {"n_pairs": len(pairs), "n_labelled": 0, "n_pred": len(preds),
                     "missing": missing, "unknown": sorted(unknown)[:20]},
        "inlier_rates": values,
    }
