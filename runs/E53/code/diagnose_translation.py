"""Measure the k4 vertical shift and verify a Val-only correction on untouched Test pairs.

Read existing affine predictions and checkpoint labels. Print diagnostic JSON without
modifying formal predictions, measurements, or weights.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CENTER = np.array([256.0, 256.0, 1.0])


def read_affines(path, expected_count):
    affines = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["pair"] in affines or row["A"] is None:
                raise ValueError(f"{path}: duplicate or failed prediction for {row['pair']}")
            affines[row["pair"]] = np.array(row["A"])
    if len(affines) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} predictions, got {len(affines)}")
    return affines


def affine_to_labels(opt, sar):
    return np.linalg.lstsq(np.c_[opt, np.ones(len(opt))], sar, rcond=None)[0].T


def summarize(ds, split, n, trial_dir, reference_dir, protocol, val_correction=None):
    trial = read_affines(trial_dir / f"{split}.jsonl", n)
    reference = read_affines(reference_dir / f"{split}.jsonl", n)
    ids = sorted(trial)
    if trial.keys() != reference.keys() or set(ids) != set(ds.labelled(split)):
        raise ValueError(f"{split}: prediction IDs differ from the reference or the labelled split")

    targets = [ds.checkpoints(split, pair) for pair in ids]
    true = np.stack([affine_to_labels(*points) for points in targets])
    a4 = np.stack([trial[pair] for pair in ids])
    a5 = np.stack([reference[pair] for pair in ids])
    bias4_y = ((a4 - true) @ CENTER)[:, 1]
    bias5_y = ((a5 - true) @ CENTER)[:, 1]
    delta_y = ((a4 - a5) @ CENTER)[:, 1]
    true_y = (true @ CENTER)[:, 1] - CENTER[1]
    correction = float(-np.median(bias4_y)) if val_correction is None else val_correction

    corrected = a4.copy()
    corrected[:, 1, 2] += correction
    e4 = [protocol.pair_error(a, *target) for a, target in zip(a4, targets)]
    e5 = [protocol.pair_error(a, *target) for a, target in zip(a5, targets)]
    corrected_errors = [protocol.pair_error(a, *target) for a, target in zip(corrected, targets)]

    def auc5(errors):
        return 100 * protocol.auc(np.asarray(errors), 5)
    quartiles = np.array_split(np.argsort(true_y), 4)
    return correction, {
        "pairs": n,
        "original_trial_auc5_percent": auc5(e4),
        "reference_auc5_percent": auc5(e5),
        "diagnostic_corrected_trial_auc5_percent": auc5(corrected_errors),
        "trial_y_bias_median_px": float(np.median(bias4_y)),
        "reference_y_bias_median_px": float(np.median(bias5_y)),
        "trial_minus_reference_y_median_px": float(np.median(delta_y)),
        "trial_minus_reference_y_negative_count": int((delta_y < 0).sum()),
        "true_y_correlation_with_delta": float(np.corrcoef(true_y, delta_y)[0, 1]),
        "trial_minus_reference_y_median_px_by_true_y_quartile": [
            float(np.median(delta_y[indices])) for indices in quartiles
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from workbench import protocol
    from workbench.dataset import Dataset

    ds = Dataset(args.data)
    runs = args.repo / "runs"
    trial = runs / "E53/preds/k4"
    reference = runs / "E41/preds/main__screen0.8"
    shift, val = summarize(ds, "val", 652, trial, reference, protocol)
    _, test = summarize(ds, "test", 953, trial, reference, protocol, val_correction=shift)
    print(json.dumps({"val_fixed_y_correction_px": shift, "val": val, "test": test}, indent=2))


if __name__ == "__main__":
    main()
