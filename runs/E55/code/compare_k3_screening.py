"""Compare K3 and its no-screen diagnostic on the shared validation grid."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from eval_checkpoint import ROOT, STEPS, read_json, write_json
from workbench.dataset import Dataset


def read_predictions(path, pairs):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    predictions = {row["pair"]: row["A"] for row in rows}
    if len(rows) != len(predictions) or predictions.keys() != pairs:
        raise ValueError(f"Val IDs do not match the labelled set: {path}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ds = Dataset(args.dataset)
    pairs = set(ds.labelled("val"))
    if len(pairs) != 652:
        raise ValueError("Expected 652 labelled validation pairs")
    centre_labels = {}
    for pair in sorted(pairs):
        opt, sar = ds.checkpoints("val", pair)
        centre = np.array([256.0, 256.0, 1.0])
        target = np.linalg.lstsq(np.c_[opt, np.ones(len(opt))], sar, rcond=None)[0]
        centre_labels[pair] = (centre, float(centre @ target[:, 1]))

    results = []
    for step in STEPS:
        roots = [ROOT / "sweep" / method for method in ("k3", "k3_no_screen")]
        score_paths = [root / "scores" / f"{step}.json" for root in roots]
        if not all(path.is_file() for path in score_paths):
            continue
        preds = [read_predictions(root / "val" / str(step) / "val.jsonl", pairs) for root in roots]
        matched = [(pair, centre, y) for pair, (centre, y) in centre_labels.items()
                   if all(mapping[pair] is not None for mapping in preds)]
        row = {"step": step, "pairs_with_two_successful_fits": len(matched),
               "auc5_screened": read_json(score_paths[0])["summary"]["auc@5"],
               "auc5_unscreened": read_json(score_paths[1])["summary"]["auc@5"]}
        if matched:
            offsets = np.array([[np.asarray(mapping[pair])[1] @ centre - y for pair, centre, y in matched]
                                for mapping in preds])
            row.update(screened_y_px=float(np.median(offsets[0])),
                       unscreened_y_px=float(np.median(offsets[1])),
                       unscreened_minus_screened_y_px=float(np.median(offsets[1] - offsets[0])),
                       unscreened_minus_screened_y_negative=int(np.sum(offsets[1] < offsets[0])))
        results.append(row)
    write_json(args.output, {"protocol": "Val; signed image-centre y error; matched successful affine fits", "steps": results})


if __name__ == "__main__":
    main()
