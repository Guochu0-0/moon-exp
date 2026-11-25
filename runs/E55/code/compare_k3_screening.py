"""Compare the four K3 certainty and screening conditions on the Val grid."""

import argparse
import json
from pathlib import Path

import numpy as np

from eval_checkpoint import ROOT, STEPS, read_json, write_json
from workbench.dataset import Dataset


CONTRASTS = (
    ("k3", "k3_no_screen"),
    ("k3_no_bce", "k3_no_screen_no_bce"),
    ("k3", "k3_no_bce"),
    ("k3_no_screen", "k3_no_screen_no_bce"),
)


def y_biases(path, labels):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    predictions = {row["pair"]: row["A"] for row in rows}
    if len(rows) != len(predictions) or predictions.keys() != labels.keys():
        raise ValueError(f"Val IDs do not match the labelled set: {path}")
    return {pair: float(np.asarray(A)[1] @ [256.0, 256.0, 1.0] - labels[pair])
            for pair, A in predictions.items() if A is not None}


def summary(values):
    array = np.array(list(values))
    return {"n": len(array), "mean_px": float(array.mean()) if len(array) else None,
            "median_px": float(np.median(array)) if len(array) else None}


def check_step(step, labels):
    models, biases, contrasts = {}, {}, {}
    for method in dict.fromkeys(m for pair in CONTRASTS for m in pair):
        root = ROOT / "sweep" / method
        score_path = root / "scores" / f"{step}.json"
        if not score_path.is_file():
            continue
        biases[method] = y_biases(root / "val" / str(step) / "val.jsonl", labels)
        models[method] = {"auc5": read_json(score_path)["summary"]["auc@5"], **summary(biases[method].values())}
    for left, right in CONTRASTS:
        if left in biases and right in biases:
            delta = [biases[right][p] - biases[left][p] for p in sorted(biases[right].keys() & biases[left].keys())]
            contrasts[f"{right} minus {left}"] = {**summary(delta), "negative_count": sum(d < 0 for d in delta)}
    return {"step": step, "models": models, "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ds = Dataset(args.dataset)
    pairs = ds.labelled("val")
    if len(pairs) != 652:
        raise ValueError("Expected 652 labelled validation pairs")
    labels = {}
    for pair in pairs:
        opt, sar = ds.checkpoints("val", pair)
        target = np.linalg.lstsq(np.c_[opt, np.ones(len(opt))], sar, rcond=None)[0]
        labels[pair] = float([256.0, 256.0, 1.0] @ target[:, 1])

    steps = [check_step(step, labels) for step in STEPS]
    write_json(args.output, {"protocol": "Val signed image-centre y error with success coverage",
                             "val_pairs": len(pairs), "steps": [row for row in steps if row["models"]]})


if __name__ == "__main__":
    main()
