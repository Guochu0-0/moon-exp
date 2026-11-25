"""Summarize full-precision pair errors across the original Val and Test splits."""
import csv
from collections import Counter
import sys

import numpy as np

from reevaluate import DATASET, HISTORY, METHODS, RUN, SPLITS, check_rows, read, require, sha256, write
from workbench import launch, protocol
from workbench.evaluate import jsonable

METRICS = ["auc@3", "auc@5", "auc@10", "sr@3", "sr@5", "sr@10"]


def collect(method, checkpoints):
    splits = {}
    errors = []
    for split in SPLITS:
        path = RUN / "preds" / method / f"{split}.jsonl"
        rows = check_rows(path, split, affine=True)
        log = read(RUN / "launch" / f"{method}__{split}.json")[-1]
        require(log["end"] and log["status"] == "ok", f"{method}/{split}: run unfinished")
        require(log["prediction_sha256"] == sha256(path), f"{path}: prediction changed after launch")
        errors.extend(protocol.pair_error(row["A"], *checkpoints[split][row["pair"]]) for row in rows)
        splits[split] = {
            "mode": log["mode"], "reason": log["reason"], "coverage": len(rows),
            "failures": log["fit_failures"], "prediction": str(path.relative_to(RUN)),
            "source_prediction": log["prediction_source"], "weights": log["weights"], "fit": log["fit"],
            "step": log.get("step"), "successful_updates": log.get("successful_updates"),
            "amp_skips": log.get("amp_skips"), "stop": log.get("stop"),
        }
    val, test = splits["val"], splits["test"]
    require(val["step"] == test["step"], f"{method}: different training steps")
    require([w["sha256"] for w in val["weights"]] == [w["sha256"] for w in test["weights"]],
            f"{method}: different weights across splits")
    return {"status": "scored", "splits": splits, "summary": protocol.summarize(np.asarray(errors))}


def main():
    ids = {split: DATASET.labelled(split) for split in SPLITS}
    all_pairs = set(DATASET.pairs("val")) | set(DATASET.pairs("test"))
    count = sum(map(len, ids.values()))
    require(len(all_pairs) == 1955 and len({p.split("/")[0] for p in all_pairs}) == 13, "Dataset union changed")
    require(count == sum(SPLITS.values()) and not (set(ids["val"]) & set(ids["test"])), "Annotation union changed")
    checkpoints = {split: {p: DATASET.checkpoints(split, p) for p in pairs} for split, pairs in ids.items()}
    launch.begin(RUN, "combined", [sys.executable, *sys.argv], phase="pooled_evaluation")
    try:
        result = {}
        for method in METHODS:
            if method == "no_ema":
                history = HISTORY / "E56/launch/no_ema.json"
                require(read(history)[-1]["status"] == "fail", "No EMA training status changed")
                result[method] = {
                    "status": "training_failed", "splits": {},
                    "source": "E56/no_ema", "source_launch": str(history),
                    "summary": dict.fromkeys(METRICS, 0.0), "zero_convention": True,
                }
            else:
                result[method] = collect(method, checkpoints)
        require(Counter(rec["status"] for rec in result.values()) == {"scored": 28, "training_failed": 1},
                "Expected 29 unique methods including the historical No EMA failure")
        export = RUN / "extra"
        write(export / "combined.json", jsonable({"n_pairs": len(all_pairs), "n_labelled": count,
                                                   "split_counts": SPLITS, "methods": result}))
        with (export / "combined.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            columns = ["method", "status", "step", "successful_updates", "amp_skips", "val_covered", "test_covered",
                       "val_failures", "test_failures", "val_source", "test_source", "weight_sha256", *METRICS]
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for method, record in result.items():
                val, test = (record["splits"].get(split, {}) for split in SPLITS)
                writer.writerow(dict(
                    method=method, status=record["status"], step=val.get("step"), amp_skips=val.get("amp_skips"),
                    successful_updates=val.get("successful_updates"), val_covered=val.get("coverage"),
                    test_covered=test.get("coverage"), val_failures=val.get("failures"), test_failures=test.get("failures"),
                    val_source=val.get("mode"), test_source=test.get("mode"),
                    weight_sha256=";".join(w["sha256"] for w in val.get("weights", [])),
                    **{k: record["summary"][k] for k in METRICS},
                ))
    except BaseException:
        launch.end(RUN, "combined", "fail")
        raise
    launch.end(RUN, "combined", "ok", scored_methods=28, labelled_pairs=count)


if __name__ == "__main__":
    main()
