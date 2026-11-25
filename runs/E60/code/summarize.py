"""Pool original Val/Test errors and export the Val and training trajectories."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from workbench import launch, protocol
from workbench.dataset import Dataset
from workbench.evaluate import evaluate_run, jsonable, pair_errors, write_metrics
from workbench.records import Run

RUN = Path(__file__).resolve().parents[1]
STAGES = ("extractor_final", "matcher_final")
METHODS = ("init", *STAGES)
PASSES = (1, 2, 4, 6, 8)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def last_run(method, phase, split=None, observation=None):
    records = read_json(RUN / "launch" / f"{method}.json")
    record = next((row for row in reversed(records) if row["phase"] == phase and row["split"] == split and
                   (observation is None or row["observation_pass"] == observation)), None)
    if not record or not record["end"] or record["status"] != "ok":
        raise RuntimeError(f"Missing completed record for {method}, {phase}, {split}, pass {observation}")
    return record


def evaluate(ds, run, method, split, observation=8):
    launch_name = method if observation == 8 else f"{run.dir.parent.name}_sweep"
    record = last_run(launch_name, "infer", split, observation)
    predictions = run.preds(method, split)
    if set(predictions) != set(ds.labelled(split)) or len(predictions) != record["prediction_count"]:
        raise RuntimeError(f"Unexpected {method}/{split} labeled-pair coverage")
    errors = pair_errors(ds, split, predictions)
    if np.count_nonzero(~np.isfinite(errors)) != record["fit_failures"]:
        raise RuntimeError(f"Inconsistent {method}/{split} failure count")
    return errors, record["identity"]


def training(method):
    record = last_run(method, "train")
    if not record["complete"]:
        raise RuntimeError(f"{method} did not finish its training budget")
    options = record["args"]["train"]
    total, updates, microbatches = 0, 0, 0
    recaps, windows = [], []
    current = []
    with (RUN / "ckpt" / method / "log.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            microbatches += 1
            total += len(row["pairs"])
            updates += "grad_norm" in row
            if (total, microbatches, updates) != (row["pairs_seen"], row["microbatches"], row["updates"]):
                raise RuntimeError(f"Discontinuous counters in {method} at microbatch {microbatches}")
            current.extend(row["pairs"])
            if len(current) >= 1000 or total == record["pairs_seen"]:
                recap = recap_pairs(current)
                windows.append(dict(recap, pairs_seen=total, updates=updates, passes=total / record["train_pairs"]))
                recaps.append(recap)
                current = []
    if (total, microbatches, updates) != (record["pairs_seen"], record["microbatches"], record["updates"]):
        raise RuntimeError(f"Train logfile and launch totals disagree at {method}")
    if total != options["epochs"] * record["train_pairs"]:
        raise RuntimeError(f"Wrong total exposure for {method}")
    overall = {key: sum(part[key] for part in recaps) for key in recaps[0]}
    return dict(overall, windows=windows, microbatches=microbatches, updates=updates, precision=record["precision"],
                final_sha256=record["final_sha256"])


def recap_pairs(rows):
    return {"pairs": len(rows), "candidates": sum(row["candidates"] for row in rows),
            "inliers": sum(row["inliers"] for row in rows), "affine_fits": sum(row["geometry"] == "ok" for row in rows),
            "few_keypoints": sum(row["geometry"] == "few_keypoints" for row in rows),
            "few_matches": sum(row["geometry"] == "few_matches" for row in rows),
            "fit_failed": sum(row["geometry"] == "fit_failed" for row in rows),
            "descriptor_fallbacks": sum(row.get("fallback") is True for row in rows),
            "positive_geometry": sum(row["inliers"] > 0 for row in rows)}


def plot(output, validation, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for index, method in enumerate(STAGES):
        col, title = index, ("Stage 1: SuperPoint", "Stage 2: LightGlue")[index]
        rows = validation[method]
        progress = [int(p) for p in rows]
        axes[0, col].plot(progress, [r["auc@5"] for r in rows.values()], "o-", label="AUC@5")
        axes[0, col].plot(progress, [r["fail_rate"] for r in rows.values()], "s-", label="Fit failure rate")
        axes[0, col].set(title=title, ylabel="Original Val", xticks=progress)
        windows = log[method]["windows"]
        passes = [r["passes"] for r in windows]
        secondary = axes[1, col].twinx()
        for key, name in (("candidates", "Candidates / pair"), ("inliers", "Inliers / pair")):
            axes[1, col].plot(passes, [r[key] / r["pairs"] for r in windows], label=name)
        secondary.plot(passes, [r["affine_fits"] / r["pairs"] for r in windows], "--", color="tab:green", label="Fit rate")
        secondary.set(ylim=(-0.02, 1.02), ylabel="Fit rate")
        axes[1, col].set(xlabel="Train passes", ylabel="Mean count", xlim=(0, 8))
        axes[0, col].legend(loc="best")
        secondary.legend(loc="upper right")
        axes[1, col].legend(loc="lower right")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("MOON_DATA", "/remote-home/xufang/YGC/dataset/Moon"))
    args = parser.parse_args()
    ds = Dataset(args.data)
    launch.begin(RUN, "summary", [sys.executable, "-m", "runs.E60.code.summarize", *sys.argv[1:]], data_root=args.data)
    try:
        run = Run(RUN.name, RUN, {}, list(METHODS))
        val_ids, test_ids = (set(ds.labelled(split)) for split in ("val", "test"))
        if len(val_ids) != 652 or len(test_ids) != 953 or val_ids & test_ids:
            raise RuntimeError("The combined #296 evaluation expects 652 Val and 953 disjoint Test labels")
        pooled, finals = {}, {}
        for method in METHODS:
            val, val_id = evaluate(ds, run, method, "val")
            test, test_id = evaluate(ds, run, method, "test")
            if val_id != test_id:
                raise RuntimeError(f"Val and Test used different weights for {method}")
            finals[method] = protocol.summarize(val)
            pooled[method] = protocol.summarize(np.concatenate((val, test)))
        summaries, validation = {}, {}
        for i, method in enumerate(STAGES):
            summaries[method] = training(method)
            observations = RUN / "sweep" / method / "S"
            sweep = Run("S", observations, {}, [f"pass{n}" for n in PASSES[:-1]])
            results = {0: finals[METHODS[i]]}
            for n in PASSES[:-1]:
                errors, _ = evaluate(ds, sweep, f"pass{n}", "val", n)
                results[n] = protocol.summarize(errors)
            results[8] = finals[method]
            validation[method] = results
            write_metrics(sweep, evaluate_run(ds, sweep))
        extra = RUN / "extra"
        extra.mkdir(exist_ok=True)
        result = {"combined": pooled, "val": validation, "training": summaries,
                  "labelled": {"val": len(val_ids), "test": len(test_ids)}}
        (extra / "results.json").write_text(json.dumps(jsonable(result), indent=1, allow_nan=False) + "\n", encoding="utf-8")
        plot(extra / "trends.png", validation, summaries)
    except BaseException as exc:
        launch.end(RUN, "summary", "fail", error=str(exc))
        raise
    launch.end(RUN, "summary", "ok", labelled_pairs=len(val_ids) + len(test_ids))


if __name__ == "__main__":
    main()
