"""Score the fixed RIPE curve and summarize its unlabeled training signal."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

from runs.E59.code.predictions import prediction_location
from workbench import evaluate, protocol
from workbench.dataset import Dataset
from workbench.records import Run

RUN = Path(__file__).resolve().parents[1]
EXTRA = RUN / "extra"


def validation_curve(ds, last, every):
    rows = []
    sweep = Run(RUN / "sweep/main/S")
    metrics = {"protocol": protocol.protocol_info(), "methods": {}}
    expected = set(ds.labelled("val"))
    for update in range(0, last + 1, every):
        out_dir, method = prediction_location(RUN, update, last)
        run = Run(out_dir)
        preds = run.preds(method, "val")
        if set(preds) != expected:
            raise ValueError(f"Incomplete validation predictions at update {update}")
        result = evaluate.evaluate_preds(ds, "val", preds)
        if out_dir == sweep.dir:
            metrics["methods"][method] = {"val": result}
        summary = result["summary"]
        rows.append({"updates": update, "auc_at_5": summary["auc@5"],
                     "failure_rate": summary["fail_rate"], "pairs": len(preds)})
    evaluate.write_metrics(sweep, metrics)
    return rows


def training_diagnostics(log_path, expected_batches, expected_updates):
    groups = {"positive": Counter(), "negative": Counter()}
    count = 0
    with log_path.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            item = json.loads(line)
            if item["batch"] != count or "failure" in item:
                raise ValueError(f"Incomplete or failed training history at batch {count}")
            for pair in item["pairs"]:
                stats = groups["positive" if pair["positive"] else "negative"]
                stats["pairs"] += 1
                stats[pair["geometry"]] += 1
                if pair["candidates"] is None:
                    continue
                stats["matched_pairs"] += 1
                stats["candidate_total"] += pair["candidates"]
                stats["inlier_total"] += pair["inliers"]
                stats["descriptor_fallbacks"] += int(pair["descriptor_fallback"])
    if count != expected_batches or item["updates"] != expected_updates:
        raise ValueError(f"Expected {expected_batches} batches and {expected_updates} updates")
    return {"batches": count, "updates": item["updates"], "groups": groups}


def main():
    with (RUN / "configs/main.toml").open("rb") as handle:
        cfg = tomllib.load(handle)["train"]
    last = cfg["batches"] // cfg["grad_accumulation"]
    ds = Dataset(os.environ.get("MOON_DATA", "/remote-home/xufang/YGC/dataset/Moon"))
    rows = validation_curve(ds, last, cfg["val_every_updates"])
    training = training_diagnostics(RUN / "ckpt/main/log.jsonl", cfg["batches"], last)
    (EXTRA / "data").mkdir(parents=True, exist_ok=True)
    (EXTRA / "data/training_diagnostics.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    path = EXTRA / "data/val_curve.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    subprocess.run(["/opt/envs/loftr/bin/python", str(Path(__file__).with_name("plot_curve.py")), str(path)], check=True)


if __name__ == "__main__":
    main()
