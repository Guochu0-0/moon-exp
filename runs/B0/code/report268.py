"""Check complete pair coverage, archived controls, and matching statistics for #268."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from workbench.dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("method", choices=("splg", "roma"))
args = parser.parse_args()
dataset = Dataset(os.environ["MOON_DATA"])
metrics = json.loads((ROOT / "runs/B0/metrics.json").read_text())
main = Path("/remote-home/xufang/YGC/moon-exp/runs/B0")
old_metrics = json.loads((main / "metrics.json").read_text())
result = {}
for split in ("val", "test"):
    expected = set(dataset.labelled(split))
    path = ROOT / "runs/B0/preds" / args.method / (split + ".jsonl")
    records = [json.loads(row) for row in path.read_text().splitlines() if row.strip()]
    pairs = [row["pair"] for row in records]
    assert len(pairs) == len(expected) and set(pairs) == expected, "Incomplete or duplicated pair list"
    raw = ROOT / "runs/B0/raw/supplement268" / args.method / (split + ".jsonl")
    raw_rows = [json.loads(row) for row in raw.read_text().splitlines() if row.strip()]
    assert len(raw_rows) == len(expected) and {row["pair"] for row in raw_rows} == expected
    baseline = {}
    references = ["anymatch_roma", "ma_roma", "minima_roma", "loftr"]
    for reference in references:
        old = main / "preds" / reference / (split + ".jsonl")
        if old.exists():
            old_pairs = {json.loads(row)["pair"] for row in old.read_text().splitlines() if row.strip()}
            assert old_pairs == expected, "Archived baseline pair list differs: " + reference
            baseline[reference] = dict(n_pairs=len(old_pairs), pair_list_equal=True)
    match_counts = np.array([row.get("n_matches", 0) for row in records])
    inlier_counts = np.array([row.get("n_inliers", 0) for row in records])
    result[split] = dict(n_pairs=len(expected), missing=0, duplicate=0, archived_pair_checks=baseline,
                         archived_summary_n={name: value[split]["summary"]["n"] for name, value in old_metrics["methods"].items()
                                             if split in value and name in references},
                         summary=metrics["methods"][args.method][split]["summary"],
                         execution_errors=sum("error" in row for row in raw_rows),
                         failure_reasons=dict(Counter(row.get("fail", "ok") for row in records)),
                         n_matches_median=float(np.median(match_counts)),
                         n_inliers_median=float(np.median(inlier_counts)),
                         matching_sec=sum(row["sec"] for row in raw_rows))
output = ROOT / "runs/B0/extra/supplement268" / (args.method + "_report.json")
output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print(json.dumps(result, indent=1))
