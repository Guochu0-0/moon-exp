"""Evaluate #268 methods while preserving all previously archived B0 metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from workbench.__main__ import main
from workbench.evaluate import write_metrics
from workbench.records import load_runs

run_dir = ROOT / "runs/B0"
parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("splg", "roma"), help="Evaluate a completed method before the other one finishes")
args = parser.parse_args()
methods = [args.method] if args.method else ["splg", "roma"]
path = run_dir / "metrics.json"
previous_text = path.read_text(encoding="utf-8")
previous = json.loads(previous_text)
if any(method in previous["methods"] for method in methods):
    raise SystemExit("#268 results already exist; review before overwriting metrics")
try:
    main(["eval", "B0"])
    current = json.loads(path.read_text(encoding="utf-8"))
finally:
    path.write_text(previous_text, encoding="utf-8")
if previous["protocol"] != current["protocol"]:
    raise SystemExit("Protocol changed; refusing to combine different evaluation protocols")
for method in methods:
    result = current["methods"][method]
    if set(result) != {"val", "test"}:
        raise SystemExit("Incomplete split coverage: " + method)
    previous["methods"][method] = result
write_metrics(load_runs(ROOT / "runs")["B0"], previous)
for method in methods:
    for split in ("val", "test"):
        print(method, split, previous["methods"][method][split]["summary"])
