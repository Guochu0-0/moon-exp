"""Report GDROS fitting outcomes and paired B0 checkpoint errors."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from workbench.dataset import Dataset
from workbench.evaluate import pair_errors
from workbench.protocol import TABLE_SR
from workbench.records import load_runs
from workbench.reference import paired_diff

RUN = ROOT / "runs/B0"


def compare(dataset, run, split, names, metrics):
    preds = [run.preds(name, split) for name in names]
    pairs = dataset.labelled(split)
    if any(set(rows) != set(pairs) for rows in preds):
        raise ValueError(f"{split}: expected one prediction row for every B0 labelled pair")
    lsr, ransac = [pair_errors(dataset, split, rows) for rows in preds]
    has_lsr, has_ransac = np.isfinite(lsr), np.isfinite(ransac)
    both = has_lsr & has_ransac
    paired = {pair: float(diff) for pair, diff in zip(np.asarray(pairs)[both], ransac[both] - lsr[both])}
    differences = np.array(list(paired.values()))
    lsr_summary, ransac_summary = [metric["summary"] for metric in metrics]
    delta = paired_diff(ransac, lsr, ransac_summary, lsr_summary)
    keys = ("auc@5", *(f"sr@{t:g}" for t in TABLE_SR))
    return {
        "labelled_pairs": len(pairs),
        "both_succeeded": int(both.sum()),
        "lsr_only": int((has_lsr & ~has_ransac).sum()),
        "ransac_only": int((~has_lsr & has_ransac).sum()),
        "both_failed": int((~has_lsr & ~has_ransac).sum()),
        "auc_and_sr_delta": {key: delta[key] for key in keys},
        "auc_and_sr_delta_ci": {key: delta[key + "_ci"] for key in keys},
        "ransac_minus_lsr_px": {
            "mean": float(differences.mean()) if paired else None,
            "percentiles": dict(zip(("p05", "p25", "p50", "p75", "p95"),
                                    np.percentile(differences, [5, 25, 50, 75, 95]).tolist())) if paired else {},
            "by_pair": paired,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("MOON_DATA"), required="MOON_DATA" not in os.environ)
    args = parser.parse_args()
    ds = Dataset(args.data)
    run = load_runs(RUN.parent)["B0"]
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))["methods"]
    report = {}
    for weight in ("whu", "os"):
        names = [f"gdros_{weight}_{fit}" for fit in ("lsr", "ransac")]
        report[weight] = {}
        for split in ("val", "test"):
            methods = {}
            for name in names:
                path = RUN / "launch" / f"{name}.json"
                entries = json.loads(path.read_text(encoding="utf-8"))
                launch = next(entry for entry in reversed(entries) if entry["args"]["split"] == split)
                if not launch["end"] or launch["status"] not in {"ok", "unmeasured"}:
                    raise ValueError(f"{name}/{split}: launch did not complete")
                result = {"status": launch["status"]}
                if launch["status"] == "ok":
                    evaluation = metrics[name][split]
                    result.update(summary=evaluation["summary"], coverage=evaluation["coverage"],
                                  n_failed=sum(error is None for error in evaluation["errors"]))
                else:
                    result["reason"] = launch["reason"]
                methods[name] = result
            if len({result["status"] for result in methods.values()}) != 1:
                raise ValueError(f"{weight}/{split}: inconsistent statuses for paired fits")
            report[weight][split] = {"methods": methods}
            if all(result["status"] == "ok" for result in methods.values()):
                report[weight][split]["paired"] = compare(ds, run, split, names, list(methods.values()))
    path = RUN / "extra" / "gdros276_results.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    print(path)


if __name__ == "__main__":
    main()
