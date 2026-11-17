"""Summarize #269 predictions/raw counts after canonical workbench eval B0.

No alternative metric definitions: copy summaries from metrics.json, which is
written by workbench.protocol. Missing methods remain unmeasured.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    run = Path("runs/B0")
    metrics = json.loads((run / "metrics.json").read_text())
    out = {"protocol": metrics["protocol"], "methods": {}, "counts": {}}
    reference_commit = "ec9ef97a78fd7e3b903c315e8324b2667fc8c1c6"
    before = json.loads(subprocess.check_output(
        ["git", "show", reference_commit + ":runs/B0/metrics.json"], text=True))
    changed = [name for name, data in before["methods"].items() if metrics["methods"].get(name) != data]
    if changed or before["protocol"] != metrics["protocol"]:
        raise RuntimeError("Existing B0 metrics changed: %s" % changed)
    out["legacy_metrics"] = {"reference_commit": reference_commit,
                             "unchanged_methods": len(before["methods"]), "changed_methods": changed}
    for method in ("redfeat", "fhreg", "vmgga", "mifnet_sp", "mifnet_xfeat",
                   "anymatch_roma", "ma_roma", "minima_roma", "anymatch_loftr", "identity"):
        out["methods"][method] = {
            split: {"summary": data["summary"], "coverage": data["coverage"]}
            for split, data in metrics["methods"].get(method, {}).items()
        }
    for split in ("val", "test"):
        pred = run / "preds" / "redfeat" / (split + ".jsonl")
        if not pred.exists():
            continue
        rows = read_lines(pred)
        raw = read_lines(run / "raw" / "redfeat" / (split + ".jsonl"))
        meta = json.loads((run / "raw" / "redfeat" / (split + ".meta.json")).read_text())
        out["counts"][split] = {
            "n_pred": len(rows), "n_raw": len(raw),
            "raw_execution_errors": sum("error" in row for row in raw),
            "fit_status": dict(Counter(row.get("fail") or "ok" for row in rows)),
            "n_matches_median": statistics.median(row.get("n_matches", 0) for row in rows),
            "n_inliers_median": statistics.median(row.get("n_inliers", 0) for row in rows),
            "sec_median": statistics.median(row["sec"] for row in raw),
            "raw_total_sec": meta["total_sec"],
            "weights_sha256": meta["weights_sha256"],
            "inference_commit": meta["commit"], "repo_commit": meta["repo_commit"],
            "same_raw_and_pred_order": [r["pair"] for r in rows] == [r["pair"] for r in raw],
        }
    launch_path = run / "launch" / "redfeat.json"
    launches = json.loads(launch_path.read_text())
    for record in launches:
        split = record.get("args", {}).get("split")
        if split not in out["counts"]:
            continue
        raw_meta_path = run / "raw" / "redfeat" / (split + ".meta.json")
        raw_meta = json.loads(raw_meta_path.read_text())
        record["env"] = raw_meta["env"]
        record["weights_sha256"] = raw_meta["weights_sha256"]
        record["repo_commit"] = raw_meta["repo_commit"]
        record["runtime_metadata_source"] = str(raw_meta_path)
    launch_path.write_text(json.dumps(launches, ensure_ascii=False, indent=1) + "\n")
    unavailable = {
        "fhreg": "Official implementation and Google Drive weight exist; download route unavailable in this environment.",
        "vmgga": "Official implementation and Baidu weight link exist; checkpoint has not been obtained.",
        "mifnet_sp": "Official implementation/checkpoints exist; required SD2.1 upstream API returned anonymous HTTP401 and no local server cache was found.",
        "mifnet_xfeat": "Official implementation/checkpoints exist; required SD2.1 upstream API returned anonymous HTTP401 and no local server cache was found.",
    }
    out["unmeasured"] = {m: reason for m, reason in unavailable.items() if not metrics["methods"].get(m)}
    dest = run / "extra" / "baseline269_results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    fields = ("n", "fail_rate", "median", "auc@3", "auc@5", "auc@10", "sr@3", "sr@5", "sr@10")
    short = {"redfeat": {s: {k: d["summary"][k] for k in fields}
                         for s, d in out["methods"]["redfeat"].items()},
             "counts": out["counts"], "unmeasured": out["unmeasured"], "legacy_metrics": out["legacy_metrics"]}
    print(json.dumps(short, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
