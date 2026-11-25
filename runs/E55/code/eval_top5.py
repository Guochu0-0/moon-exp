"""Evaluate the predeclared top five K2 and K5 validation checkpoints."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from eval_checkpoint import ROOT, STEPS, evaluate, hash_file, read_json, score, write_json

METHODS = ("k2_no_bce", "k5_no_bce")


def ranks(method):
    path = ROOT / "sweep" / method / "scores"
    rows = []
    for step in STEPS:
        item = read_json(path / f"{step}.json")
        if item["step"] != step or item["summary"]["n"] != 652:
            raise ValueError(f"Invalid Val candidate {method}/{step}")
        rows.append((step, item["summary"]["auc@5"], item["match"]["weights_sha256"]))
    return sorted(rows, key=lambda row: (-row[1], row[0]))[:5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("step", type=int)
    args = parser.parse_args()
    method, step = args.method, args.step
    ranked = ranks(method)
    rank, (_, val, digest) = next((i, row) for i, row in enumerate(ranked, 1) if row[0] == step)
    path = ROOT / "sweep" / method
    dst = path / "top5-test" / f"step{step}"
    if hash_file(ROOT / "ckpt" / method / f"ckpt_{step}.pt") != digest:
        raise ValueError("Checkpoint differs from Val weights")
    peak = read_json(path / "peak.json")
    if step == peak["step"]:
        rec = read_json(ROOT / "launch" / f"{method}_eval.json")[-1]
        if rec["status"] != "ok" or peak["match"]["weights_sha256"] != digest:
            raise ValueError("Existing Test does not match the requested Val weights")
        summary = score(ROOT, method, "test", 953)["summary"]
        if summary != peak["test"]:
            raise ValueError("Existing Test summary differs from recalculated predictions")
        meta = peak["match"]
        source = "verified existing Val-selected Test"
    else:
        with tempfile.TemporaryDirectory(prefix="moon-e55-ranking-") as tmp:
            summary, meta, preds = evaluate(method, step, "test", Path(tmp))
            if meta["weights_sha256"] != digest:
                raise ValueError("Test loaded different weights than Val")
            dst.mkdir(parents=True, exist_ok=True)
            for file in preds.iterdir():
                shutil.copy2(file, dst / file.name)
        source = "top-five Test run"
    write_json(dst / "result.json", {"rank": rank, "step": step, "val_auc5": val,
                                     "test": summary, "match": meta, "source": source})


if __name__ == "__main__":
    main()
