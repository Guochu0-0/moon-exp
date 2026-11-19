"""Evaluate the new lunar texture sample, retaining the archived WHU urban comparison."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .compare_e51 import DETECTORS, random_reference
from .random_baseline import SEED, TRIALS
from .repeatability import RUN, TOP_K, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("points", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    cache = {}
    results = []
    with ExitStack() as stack:
        archives = {
            method: stack.enter_context(np.load(args.points / f"texture_{method}.npz"))
            for method in DETECTORS
        }
        for index, row in enumerate(rows, start=1):
            label = args.labels / "Val" / f"{row['pair']}.txt"
            methods = {}
            for method, archive in archives.items():
                values = evaluate(row, label, archive, top_k=TOP_K)
                counts = (values["optical_top_points"], values["sar_top_points"])
                values["uniform_random"] = random_reference(row["window_xywh"][2:], values["reference_affine"],
                                                           counts, cache, rng)
                methods[method] = values
            results.append({**row, "detectors": methods})
            if index % 8 == 0:
                print(f"Evaluated {index}/{len(rows)} lunar patch pairs", flush=True)

    urban = json.loads((RUN / "extra" / "e51_comparison.json").read_text(encoding="utf-8"))
    if urban["top_k_per_image"] != TOP_K:
        raise ValueError("The WHU urban comparison uses a different point budget")
    result = {"top_k_per_image": TOP_K, "uniform_trials": TRIALS, "uniform_seed": SEED,
              "moon": results, "whu_urban": urban["batches"]["whu_urban"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
