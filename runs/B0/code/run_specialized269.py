"""Ticket #269: reproducible zero-shot self-check/inference, only new B0 methods.

Run on the allocated server GPU; sources/checkpoints must already be present.
The self-check uses each ReDFeat branch separately as well as the production
cross-branch path. This distinction is diagnostic only, not a tuned variant.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baselines import adapters
from baselines.data import Data
from baselines.match import git_head, seed_all, sha256
from moonlib import inputs
from workbench.launch import begin, end


def selfcheck(config, data_root, weights_root):
    import cv2
    import torch

    repo = Path(config["repo"]).resolve()
    w = config["weights"]
    weights = repo / w[5:] if w.startswith("repo:") else Path(weights_root) / w
    seed_all(config["seed"])
    matcher = adapters.load(config["adapter"])(repo=repo, weights=weights,
                                                device="cuda", **config["params"])
    ds = Data(data_root)
    records = []
    for pair in ds.pairs("val", labelled_only=True)[:3]:
        opt = inputs.optical_div255(ds.optical("val", pair))
        shifted = cv2.warpAffine(opt, np.float32([[1, 0, 7], [0, 1, -5]]), opt.shape[::-1])
        modes = ("cross_branch", "forward1", "forward2") if config["method"] == "redfeat" else ("cross_branch",)
        for mode in modes:
            seed_all(config["seed"])
            original = getattr(matcher, "_features", None)
            if mode != "cross_branch":
                matcher._features = lambda image, ignored, fn=original, branch=mode: fn(image, branch)
            try:
                k0, k1, _ = matcher.match(opt, shifted)
            finally:
                if original is not None:
                    matcher._features = original
            k0, k1 = np.asarray(k0), np.asarray(k1)
            affine, inliers = (cv2.estimateAffine2D(k0, k1, method=cv2.RANSAC,
                                                   ransacReprojThreshold=3.0)
                               if len(k0) >= 3 else (None, None))
            rec = {"pair": pair, "mode": mode, "n_matches": len(k0),
                   "n_inliers": int(inliers.sum()) if inliers is not None else 0,
                   "affine": affine.tolist() if affine is not None else None,
                   "known_translation": [7, -5],
                   "raw_shift_residual_median": float(np.median(np.linalg.norm(k1 - k0 - [7, -5], axis=1))) if len(k0) else None}
            if affine is not None:
                rec["translation_error"] = float(np.linalg.norm(affine[:, 2] - [7, -5]))
                rec["linear_max_error"] = float(np.abs(affine[:, :2] - np.eye(2)).max())
            records.append(rec)
            print(json.dumps(rec), flush=True)
    return {"config": config, "weights_sha256": sha256(weights), "repo_commit": git_head(repo),
            "gpu": torch.cuda.get_device_name(0), "seed": config["seed"],
            "notes": matcher.notes, "checks": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=("redfeat", "fhreg", "vmgga"))
    ap.add_argument("--stage", choices=("selfcheck", "infer"), required=True)
    ap.add_argument("--split", choices=("val", "test"), help="Infer one split per allocated GPU segment")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS"))
    args = ap.parse_args()
    if args.stage == "infer" and args.split is None:
        ap.error("--stage infer requires --split so each split can release its GPU allocation")
    config_path = Path("configs/baselines") / (args.method + ".json")
    config = json.loads(config_path.read_text())
    run = Path("runs/B0")
    launch_method = args.method + "_selfcheck269" if args.stage == "selfcheck" else args.method
    begin(run, launch_method, [sys.executable, *sys.argv], entry=str(Path(__file__)),
          config_file=str(config_path), config=config, args=vars(args),
          precision={"tf32": False, "float32_matmul_precision": "highest"})
    try:
        import torch

        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.stage == "selfcheck":
            result = selfcheck(config, args.data, args.weights_root)
            dest = run / "extra" / ("selfcheck269_" + args.method + ".json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(result, indent=2) + "\n")
        else:
            from baselines.match import main as match_main

            # Same process preserves the explicitly recorded precision settings.
            # Baselines.match supplies the configured per-patch seed and metadata.
            for split in (args.split,):
                match_main([str(config_path), "--split", split, "--data", args.data,
                            "--weights-root", args.weights_root, "--out", str(run / "raw"), "--resume"])
        end(run, launch_method, "ok")
    except BaseException:
        end(run, launch_method, "fail")
        raise


if __name__ == "__main__":
    main()
