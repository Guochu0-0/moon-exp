"""Compare the original affine with the same checkpoint's scale-1 position correction disabled.

Run only on Val, using fixed pairs from every ROI and the original matching and RANSAC settings.
This diagnostic prints JSON and never writes predictions or formal metrics.
"""
import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def chosen_pairs(data, per_roi):
    groups = defaultdict(list)
    for pair in data.pairs("val"):
        groups[pair.split("/")[0]].append(pair)
    return [
        pairs[min((2 * i + 1) * len(pairs) // (2 * per_roi), len(pairs) - 1)]
        for pairs in groups.values()
        for i in range(min(len(pairs), per_roi))
    ]


def feature_stats(A, cp, pair_error):
    opt, sar = cp
    design = np.c_[opt, np.ones(len(opt))]
    target = np.linalg.lstsq(design, sar, rcond=None)[0].T
    bias = (A - target) @ [256.0, 256.0, 1.0]
    return {
        "affine": A.tolist(),
        "error": float(pair_error(A, opt, sar)),
        "center_bias_xy": bias.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("weights", nargs="+", help="NAME=/path/to/checkpoint")
    parser.add_argument("--per-roi", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-normal", action="store_true", help="Only repeat the masked mode for an existing normal result")
    args = parser.parse_args()
    if args.per_roi < 1 or args.limit < 0:
        parser.error("--per-roi must be positive and --limit nonnegative")

    sys.path.insert(0, str(args.repo))
    import torch
    from baselines.adapters.romatch import RomatchAdapter
    from baselines.data import Data
    from baselines.match import seed_all
    from moonlib import inputs
    from moonlib.ransac import fit_affine
    from workbench.dataset import Dataset
    from workbench.protocol import pair_error

    opts = json.loads((args.repo / "configs/baselines/anymatch_roma__minmax.json").read_text())
    if args.device == "cpu":
        torch.set_num_threads(1)
    data, labels = Data(args.data), Dataset(args.data)
    pairs = chosen_pairs(data, args.per_roi)
    if args.limit:
        pairs = pairs[:args.limit]
    modes = ("no_scale1_position",) if args.skip_normal else ("normal", "no_scale1_position")
    opt_map = inputs.get("optical", opts["input"]["optical"])
    sar_map = inputs.get("sar", opts["input"]["sar"])
    print(json.dumps({"input": opts["input"], "params": opts["params"], "seed": opts["seed"], "pairs": pairs,
                      "device": args.device, "kde": "fp32-blocks" if args.device == "cpu" else "original"}), flush=True)

    for item in args.weights:
        name, weights = item.split("=", 1)
        matcher = RomatchAdapter(args.repo / opts["repo"], weights, device=args.device, **opts["params"])
        if args.device == "cpu":
            # The upstream KDE casts to half, but CPU torch.cdist has no Half kernel.
            # Keep the same queries/references and Gaussian kernel in small FP32 blocks.
            def cpu_kde(x, std=0.1, half=True, down=None):
                x = x.half().float() if half else x.float()
                reference = x[::down] if down is not None else x
                return torch.cat([
                    torch.cdist(block, reference).square_().mul_(-0.5 / std ** 2).exp_().sum(dim=-1)
                    for block in x.split(512)
                ])

            importlib.import_module("romatch.models.matcher").kde = cpu_kde
        refiner = matcher.model.decoder.conv_refiner["1"]
        calls = []

        def omit_position(_module, _inputs, output):
            delta, cert = output
            calls.append(tuple(delta.shape))
            return torch.zeros_like(delta), cert

        for pair in pairs:
            t0 = time.monotonic()
            opt = opt_map(data.optical("val", pair))
            sar = sar_map(data.sar("val", pair))
            cp = labels.checkpoints("val", pair)
            result = {"method": name, "pair": pair}
            for mode in modes:
                seed_all(opts["seed"])
                calls.clear()
                hook = refiner.register_forward_hook(omit_position) if mode == "no_scale1_position" else None
                try:
                    x0, x1, conf = matcher.match(opt, sar)
                finally:
                    if hook is not None:
                        hook.remove()
                affine, inliers, failure = fit_affine(np.column_stack((x0, x1, conf)), 3.0)
                stats = feature_stats(affine, cp, pair_error) if affine is not None else {"failure": failure}
                stats["n_inliers"] = int(inliers.sum()) if inliers is not None else 0
                if mode == "no_scale1_position" and not calls:
                    raise RuntimeError("Scale-1 output hook did not run")
                stats["hook_shapes"] = calls.copy()
                result[mode] = stats
                print(f"{name} {pair} {mode}: {json.dumps(stats)}", file=sys.stderr, flush=True)
            result["seconds"] = time.monotonic() - t0
            print(json.dumps(result), flush=True)
        del refiner, matcher
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
