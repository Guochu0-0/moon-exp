"""Trace native RoMa flow at labelled checkpoints before and after each refiner."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from probe_finest_inference import chosen_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("weights", nargs="+", help="NAME=/path/to/checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--per-roi", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))

    import torch
    from torch.nn.functional import grid_sample
    from baselines.adapters.romatch import RomatchAdapter
    from baselines.data import Data
    from baselines.match import seed_all
    from moonlib import inputs
    from workbench.dataset import Dataset

    if args.device == "cpu":
        torch.set_num_threads(1)
    options = json.loads((args.repo / "configs/baselines/anymatch_roma__minmax.json").read_text())
    opt_map = inputs.get("optical", options["input"]["optical"])
    sar_map = inputs.get("sar", options["input"]["sar"])
    data, labels = Data(args.data), Dataset(args.data)
    pairs = chosen_pairs(data, args.per_roi)
    if args.limit:
        pairs = pairs[:args.limit]
    print(json.dumps({"pairs": pairs, "device": args.device, "rng_seed": options["seed"]}), flush=True)

    for item in args.weights:
        name, weight = item.split("=", 1)
        matcher = RomatchAdapter(args.repo / options["repo"], weight, device=args.device, **options["params"])
        decoder = matcher.model.decoder
        for pair in pairs:
            pil_a, orig_a = matcher._pil(opt_map(data.optical("val", pair)))
            pil_b, orig_b = matcher._pil(sar_map(data.sar("val", pair)))
            opt, sar = labels.checkpoints("val", pair)
            size_a = np.array([orig_a[1], orig_a[0]])
            size_b = np.array([orig_b[1], orig_b[0]])
            sampled_pre = {}
            stages = []

            def sample(flow):
                result = {}
                for idx, (key, src, src_size, dst, dst_size) in enumerate((
                    ("opt_to_sar", opt, size_a, sar, size_b),
                    ("sar_to_opt", sar, size_b, opt, size_a),
                )):
                    points = torch.as_tensor((src + 0.5) * 2 / src_size - 1,
                                             dtype=torch.float32, device=flow.device).reshape(1, 1, -1, 2)
                    output = grid_sample(flow[idx:idx + 1].float(), points, padding_mode="border", align_corners=False)
                    result[key] = (output[0, :, 0].T.cpu().numpy() + 1) * dst_size / 2 - 0.5 - dst
                return result

            def before(scale):
                def hook(_module, inputs):
                    sampled_pre[scale] = sample(inputs[2])
                return hook

            def after(_module, _inputs, outputs):
                result = {}
                for scale, values in outputs.items():
                    if scale not in sampled_pre:
                        continue
                    incoming = sampled_pre[scale]
                    outgoing = sample(values["flow"])
                    result[scale] = {
                        direction: {
                            "before_px": np.median(before, axis=0).tolist(),
                            "after_px": np.median(outgoing[direction], axis=0).tolist(),
                            "delta_px": np.median(outgoing[direction] - before, axis=0).tolist(),
                        }
                        for direction, before in incoming.items()
                    }
                stages.append(result)
                sampled_pre.clear()

            pre_hooks = [refiner.register_forward_pre_hook(before(int(scale))) for scale, refiner in decoder.conv_refiner.items()]
            post_hook = decoder.register_forward_hook(after)
            try:
                seed_all(options["seed"])
                with torch.inference_mode():
                    matcher.model.match(pil_a, pil_b, batched=False, device=args.device)
            finally:
                for hook in (*pre_hooks, post_hook):
                    hook.remove()
            if [set(p) for p in stages] != [{16, 8, 4, 2, 1}, {8, 4, 2, 1}]:
                raise RuntimeError(f"unexpected sequence of decoder passes: {stages}")
            print(json.dumps({"method": name, "pair": pair, "labelled_points": len(opt),
                              "coarse": stages[0], "upsample": stages[1]}), flush=True)
        del decoder, matcher
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
