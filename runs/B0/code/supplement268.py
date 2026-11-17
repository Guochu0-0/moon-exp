"""One-ticket driver for the original SP+LG/RoMa B0 supplement (#268)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from workbench import launch


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare_weights(method, config, weights_root):
    """Pin automatic hub loading to the exact official files recorded in launch."""
    cache = weights_root / "supplement268_torch_home"
    checkpoints = cache / "hub/checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    sources = {"superpoint": weights_root / "official/superpoint/superpoint_v1.pth",
               "lightglue": weights_root / "official/lightglue/superpoint_lightglue_v0-1_arxiv.pth"} if method == "splg" else {
                   "dinov2": weights_root / "torch_home/hub/checkpoints/dinov2_vitl14_pretrain.pth"}
    metadata = {}
    for name, source in sources.items():
        actual = checkpoints / source.name
        source = source.resolve(strict=True)
        if not actual.exists() and not actual.is_symlink():
            actual.symlink_to(source)
        if not actual.exists() or not actual.samefile(source):
            raise RuntimeError("Official weight cache differs from recorded source: " + str(actual))
        metadata[name] = dict(path=str(actual), official_file=str(source), sha256=digest(actual))
    if method == "roma":
        direct = (weights_root / config["weights"]).resolve(strict=True)
        metadata["roma"] = dict(path=str(direct), official_file=str(direct), sha256=digest(direct))
    os.environ["TORCH_HOME"] = str(cache)
    return metadata


def selfcheck(config, weights_root):
    import cv2
    import numpy as np
    from baselines import adapters
    from baselines.data import Data
    from baselines.match import seed_all
    from moonlib.inputs import optical_div255

    seed_all(config["seed"])
    weights = weights_root / config["weights"] if config["weights"] else None
    model = adapters.load(config["adapter"])(repo=ROOT / config["repo"], weights=weights,
                                              device="cuda", **config["params"])
    dataset = Data(os.environ["MOON_DATA"])
    records = []
    for pair in dataset.pairs("val", labelled_only=True)[:3]:
        original = optical_div255(dataset.optical("val", pair))
        shifted = cv2.warpAffine(original, np.float32([[1, 0, 7], [0, 1, -5]]), original.shape[::-1])
        seed_all(config["seed"])
        first, second, _ = model.match(original, shifted)
        affine, inliers = cv2.estimateAffine2D(first, second, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if affine is None:
            raise RuntimeError("Self-check could not estimate an affine: " + pair)
        expected = first + [7, -5]
        residual = np.linalg.norm(second - expected, axis=1)
        # Probe the recovered transform throughout the patch, not only its translation term.
        probe = np.array([[0, 0], [511, 0], [0, 511], [511, 511], [255.5, 255.5]])
        transform_error = np.linalg.norm(probe @ affine[:, :2].T + affine[:, 2] - (probe + [7, -5]), axis=1)
        record = dict(pair=pair, n_matches=len(first), n_inliers=int(inliers.sum()), affine=affine.tolist(),
                      median_match_residual=float(np.median(residual)), max_transform_error=float(transform_error.max()))
        records.append(record)
        print(json.dumps(record), flush=True)
        if transform_error.max() > 1.0:
            raise RuntimeError("Coordinate self-check exceeds 1px: " + pair)
    path = ROOT / "runs/B0/extra/supplement268" / (config["method"] + "_selfcheck.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if config["method"] == "splg":
        effective = {"superpoint": dict(model.extractor.conf.__dict__),
                     "preprocess": dict(model.extractor.preprocess_conf),
                     "lightglue": dict(model.matcher.conf.__dict__)}
    else:
        effective = {"coarse_res": [model.model.h_resized, model.model.w_resized],
                     "upsample_res": list(model.model.upsample_res), "outer_resize": None,
                     "symmetric": model.model.symmetric, "sample_mode": model.model.sample_mode,
                     "sample_thresh": model.model.sample_thresh, "sample_num": 10000,
                     "input": "float tensor, official bicubic antialias resize and ImageNet normalization",
                     "use_custom_corr": config["params"].get("use_custom_corr", True)}
    path.write_text(json.dumps(dict(records=records, effective_config=effective), indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("splg", "roma"))
    parser.add_argument("--stage", choices=("selfcheck", "infer"), default="infer")
    args = parser.parse_args()
    os.chdir(ROOT)
    config_path = Path("configs/baselines") / (args.method + ".json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights_root = Path(os.environ["MOON_WEIGHTS"])
    metadata = prepare_weights(args.method, config, weights_root)
    probe = ROOT / "runs/B0/extra/supplement268" / (args.method + "_selfcheck.json")
    effective = json.loads(probe.read_text(encoding="utf-8"))["effective_config"] if probe.exists() else None
    launch.begin("runs/B0", args.method, [sys.executable, *sys.argv], entry=str(Path(__file__).relative_to(ROOT)),
                 config_file=str(config_path), config=config, args=dict(stage=args.stage, splits=["val", "test"],
                 seed=0, pair_timeout=120, ransac=3.0, raw_out="runs/B0/raw/supplement268", effective_config=effective),
                 weights=metadata, torch_home=os.environ.get("TORCH_HOME"))
    try:
        if args.stage == "selfcheck":
            selfcheck(config, weights_root)
        else:
            if not probe.exists():
                raise RuntimeError("Run the committed coordinate self-check before inference")
            for split in ("val", "test"):
                subprocess.run([sys.executable, "-m", "baselines.match", str(config_path), "--split", split,
                                "--out", "runs/B0/raw/supplement268", "--timeout", "120"], check=True)
                subprocess.run(["/opt/envs/wb/bin/python", "-m", "baselines.fit", "runs/B0/raw/supplement268/" + args.method,
                                "--split", split, "--run", "runs/B0", "--ransac", "3"], check=True)
                raw_meta = ROOT / "runs/B0/raw/supplement268" / args.method / (split + ".meta.json")
                (probe.parent / (args.method + "_" + split + "_meta.json")).write_text(
                    raw_meta.read_text(encoding="utf-8"), encoding="utf-8")
    except BaseException:
        launch.end("runs/B0", args.method, "fail")
        raise
    else:
        launch.end("runs/B0", args.method, "ok")


if __name__ == "__main__":
    main()
