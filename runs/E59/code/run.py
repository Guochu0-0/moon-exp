"""Tracked entry point for RIPE preflight, resumable training and fixed-checkpoint inference."""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib

import numpy as np
import torch

from baselines.adapters.base import to_original
from baselines.data import Data
from baselines.match import sha256
from moonlib import inputs
from moonlib.ransac import fit_affine
from runs.E59.code.predictions import prediction_location, run_record
from runs.E59.code.train import train
from workbench import launch
from workbench.records import PredWriter

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/E59"
CONFIG = Path("runs/E59/configs/main.toml")
ENTRY = "runs.E59.code.run"
BASE = Path("/remote-home/xufang/YGC")


def verify_source(cfg):
    repo = ROOT / cfg["repo"]
    if not (repo / "ripe/model_zoo/vgg_hyper.py").is_file():
        raise RuntimeError(f"Initialize the pinned RIPE submodule at {repo}")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=30).strip()
    dirty = subprocess.check_output(["git", "--no-optional-locks", "-C", str(repo), "status",
                                     "--porcelain", "--untracked-files=no"], text=True, timeout=30)
    if head != cfg["revision"] or dirty:
        raise RuntimeError(f"Expected clean upstream RIPE at {cfg['revision']}; got {head}")
    sys.path.insert(0, str(repo))


def image_features(model, image, cfg):
    from ripe.utils.utils import resize_image

    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None].cuda()
    tensor = resize_image(tensor, min_size=cfg["min_resize"], max_size=cfg["max_resize"])
    try:
        points, descriptors, _ = model.detectAndCompute(tensor, threshold=cfg["threshold"], top_k=cfg["top_k"])
    except RuntimeError as exc:
        if str(exc) == "No keypoints detected":
            return None
        raise
    return to_original(points.cpu().numpy(), *image.shape, *tensor.shape[-2:]), descriptors


@torch.inference_mode()
def infer(data_path, official_weights, checkpoint, cfg, out_dir, method, split):
    from kornia.feature import DescriptorMatcher
    from ripe import vgg_hyper

    data = Data(data_path)
    pair_ids = data.pairs(split)
    if not pair_ids:
        raise RuntimeError(f"Empty evaluation split {split}")
    torch.manual_seed(cfg["inference"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = vgg_hyper(model_path=official_weights)
    if checkpoint != official_weights:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().cuda()
    matcher = DescriptorMatcher("mnn")
    opt_map = inputs.get("optical", cfg["input"]["optical"])
    sar_map = inputs.get("sar", cfg["input"]["sar"])
    with PredWriter(out_dir, method, split) as writer:
        for index, pair in enumerate(pair_ids, 1):
            started = time.monotonic()
            optical = image_features(model, opt_map(data.optical(split, pair)), cfg["inference"])
            sar = image_features(model, sar_map(data.sar(split, pair)), cfg["inference"])
            if optical is None or sar is None:
                matches = np.empty((0, 5))
            else:
                xy0, desc0 = optical
                xy1, desc1 = sar
                _, pairs = matcher(desc0, desc1)
                pairs = pairs.cpu().numpy()
                matches = np.column_stack((xy0[pairs[:, 0]], xy1[pairs[:, 1]], np.ones(len(pairs))))
            matrix, mask, reason = fit_affine(matches, cfg["inference"]["inlier_threshold_px"])
            extra = {"fail": reason} if matrix is None else {"n_inliers": int(mask.sum())}
            writer.write(pair, matrix, matches=matches, sec=time.monotonic() - started, **extra)
            if index % 100 == 0:
                print(f"{method}/{split}: {index}/{len(pair_ids)}", flush=True)
    completed = run_record(out_dir).preds(method, split)
    if set(completed) != set(pair_ids):
        raise RuntimeError(f"Incomplete {method}/{split} prediction coverage")
    return {"pairs": len(completed), "fitted": sum(pred.A is not None for pred in completed.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "train", "infer"))
    parser.add_argument("--seconds", type=int, default=13_200)
    parser.add_argument("--step", type=int)
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--data", type=Path, default=os.environ.get("MOON_DATA", BASE / "dataset/Moon"))
    parser.add_argument("--weights-root", type=Path, default=os.environ.get("MOON_WEIGHTS", BASE / "weights"))
    cli = parser.parse_args()
    if cli.seconds <= 0:
        parser.error("--seconds must be positive")
    with (ROOT / CONFIG).open("rb") as handle:
        cfg = tomllib.load(handle)
    if not cfg["model"]["instance_norm"] or cfg["signal"]["min_keypoints_per_image"] != 16 or cfg["signal"]["descriptor_fallback_below"] != 8:
        raise RuntimeError("Released-model InstanceNorm and upstream matching and descriptor gates are fixed")
    verify_source(cfg["model"])
    weights = cli.weights_root / cfg["model"]["weights"]
    weight_hash = sha256(weights)
    num_updates = cfg["train"]["batches"] // cfg["train"]["grad_accumulation"]
    phase = cli.phase
    inference = dict(cfg["inference"], input=cfg["input"])
    if phase == "infer":
        if cli.step is None or cli.split is None:
            parser.error("infer needs --step and --split")
        scheduled = range(0, num_updates + 1, cfg["train"]["val_every_updates"])
        if cli.step not in scheduled or (cli.split == "test" and cli.step not in (0, num_updates)):
            parser.error("Only scheduled Val checkpoints and the fixed init/final Test are allowed")
        checkpoint = weights if cli.step == 0 else RUN / "ckpt/main" / f"ckpt_{cli.step}.pt"
        out_dir, method = prediction_location(RUN, cli.step, num_updates)
        inference.update(split=cli.split, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        name = "init" if cli.step == 0 else "main" if cli.step == num_updates else "val_sweep"
    else:
        name = "preflight" if phase == "preflight" else "main"
    claims = RUN / ".claims"
    claims.mkdir(exist_ok=True)
    with (claims / f"{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch.begin(RUN, name, [sys.executable, "-m", ENTRY, *sys.argv[1:]], entry=ENTRY,
                     config_file=CONFIG.as_posix(), args=cfg, config=inference,
                     phase=phase, release_weights_sha256=weight_hash,
                     data_path=str(cli.data), release_weights_path=str(weights), seconds=cli.seconds)
        try:
            if phase == "infer":
                if out_dir != RUN:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "exp.toml").write_text('id = "S"\ntitle = "RIPE Val checkpoints"\n', encoding="utf-8")
                result = infer(cli.data, weights, checkpoint, cfg, out_dir, method, cli.split)
            else:
                stop_requested = Event()
                previous_handler = signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
                try:
                    result = train(RUN, cli.data, weights, weight_hash, cfg, smoke=phase == "preflight",
                                   seconds=cli.seconds, stop_requested=stop_requested)
                finally:
                    signal.signal(signal.SIGTERM, previous_handler)
            print(result, flush=True)
        except BaseException as exc:
            launch.end(RUN, name, "fail", error_type=type(exc).__name__, error=str(exc))
            raise
        launch.end(RUN, name, "ok", **result)


if __name__ == "__main__":
    main()
