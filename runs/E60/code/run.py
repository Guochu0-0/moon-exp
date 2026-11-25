"""Preflight, train, or predict with a fixed RIPE++ lunar checkpoint."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal as process_signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

try:
    import tomllib
except ModuleNotFoundError:
    from pip._vendor import tomli as tomllib

import numpy as np
import torch

from baselines.adapters.splg import LG_CONF, SP_CONF, SPLGAdapter
from baselines.data import Data
from baselines.match import sha256
from moonlib import inputs
from moonlib.ransac import fit_affine
from runs.E60.code import signal
from runs.E60.code.train import train
from workbench import launch
from workbench.records import PredWriter, Run

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/E60"
ENTRY = "runs.E60.code.run"
BASE = Path("/remote-home/xufang/YGC")
BASELINE = "configs/baselines/minima_splg.json"
LIGHTGLUE_REVISION = "edb2b838efb2ecfe3f88097c5fad9887d95aedad"
MINIMA_SHA256 = "bd981be2f5e8cdab120bb0380d132bf7d2ec20cd61499f92f283660bf2f5b126"
SUPERPOINT_SHA256 = "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
METHODS = ("init", "extractor_final", "matcher_final")


def read_config(method):
    if method == "init":
        return BASELINE, None
    path = f"runs/E60/configs/{method}.toml"
    return path, tomllib.loads((ROOT / path).read_text(encoding="utf-8"))


def model_files(options, baseline):
    repo = ROOT / baseline["repo"]
    revision = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True, timeout=30).strip()
    dirty = subprocess.check_output(["git", "--no-optional-locks", "-C", repo, "status", "--porcelain"],
                                    text=True, timeout=60).strip()
    if revision != LIGHTGLUE_REVISION or dirty:
        raise RuntimeError(f"Expected the clean LightGlue submodule at {LIGHTGLUE_REVISION}, found {revision}")
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    files = {"superpoint": hub / "superpoint_v1.pth", "minima": options.weights / baseline["weights"],
             "lightglue_before_minima_override": hub / "superpoint_lightglue_v0-1_arxiv.pth"}
    hashes = {name: sha256(path) for name, path in files.items()}
    if hashes["minima"] != MINIMA_SHA256:
        raise RuntimeError("MINIMA weight differs from B0/minima_splg")
    if hashes["superpoint"] != SUPERPOINT_SHA256:
        raise RuntimeError("SuperPoint weight differs from the official v1 release")
    return repo, files, hashes


def load_model(repo, files, baseline, extractor=None, matcher=None):
    adapter = SPLGAdapter(repo, files["minima"], **baseline["params"])
    expected = torch.load(files["superpoint"], map_location="cpu", weights_only=True)
    actual = adapter.extractor.state_dict()
    if expected.keys() != actual.keys() or any(not torch.equal(weight, actual[key].cpu()) for key, weight in expected.items()):
        raise RuntimeError("SuperPoint does not exactly match the official release")
    for model, path in ((adapter.extractor, extractor), (adapter.matcher, matcher)):
        if path:
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return adapter


@torch.no_grad()
def verify_init(adapter, repo, files, config, data_root):
    reference = SPLGAdapter(repo, files["minima"], **config["params"])
    data = Data(data_root)
    pairs = data.pairs("val")[:4]
    if len(pairs) != 4:
        raise RuntimeError("Val is missing the pairs for the init comparison")
    differences = []
    for pair in pairs:
        images = [inputs.get(key, config["input"][key])(getattr(data, key)("val", pair))
                  for key in ("optical", "sar")]
        first = adapter.match(*images)
        second = reference.match(*images)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)
        differences.append({"pair": pair, "matches": len(first[0]), "same": True})
    if not any(row["matches"] for row in differences):
        raise RuntimeError("Init verification did not exercise matching")
    coords = np.array([[0., 0.], [21., 0.], [0., 32.], [21., 32.], [50., 0.], [0., 50.],
                       [50., 50.], [74., 32.], [61., 74.], [74., 56.], [42., 76.], [66., 15.]])
    shifted = coords + [17., -3.]
    shifted[-1] += [12., 12.]
    source = torch.from_numpy((coords + .5) * 2 - .5)
    target = torch.from_numpy((shifted + .5) * 2 - .5)
    indices = torch.arange(len(coords)).repeat(2, 1).T
    cfg = {"min_candidates": 8, "threshold_px": 3.0}
    grid = (512, 512, 1024, 1024)
    mask, state = signal.geometry(source, target, indices, grid, grid, cfg)
    if state != "ok" or not mask[:-1].all() or mask[-1]:
        raise RuntimeError("Known affine inlier mask differs on the 512 grid")
    if signal.geometry(source, target, indices[:7], grid, grid, cfg)[1] != "few_matches":
        raise RuntimeError("Training affine attempted fitting fewer than 8 candidates")
    return {"same_as_minima_splg": differences, "synthetic_affine_mask": mask.tolist()}


@torch.inference_mode()
def predict(adapter, baseline, data_root, out, method, split, stop):
    data = Data(data_root)
    ids = data.pairs(split)
    if not ids:
        raise RuntimeError(f"Empty evaluation split {split}")
    maps = {key: inputs.get(key, baseline["input"][key]) for key in ("optical", "sar")}
    torch.manual_seed(baseline["seed"])
    failures = 0
    with PredWriter(out, method, split) as writer:
        for index, pair in enumerate(ids, 1):
            if stop.is_set():
                raise InterruptedError("Evaluation stopped on request")
            started = time.monotonic()
            optical = maps["optical"](data.optical(split, pair))
            sar = maps["sar"](data.sar(split, pair))
            matches = np.column_stack(adapter.match(optical, sar))
            matrix, inliers, reason = fit_affine(matches, 3.0)
            if matrix is None or not np.isfinite(matrix).all():
                matrix = None
                fields = {"fail": reason or "non_finite"}
                failures += 1
            else:
                fields = {"n_inliers": int(inliers.sum())}
            writer.write(pair, matrix, matches=matches, sec=time.monotonic() - started, **fields)
            if index % 100 == 0:
                print(f"{method}/{split} {index}/{len(ids)}", flush=True)
    result = Run(out, {}).preds(method, split)
    if set(result) != set(ids):
        raise RuntimeError(f"Incomplete prediction coverage for {method}/{split}")
    return {"prediction_count": len(result), "fit_failures": failures,
            "prediction_sha256": sha256(out / "preds" / method / f"{split}.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "train", "infer"))
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--pass", dest="observation", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=13_000)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("MOON_DATA", BASE / "dataset/Moon")))
    parser.add_argument("--weights", type=Path, default=Path(os.environ.get("MOON_WEIGHTS", BASE / "weights")))
    cli = parser.parse_args()
    phase, method = cli.phase, cli.method
    config_file, cfg = read_config(method)
    if (phase == "infer") != bool(cli.split) or (phase == "train" and method == "init") or cli.seconds <= 0:
        parser.error("infer needs a split, training needs a trainable method, and seconds must be positive")
    if cli.observation != 8 and (method == "init" or phase != "infer" or cli.split != "val" or
                                 cli.observation not in cfg["train"]["observation_passes"]):
        parser.error("Intermediate training checkpoints are available for scheduled Val observations")

    baseline = json.loads((ROOT / BASELINE).read_text(encoding="utf-8"))
    repo, files, release_hashes = model_files(cli, baseline)
    checkpoints = {"extractor_final": None, "matcher_final": None}
    def weight(stage, observation):
        return RUN / "ckpt" / stage / f"pass_{observation}.pt"

    if method == "matcher_final" and (phase != "preflight" or weight("extractor_final", 8).is_file()):
        checkpoints["extractor_final"] = weight("extractor_final", 8)
    if phase == "infer" and method != "init":
        checkpoints[method] = weight(method, cli.observation)
    identity = {"release": release_hashes, "revision": LIGHTGLUE_REVISION,
                "checkpoints": {k: sha256(v) for k, v in checkpoints.items() if v}}
    stop = Event()
    process_signal.signal(process_signal.SIGTERM, lambda *_: stop.set())
    claims = RUN / ".claims"
    claims.mkdir(exist_ok=True)
    out, label = RUN, method
    name = method
    if phase == "infer" and cli.observation != 8:
        out, label, name = RUN / "sweep" / method / "S", f"pass{cli.observation}", f"{method}_sweep"
    with (claims / f"{method}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch.begin(RUN, name, [sys.executable, "-m", ENTRY, *sys.argv[1:]], entry=ENTRY,
                     config_file=config_file, config={**baseline, "superpoint": SP_CONF, "lightglue": LG_CONF,
                                                      "inference_affine_threshold_px": 3.0},
                     args=cfg, phase=phase, split=cli.split, observation_pass=cli.observation,
                     data_root=str(cli.data), weight_files={k: str(v) for k, v in files.items()},
                     checkpoint_files={k: str(v) for k, v in checkpoints.items() if v}, identity=identity,
                     segment_seconds=cli.seconds)
        try:
            adapter = load_model(repo, files, baseline, *checkpoints.values())
            if method == "init" and phase == "preflight":
                result = verify_init(adapter, repo, files, baseline, cli.data)
            elif phase == "infer":
                if out != RUN:
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "exp.toml").write_text('id = "S"\ntitle = "RIPE++ Val observations"\n', encoding="utf-8")
                result = predict(adapter, baseline, cli.data, out, label, cli.split, stop)
            else:
                result = train(method, adapter, RUN, cli.data, baseline, cfg, identity,
                               smoke=phase == "preflight", seconds=cli.seconds, stop_requested=stop)
                if result["complete"] and phase == "train":
                    result["final_sha256"] = sha256(weight(method, 8))
        except BaseException as exc:
            launch.end(RUN, name, "fail", error_type=type(exc).__name__, error=str(exc))
            raise
        launch.end(RUN, name, "ok", **result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
