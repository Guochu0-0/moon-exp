"""Run the two B0 affine fits on one GDROS forward pass per patch."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from baselines.data import Data
from baselines.match import env_info, seed_all, sha256
from moonlib import inputs, ransac
from workbench.launch import begin, end
from workbench.records import MATCHES_CAP, PredWriter

RUN = ROOT / "runs/B0"
SPLIT_SIZE = {"val": 652, "test": 953}
FIT_NAMES = ("lsr", "ransac")


def configs(weight):
    paths = {fit: RUN / "configs" / f"gdros_{weight}_{fit}.json" for fit in FIT_NAMES}
    return {fit: (path, json.loads(path.read_text(encoding="utf-8"))) for fit, path in paths.items()}


def load_upstream(repo, expected_commit):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    if commit != expected_commit or dirty:
        raise ValueError(f"Expected clean GDROS {expected_commit}, got {commit}: {dirty}")
    sys.path[:0] = [str(repo), str(repo / "core")]
    from configs.default import get_cfg
    from core.GDROSnet.FeatureFormer.gdros import GDROS
    from core.LSRnet.LSmodel import LSR, LSR_

    cfg = get_cfg()
    return GDROS, cfg[cfg.transformer], LSR, LSR_


def grid(size, start, stop, device):
    idx = torch.arange(start, stop, device=device)
    x, y = idx % size, idx // size
    return torch.stack((x, y, torch.ones_like(idx)), dim=-1).float()


def fit_lsr(src, flow, solve):
    # The upstream test fit regresses displacement; adding I gives x -> x + flow.
    affine = solve(src[None], flow[None])[0].cpu().numpy().astype(np.float64)
    affine[:, :2] += np.eye(2)
    affine[:, 2] += 0.5 - affine[:, :2] @ np.full(2, 0.5)
    return affine


def selfcheck(size, start, stop, lsr, solve):
    # Define the known transform in ArcGIS coordinates, then synthesize center-grid flow.
    known = np.array([[0.97, -0.11, 7.3], [0.13, 1.04, -5.9]])
    full_grid = grid(size, 0, size * size, "cuda")
    centers = full_grid[:, :2] + 0.5
    matrix = torch.tensor(known, device="cuda", dtype=centers.dtype)
    flow = centers @ matrix[:, :2].T + matrix[:, 2] - centers
    src, delta = full_grid[start:stop], flow[start:stop]
    matches = torch.cat((src[:, :2], src[:, :2] + delta), dim=1).cpu().numpy()
    a_lsr = fit_lsr(src, delta, solve)
    a_ransac, inliers, reason = ransac.fit_affine(matches, 3.0)
    if reason or not inliers.all():
        raise AssertionError(f"Synthetic RANSAC failed: {reason}")
    upstream = lsr(1, size, size, flow.T.reshape(1, 2, size, size), stop - start)
    errors = {
        "lsr_corner_max_abs": float(np.max(np.abs(a_lsr - known))),
        "ransac_corner_max_abs": float(np.max(np.abs(a_ransac - known))),
        "upstream_lsr_flow_max_abs": float((upstream.reshape(2, -1).T - flow.cpu()).abs().max()),
    }
    if max(errors.values()) > 0.005:
        raise AssertionError(errors)
    return errors


def image(ds, split, pair, kind, mapping, size):
    grey = inputs.get(kind, mapping)(getattr(ds, kind)(split, pair))
    if grey.shape != (size, size):
        raise ValueError(f"{pair} {kind}: expected {size}x{size}, got {grey.shape}")
    return torch.from_numpy(grey).cuda()[None, None].repeat(1, 3, 1, 1) * 255.0


@torch.no_grad()
def infer(model, solve, ds, pairs, split, files, config, src):
    start, stop = config["flow_slice"]
    size = config["size"]
    for index, pair in enumerate(pairs, 1):
        t0 = perf_counter()
        opt = image(ds, split, pair, "optical", config["input"]["optical"], size)
        sar = image(ds, split, pair, "sar", config["input"]["sar"], size)
        _, flow = model(opt, sar, iters=config["iters"], test_mode=True)
        if flow.shape != (1, 2, size, size):
            raise ValueError(f"{pair}: unexpected final flow shape {flow.shape}")
        delta = flow[0].permute(1, 2, 0).reshape(-1, 2)[start:stop]
        if not torch.isfinite(delta).all():
            for writer in files.values():
                writer.write(pair, None, fail="non_finite_flow")
            continue
        matches = torch.cat((src[:, :2], src[:, :2] + delta), dim=1).cpu().numpy()
        # Uniformly sample the archive only. Both estimators see all 200,000 points.
        archive = matches[np.linspace(0, len(matches) - 1, MATCHES_CAP, dtype=int)]
        a_lsr = fit_lsr(src, delta, solve)
        a_ransac, inliers, reason = ransac.fit_affine(matches, config["ransac"]["threshold"])
        common = {"matches": archive, "n_matches": len(matches), "sec_shared": perf_counter() - t0}
        files["lsr"].write(pair, a_lsr, **common)
        files["ransac"].write(pair, a_ransac, fail=reason,
                              n_inliers=int(inliers.sum()) if inliers is not None else 0, **common)
        if index % 25 == 0 or index == len(pairs):
            print(f"{split}: {index}/{len(pairs)} {pair}", flush=True)


def run(args, methods):
    config = methods["lsr"][1]
    if {k: v for k, v in config.items() if k != "fit"} != {
        k: v for k, v in methods["ransac"][1].items() if k != "fit"
    }:
        raise ValueError("Paired configurations must share weights and flow settings")
    repo = args.repo.resolve()
    weights = args.weights_root / config["checkpoint"]
    evidence = {}
    started = []
    status = "fail"
    try:
        for fit, (path, method_config) in methods.items():
            method = path.stem
            begin(RUN, method, [sys.executable, *sys.argv], entry=str(Path(__file__).relative_to(ROOT)),
                  config_file=str(path.relative_to(ROOT)), config=method_config,
                  args={key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()})
            started.append(method)
        if not weights.is_file():
            status = "unmeasured"
            evidence["reason"] = f"Checkpoint unavailable at {weights}"
            print(evidence["reason"], flush=True)
            return
        gdros, cfg, lsr, solve = load_upstream(repo, config["upstream_commit"])
        evidence.update(weights_sha256=sha256(weights), upstream_model_config=json.loads(json.dumps(cfg)),
                        env=env_info(), gpu=torch.cuda.get_device_name(0))
        seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = torch.nn.DataParallel(gdros(cfg))
        try:
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            status = "unmeasured"
            evidence["reason"] = f"Strict load failed: {exc}"
            print(evidence["reason"], flush=True)
            return
        model = model.module.cuda().eval()
        size = config["size"]
        start, stop = config["flow_slice"]
        evidence["synthetic_check"] = selfcheck(size, start, stop, lsr, solve)
        print(json.dumps(evidence), flush=True)
        src = grid(size, start, stop, "cuda")
        ds = Data(args.data)
        pairs = ds.pairs(args.split, labelled_only=True)
        if len(pairs) != SPLIT_SIZE[args.split]:
            raise ValueError(f"Expected {SPLIT_SIZE[args.split]} B0 pairs, got {len(pairs)}")
        scratch = RUN / "ckpt"
        scratch.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=scratch, prefix="gdros276-") as path:
            staging = Path(path)
            with ExitStack() as stack:
                writers = {fit: stack.enter_context(PredWriter(staging, config_path.stem, args.split))
                           for fit, (config_path, _) in methods.items()}
                infer(model, solve, ds, pairs, args.split, writers, config, src)
            for method in started:
                dest = RUN / "preds" / method
                dest.mkdir(parents=True, exist_ok=True)
                for file in (staging / "preds" / method).iterdir():
                    file.replace(dest / file.name)
        status = "ok"
    finally:
        for method in started:
            end(RUN, method, status, **evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weight", choices=("whu", "os"))
    parser.add_argument("split", choices=tuple(SPLIT_SIZE))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=os.environ.get("MOON_DATA"),
                        required="MOON_DATA" not in os.environ)
    args = parser.parse_args()
    run(args, configs(args.weight))


if __name__ == "__main__":
    main()
