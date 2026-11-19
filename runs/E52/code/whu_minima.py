"""Evaluate B0 MINIMA-LightGlue on all E52 WHU Test windows."""

import argparse
from collections import Counter
import csv
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
from rasterio.windows import Window
import torch

from baselines.adapters.splg import SPLGAdapter
from baselines.match import env_info, seed_all, sha256
from moonlib.ransac import fit_affine
from runs.E51.code.extract_moon import ROOT, SIZE, write_json
from runs.E51.code.extract_superpoint import WEIGHTS_SHA256
from runs.E51.code.extract_whu import map_input
from workbench import launch
from workbench.protocol import auc, pair_error
from .full_data import open_whu, whu_pairs
from .repeatability import RUN, TEST_COUNTS
from .texture_detect import verified_commit

CONFIG = "configs/baselines/minima_splg.json"
THRESHOLD = 3.0
SHIFT_MAGNITUDES = (20, 15)
CHECKPOINTS = np.array([(x, y) for y in (64, 256, 448) for x in (64, 256, 448)])
CONDITIONS = ("native", "translated")


def shifted_window(window, width, height):
    """Choose a real SAR crop whose relative displacement stays within the scene."""
    dx, dy = SHIFT_MAGNITUDES
    if window.col_off < dx:
        dx = -dx
    if window.row_off < dy:
        dy = -dy
    shifted = Window(window.col_off - dx, window.row_off - dy, window.width, window.height)
    if not (0 <= shifted.col_off <= width - window.width
            and 0 <= shifted.row_off <= height - window.height):
        raise ValueError(f"Translated crop outside scene: {shifted}")
    return shifted, (dx, dy)


def evaluate(matcher, opt, sar, shift):
    p, q, confidence = matcher.match(opt, sar)
    if p.shape != q.shape or p.shape != (len(confidence), 2):
        raise ValueError("Inconsistent matcher output")
    matches = np.column_stack((p, q, confidence))
    if not np.isfinite(matches).all():
        raise ValueError("Nonfinite matcher output")

    residuals = np.linalg.norm(p + shift - q, axis=1)
    agreement = int(np.count_nonzero(residuals <= THRESHOLD))
    affine, mask, failure = fit_affine(matches, THRESHOLD)
    error = pair_error(affine, CHECKPOINTS, CHECKPOINTS + shift)
    return {
        "dx": shift[0], "dy": shift[1], "n_matches": len(p),
        "n_grid_agree_3px": agreement, "n_ransac_inliers": 0 if mask is None else int(mask.sum()),
        "failure": failure, "checkpoint_error_px": float(error) if np.isfinite(error) else None,
        "affine": None if affine is None else affine.tolist(),
    }


def run(args, cfg):
    weights = args.weights / cfg["weights"]
    repo = ROOT / cfg["repo"]
    lightglue_commit = verified_commit(repo, cfg["repo"])
    superpoint_hash = sha256(Path(torch.hub.get_dir()) / "checkpoints" / "superpoint_v1.pth")
    if superpoint_hash != WEIGHTS_SHA256:
        raise ValueError("Unexpected SuperPoint checkpoint")
    hash_weights = sha256(weights)
    seed_all(cfg["seed"])
    torch.set_num_threads(2)
    matcher = SPLGAdapter(repo, weights, device=args.device, **cfg["params"])

    manifest = list(whu_pairs(args.dataset))
    expected = TEST_COUNTS["whu"][0]
    if len(manifest) != expected or len({row["pair"] for row in manifest}) != expected:
        raise ValueError("Unexpected WHU Test coverage")
    args.raw.mkdir(parents=True, exist_ok=True)
    for index, (scene, items) in enumerate(groupby(manifest, key=lambda row: row["group"])):
        if index % args.shards != args.shard:
            continue
        rows = list(items)
        destination = args.raw / f"{scene}.json"
        if destination.is_file():
            saved = json.loads(destination.read_text(encoding="utf-8"))
            if [item["pair"] for item in saved] != [row["pair"] for row in rows]:
                raise ValueError(f"Existing scene results differ from the manifest: {scene}")
            print(f"{scene}: reusing {len(rows)} pairs", flush=True)
            continue
        results = []
        with open_whu(rows[0]["paths"]) as sources:
            optical, sar = sources["optical"], sources["sar"]
            for row in rows:
                window = Window(*row["window_xywh"])
                shifted, displacement = shifted_window(window, sar.width, sar.height)
                opt = map_input(optical, "optical", window)[0]
                native = map_input(sar, "sar", window)[0]
                translated = map_input(sar, "sar", shifted)[0]
                results.append({
                    "pair": row["pair"], "scene": scene,
                    "native": evaluate(matcher, opt, native, (0, 0)),
                    "translated": evaluate(matcher, opt, translated, displacement),
                })
        write_json(destination, results)
        print(f"{scene}: {len(results)} pairs", flush=True)
    return {"env": env_info(), "lightglue_commit": lightglue_commit, "weights_sha256": hash_weights,
            "superpoint_sha256": superpoint_hash, "matcher": matcher.notes}


def statistics(rows, condition):
    values = [row[condition] for row in rows]
    matches = np.array([item["n_matches"] for item in values])
    agree = np.array([item["n_grid_agree_3px"] for item in values])
    errors = np.array([np.inf if item["checkpoint_error_px"] is None else item["checkpoint_error_px"]
                       for item in values])
    failures = Counter(item["failure"] for item in values if item["failure"])
    predicted = np.isfinite(errors)
    report = {
        "pairs": len(rows), "affines": int(predicted.sum()), "failures": dict(failures),
        "median_matches": float(np.median(matches)), "no_matches": int(np.count_nonzero(matches == 0)),
        "median_grid_agree_3px": float(np.median(agree)),
        "at_least_3_grid_agree_3px": int(np.count_nonzero(agree >= 3)),
        "grid_agree_fraction_of_matches": float(agree.sum() / matches.sum()) if matches.sum() else None,
        "median_error_solved_px": float(np.median(errors[predicted])) if predicted.any() else None,
        "mean_error_identity_px": float(np.mean([np.hypot(item["dx"], item["dy"]) for item in values])),
    }
    for threshold in (3, 5, 10):
        success = errors <= threshold
        report[f"successful_at_{threshold}px"] = int(success.sum())
        report[f"sr_at_{threshold}px"] = float(success.mean())
        report[f"auc_at_{threshold}px"] = auc(errors, threshold)
    return report


def summarize(raw, output):
    rows = []
    for path in sorted(raw.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    scenes = Counter(row["scene"] for row in rows)
    if len(rows) != TEST_COUNTS["whu"][0] or len({row["pair"] for row in rows}) != len(rows):
        raise ValueError(f"Incomplete or duplicated WHU results: {scenes}")
    if len(scenes) != 20 or set(scenes.values()) != {88}:
        raise ValueError(f"Incomplete scene coverage: {scenes}")
    report = {condition: statistics(rows, condition) for condition in CONDITIONS}
    report["per_scene"] = {
        scene: {condition: statistics([row for row in rows if row["scene"] == scene], condition)
                for condition in CONDITIONS}
        for scene in scenes
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    with (output / "per_pair.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("pair", "scene", "condition", *rows[0]["native"])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for condition in CONDITIONS:
                writer.writerow({"pair": row["pair"], "scene": row["scene"], "condition": condition,
                                 **row[condition]})
    print(json.dumps({condition: report[condition] for condition in CONDITIONS}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match")
    match.add_argument("dataset", type=Path)
    match.add_argument("weights", type=Path)
    match.add_argument("raw", type=Path)
    match.add_argument("--shard", type=int, default=0)
    match.add_argument("--shards", type=int, default=1)
    match.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    report = commands.add_parser("summarize")
    report.add_argument("raw", type=Path)
    report.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.raw, args.output)
        return
    if not 0 <= args.shard < args.shards:
        parser.error("Invalid shard")
    cfg = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    name = f"whu_minima_lightglue_shard{args.shard}"
    launch.begin(RUN, name, [sys.executable, "-m", __spec__.name, *sys.argv[1:]],
                 config_file=CONFIG, config=cfg, shift_magnitudes=SHIFT_MAGNITUDES,
                 ransac_threshold_px=THRESHOLD, checkpoints=CHECKPOINTS.tolist())
    try:
        info = run(args, cfg)
    except BaseException:
        launch.end(RUN, name, "failed")
        raise
    launch.end(RUN, name, "ok", **info)


if __name__ == "__main__":
    main()
