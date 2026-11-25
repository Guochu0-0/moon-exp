"""Collect checked historical predictions and fill missing terminal predictions for #296."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from workbench import launch
from workbench.dataset import Dataset

YGC = Path("/remote-home/xufang/YGC")
HISTORY = YGC / "moon-exp/runs"
WEIGHTS = YGC / "weights"
DATA = YGC / "dataset/Moon"
WB = "/opt/envs/wb/bin/python"
PY = "/opt/envs/loftr/bin/python"
DATASET = Dataset(DATA)
SPLITS = {"val": 652, "test": 953}
BASELINES = (
    "identity", "spsg", "splg", "rift2", "redfeat", "minima_splg",
    "loftr", "xoftr", "minima_xoftr", "minima_eloftr", "minima_loftr",
    "ma_eloftr", "anymatch_edm", "anymatch_loftr", "roma", "gdros_os_lsr",
    "minima_roma", "ma_roma", "anymatch_roma__minmax",
)
TRAINED = {
    **{f"k{k}_no_bce": ("E55", f"k{k}_no_bce") for k in range(6)},
    "frozen_dino": ("E56", "frozen_dino"),
    "no_screen": ("E56", "no_screen"),
    "scenes": ("E50", "main"),
}
METHODS = (*BASELINES, *TRAINED, "no_ema")
INFER = "configs/baselines/anymatch_roma__minmax.json"
ENV = {
    "MOON_DATA": str(DATA), "MOON_WEIGHTS": str(WEIGHTS),
    "TORCH_HOME": str(WEIGHTS / "torch_home"), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
}
REAL_FAILURES = {"few_matches", "few_inliers", "no_affine", "ransac_failed", "degenerate"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labelled(split):
    ids = set(DATASET.labelled(split))
    require(len(ids) == SPLITS[split], f"{split}: unexpected annotation count {len(ids)}")
    return ids


def check_rows(path, split, *, affine):
    data = rows(path)
    ids = labelled(split)
    require(len(data) == len(ids) and {r["pair"] for r in data} == ids, f"{path}: incomplete or duplicate pair IDs")
    bad = []
    for row in data:
        if not affine:
            if "error" in row:
                bad.append(row["pair"])
        elif row["A"] is None:
            if row.get("fail") not in REAL_FAILURES:
                bad.append(f"{row['pair']} ({row.get('fail')})")
        else:
            A = row["A"]
            require(len(A) == 2 and all(len(r) == 3 and all(math.isfinite(v) for v in r) for r in A),
                    f"{path}: malformed affine for {row['pair']}")
    require(not bad, f"{path}: {len(bad)} execution failures, for example {bad[:3]}")
    return data


def checked_weights(path, expected=None):
    digest = sha256(path)
    require(expected is None or digest == expected, f"{path}: SHA-256 differs from historical record")
    return {"path": str(path), "sha256": digest, "historical_sha256": expected}


def baseline_launch(origin, method, split):
    log = origin / "launch" / f"{method}.json"
    for rec in reversed(read(log)):
        args = rec.get("args") or {}
        if rec.get("status") != "ok":
            continue
        if rec.get("split", args.get("split")) == split:
            return log, rec
        if args.get("stage") == "infer" and args.get("splits") == [split]:
            return log, rec
        if method == "anymatch_roma__minmax" and split == "test" and "scripts/finetune/eval_ckpt.sh" in rec["cmd"]:
            return log, rec
    raise ValueError(f"{log}: no successful source for {split}")


def baseline_raw(origin, method, split):
    if method == "anymatch_roma__minmax" and split == "test":
        return origin / f"sweep/{method}/match/{method}"
    area = "baselines_ablation" if origin.name == "B0m" else "baselines"
    return YGC / "results" / area / method


def baseline(method, split):
    origin = HISTORY / ("B0m" if method == "anymatch_roma__minmax" else "B0")
    config_path = f"configs/baselines/{method}.json"
    if method == "gdros_os_lsr":
        config_path = "runs/B0/configs/gdros_os_lsr.json"
    config = read(REPO / config_path)
    path = origin / "preds" / method / f"{split}.jsonl"
    log, record = baseline_launch(origin, method, split)
    raw = baseline_raw(origin, method, split)
    raw_meta = raw / f"{split}.meta.json"
    if method == "anymatch_roma__minmax" and split == "test":
        if raw_meta.is_file():
            meta = read(raw_meta)
            require(meta["config"] == config, f"{raw_meta}: effective config mismatch")
            require(meta["weights_sha256"], f"{raw_meta}: missing evaluated weight SHA-256")
            weights = [checked_weights(WEIGHTS / config["weights"], meta["weights_sha256"])]
        else:
            return config_path, path, None, "inference", "缺少 B0m Test 原始权重哈希", {"source_launch": str(log)}
    else:
        require(record["config"] == config, f"{log}: effective config mismatch")
        args = record.get("args") or {}
        if isinstance(record.get("weights"), dict):
            weights = [checked_weights(Path(item["path"]), item["sha256"])
                       for item in record["weights"].values()]
        elif config.get("weights"):
            weight = config["weights"]
            weight = REPO / config["repo"] / weight[5:] if weight.startswith("repo:") else WEIGHTS / weight
            weights = [checked_weights(weight, record.get("weights_sha256") or args.get("weights_sha256"))]
        elif method == "gdros_os_lsr":
            weights = [checked_weights(Path(args["weights_root"]) / config["checkpoint"], record["weights_sha256"])]
        elif method == "spsg":
            source_commit = "ddcf11f"
            repo = REPO / config["repo"]
            weights = []
            for filename in ("superpoint_v1.pth", "superglue_outdoor.pth"):
                rel = f"models/weights/{filename}"
                expected = subprocess.check_output(["git", "-C", repo, "rev-parse", f"{source_commit}:{rel}"], text=True).strip()
                actual = subprocess.check_output(["git", "-C", repo, "hash-object", rel], text=True).strip()
                require(actual == expected, f"{repo / rel}: bundled weight differs from the recorded upstream commit")
                weights.append({**checked_weights(repo / rel), "historical_git_blob": expected})
        else:
            weights = []
    evidence = {"source_launch": str(log), "source_launch_commit": record["commit"], "weights": weights,
                "fit": "GDROS OS least squares" if method == "gdros_os_lsr" else "RANSAC 3 px"}
    if raw_meta.is_file():
        meta = read(raw_meta)
        require(meta["config"] == config, f"{raw_meta}: effective config mismatch")
        if meta.get("weights_sha256"):
            require(any(w["sha256"] == meta["weights_sha256"] for w in weights), f"{raw_meta}: evaluated weight mismatch")
        evidence["inference_metadata"] = str(raw_meta)
        evidence["resolved_params"] = meta.get("resolved_params")
        evidence["upstream_commit"] = meta.get("repo_commit")
    mode, reason = "reuse", "历史预测与运行、配置和权重记录吻合"
    if method in ("xoftr", "minima_xoftr"):
        mode, reason = "inference", "原匹配器上游存在未提交改动，重新确认有效推理"
    if method == "rift2" and split == "val":
        check_rows(raw / f"{split}.jsonl", split, affine=False)
        mode, reason = "refit", "原 Val 拟合提交缺失，使用原始匹配点对重拟合"
    return config_path, path, raw, mode, reason, evidence


@lru_cache
def terminal(method):
    experiment, name = TRAINED[method]
    home = HISTORY / experiment
    folder = home / "ckpt" / name
    history = rows(folder / "log.jsonl")
    step = history[-1]["step"]
    require([r["step"] for r in history] == list(range(1, step + 1)), f"{folder}: nonconsecutive steps")
    halt_path = folder / "halt.json"
    halt = read(halt_path) if halt_path.is_file() else None
    require((halt and halt["step"] == step) or (folder / "train.done").is_file(), f"{folder}: training did not finish")
    skip = 0
    if experiment == "E50":
        overflow = folder / "overflow.jsonl"
        skip = len(rows(overflow)) if overflow.is_file() else 0
        updates = step
    else:
        scale = 65536.0
        for record in history:
            require("gnorm" in record and "scale" in record, f"{folder}: missing AMP audit fields")
            next_scale = record["scale"]
            require((next_scale < scale) == (not math.isfinite(record["gnorm"])), f"{folder}: inconsistent AMP record")
            skip += next_scale < scale
            scale = next_scale
        updates = step - skip
    checkpoint = folder / f"ckpt_{step}.pt"
    return home, name, step, checkpoint, {"step": step, "successful_updates": updates, "amp_skips": skip,
                                          "stop": halt or "training_budget", "source": f"{experiment}/{name}"}


def trained(method, split):
    home, name, step, checkpoint, evidence = terminal(method)
    evidence = {**evidence, "weights": [checked_weights(checkpoint)], "fit": "RANSAC 3 px"}
    sweep = home / "sweep" / name
    file = meta_path = None
    key = None
    if home.name == "E50":
        file = home / f"preds/{name}/{split}.jsonl"
        meta_path = sweep / f"match/step{step}/{split}.meta.json"
    elif home.name == "E55" and split == "val":
        file = sweep / f"val/{step}/val.jsonl"
        meta_path, key = sweep / f"scores/{step}.json", "match"
    elif method == "k5_no_bce" and split == "test":
        file, meta_path, key = sweep / "last-test/test.jsonl", sweep / "last-test/result.json", "match"
    elif home.name == "E56" and split == "val":
        folder = home / "extra/data/val_selection" / name
        file, meta_path, key = folder / f"{step}.val.jsonl", folder / f"{step}.json", "inference"
    if not file or not file.is_file() or not meta_path.is_file():
        return INFER, file, checkpoint, "inference", "缺少训练末步逐对预测或原始推理记录", evidence
    meta = read(meta_path)
    if key:
        require(meta["step"] == step, f"{meta_path}: different training step")
        meta = meta[key]
    config = read(REPO / INFER)
    config.update(method=meta["method"], weights=meta["weights"])
    require(meta["config"] == config, f"{meta_path}: effective inference config differs")
    require(Path(meta["weights"]).name == checkpoint.name and meta["split"] == split and meta["finished"],
            f"{meta_path}: different checkpoint or incomplete inference")
    evidence["weights"] = [checked_weights(checkpoint, meta["weights_sha256"])]
    evidence["inference_metadata"] = str(meta_path)
    evidence["resolved_params"] = meta.get("resolved_params")
    return INFER, file, checkpoint, "reuse", "训练末步预测已核对原始权重 SHA-256 和推理配置", evidence


def subrun(args):
    subprocess.run(["nice", "-n", "10", *map(str, args)], cwd=REPO, env={**os.environ, **ENV}, check=True)


def publish(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, pending)
    pending.replace(target)


def assess(method, split, allow_inference):
    if method in TRAINED:
        config, path, checkpoint, mode, reason, evidence = trained(method, split)
        raw = None
    else:
        config, path, raw, mode, reason, evidence = baseline(method, split)
        checkpoint = None
    if mode == "reuse":
        try:
            prediction = check_rows(path, split, affine=True)
        except (OSError, ValueError) as exc:
            mode, reason = "inference", str(exc)
    if mode == "inference" and not allow_inference:
        print(f"DEFER {method}/{split}: {reason}", flush=True)
        return
    run_name = f"{method}__{split}"
    lock = RUN / ".claims" / run_name
    lock.parent.mkdir(exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        print(f"SKIP {method}/{split}: another worker has this split", flush=True)
        return
    try:
        launch.begin(RUN, run_name, [sys.executable, *sys.argv], entry=f"runs/{RUN.name}/code/reevaluate.py",
                     config_file=config, config=read(REPO / config), split=split, mode=mode, reason=reason)
        try:
            if mode != "reuse":
                stage = RUN / "raw" / method / split / uuid.uuid4().hex
                if mode == "inference":
                    options = ["--weights", checkpoint] if checkpoint else []
                    subrun([PY, "-m", "baselines.match", config, "--name", method, "--split", split, "--out", stage, *options])
                    raw = stage / method
                    meta_path = raw / f"{split}.meta.json"
                    meta = read(meta_path)
                    check_rows(raw / f"{split}.jsonl", split, affine=False)
                    require(meta["finished"] and meta["n_pairs"] == SPLITS[split], f"{meta_path}: inference unfinished")
                    if checkpoint:
                        require(meta["weights_sha256"] == evidence["weights"][0]["sha256"], f"{meta_path}: checkpoint hash mismatch")
                    elif not evidence.get("weights") and meta["weights"]:
                        evidence["weights"] = [checked_weights(Path(meta["weights"]), meta["weights_sha256"])]
                    evidence.update(inference_metadata=str(meta_path), resolved_params=meta.get("resolved_params"),
                                    upstream_commit=meta.get("repo_commit"))
                subrun([WB, "-m", "baselines.fit", raw, "--split", split, "--run", stage, "--name", method, "--ransac", "3"])
                path = stage / "preds" / method / f"{split}.jsonl"
                prediction = check_rows(path, split, affine=True)
            target = RUN / "preds" / method
            for name in (f"{split}.meta.json", f"{split}_matches.npz"):
                attachment = path.parent / name
                if attachment.is_file():
                    publish(attachment, target / name)
            publish(path, target / f"{split}.jsonl")
            failures = Counter(r["fail"] for r in prediction if r["A"] is None)
            launch.end(RUN, run_name, "ok", prediction_source=str(path),
                       prediction_sha256=sha256(path), prediction_count=len(prediction),
                       fit_failures=sum(failures.values()), failure_reasons=dict(failures), **evidence)
        except BaseException as exc:
            launch.end(RUN, run_name, "fail", error=str(exc))
            raise
        print(f"OK {method}/{split} {mode}, {sum(failures.values())} registration failures", flush=True)
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=(*BASELINES, *TRAINED))
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--infer", action="store_true", help="run missing or unverified inference on a reserved GPU")
    args = parser.parse_args()
    os.chdir(REPO)
    os.environ.update(ENV)
    launch.require_clean()
    errors = []
    for method in args.methods or (*BASELINES, *TRAINED):
        for split in (args.split,) if args.split else SPLITS:
            log = RUN / "launch" / f"{method}__{split}.json"
            if log.is_file() and read(log)[-1]["status"] == "ok":
                continue
            try:
                assess(method, split, args.infer)
            except (OSError, ValueError, subprocess.CalledProcessError, KeyError) as exc:
                errors.append(f"{method}/{split}: {exc}")
                print(f"ERROR {errors[-1]}", flush=True)
    if errors:
        sys.exit("\n".join(errors))


if __name__ == "__main__":
    main()
