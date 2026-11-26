"""Audit and complete the fixed E55 K5 checkpoint evaluation grid for #301."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from workbench import launch
from workbench.dataset import Dataset
from workbench.protocol import auc, pair_error

YGC = Path("/remote-home/xufang/YGC")
E55 = YGC / "moon-exp/runs/E55"
METHOD = "k5_no_bce"
TRAIN = E55 / "ckpt" / METHOD
SWEEP = E55 / "sweep" / METHOD
INIT = YGC / "weights/anymatch/RoMa_AnyMatch.pth"
INIT_SHA = "386bcc2173737b5b9309a54ee7f0f7268045d73438be9946fa93bb5afe6b3f56"
CONFIG_PATH = "configs/baselines/anymatch_roma__minmax.json"
MATCHER = "259ad343ffa9ca0703965f56880b3b5d2ecbc9e1"
STEPS = (0, *range(250, 7907, 250), 7907)
COUNTS = {"val": 652, "test": 953}
PY = "/opt/envs/loftr/bin/python"
WB = "/opt/envs/wb/bin/python"
DATA_PATH = YGC / "dataset/Moon"
DATA = Dataset(DATA_PATH)
CACHE = RUN / "ckpt/results"
FITTING = ["baselines/fit.py", "moonlib/ransac.py", "workbench/protocol.py"]
INFERENCE = ["baselines/match.py", "baselines/adapters/romatch.py", "baselines/adapters/base.py"]


def demand(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
                       encoding="utf-8")
    pending.replace(path)


@lru_cache
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1 << 22), b""):
            digest.update(part)
    return digest.hexdigest()


@lru_cache
def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True, timeout=120).stdout.strip()


@lru_cache
def labelled(split):
    ids = set(DATA.labelled(split))
    demand(len(ids) == COUNTS[split], f"{split}: annotation count {len(ids)}")
    return ids


@lru_cache
def config():
    return read(REPO / CONFIG_PATH)


@lru_cache
def reference_params():
    return read(SWEEP / "scores/7907.json")["match"]["resolved_params"]


def check_meta(meta, step, split, digest):
    expected_name = f"step{step}"
    expected_file = TRAIN / f"ckpt_{step}.pt" if step else INIT
    expected = dict(config(), method=expected_name, weights=meta["weights"])
    demand(meta["config"] == expected and meta["config_path"] == CONFIG_PATH,
           "effective normalization, model or seed differs")
    demand(meta["method"] == expected_name and Path(meta["weights"]).name == expected_file.name
           and meta["weights_sha256"] == digest, "weight identity differs")
    demand(meta["split"] == split and meta["finished"] and not meta["perturb"]
           and meta["n_pairs"] == COUNTS[split] and meta["n_errors_this_session"] == 0,
           "inference did not complete with the requested split and coverage")
    demand(meta["repo_commit"] == MATCHER and meta["resolved_params"] == reference_params(),
           "inference implementation or resolved parameters differ")
    demand(not git("diff", meta["commit"], "HEAD", "--", *INFERENCE),
           "historical inference implementation differs from the current code")


def summarize(path, split):
    data = rows(path)
    ids = labelled(split)
    demand(len(data) == len(ids) and {row["pair"] for row in data} == ids,
           f"{path}: prediction IDs differ from the annotation manifest")
    errors, inlier_ratios = [], []
    failures = Counter()
    zero_matches = 0
    for row in data:
        pair, affine = row["pair"], row["A"]
        if affine is None:
            reason = row.get("fail")
            demand(reason in {"few_matches", "few_inliers"}, f"{pair}: execution error {reason}")
            failures[reason] += 1
        else:
            demand(np.shape(affine) == (2, 3) and np.isfinite(affine).all(), f"{pair}: bad affine")
        errors.append(pair_error(affine, *DATA.checkpoints(split, pair)))
        if split == "val":
            matches = row["n_matches"]
            inliers = row["n_inliers"] if affine is not None else row.get("n_inliers", 0)
            demand(isinstance(matches, int) and isinstance(inliers, int) and 0 <= inliers <= matches
                   and (affine is None or inliers >= 3), f"{pair}: invalid match counts")
            zero_matches += matches == 0
            inlier_ratios.append(inliers / matches if matches else 0)
    summary = {"covered": len(data), "fit_failures": sum(failures.values()),
               "failure_reasons": dict(failures), "auc5": auc(np.array(errors), 5),
               "prediction_sha256": sha256(path)}
    if split == "val":
        summary.update(inlier_mean=float(np.mean(inlier_ratios)),
                       inlier_median=float(np.median(inlier_ratios)), zero_matches=zero_matches)
    return summary


def history_paths(step, split):
    if step == 0:
        raise ValueError("E58 initialization predictions predate resolved inference metadata; Val fit is marked dirty")
    if split == "val":
        return SWEEP / f"val/{step}/val.jsonl", SWEEP / f"scores/{step}.json", "summary"
    if step == 7907:
        return SWEEP / "last-test/test.jsonl", SWEEP / "last-test/result.json", "test"
    source = SWEEP / f"top5-test/step{step}"
    path = E55 / f"preds/{METHOD}/test.jsonl" if step == 7000 else source / "test.jsonl"
    return path, source / "result.json", "test"


def reuse(step, split, digest):
    path, record, summary_key = history_paths(step, split)
    historical = read(record)
    meta = historical["match"]
    demand(historical["step"] == step, "historical result represents a different step")
    check_meta(meta, step, split, digest)
    fit_meta = path.with_suffix(".meta.json")
    if fit_meta.exists():
        fit = read(fit_meta)
        demand(fit["commit"] == meta["commit"] and not fit["dirty"], "fit metadata differs")
    # The original E55 driver invokes baselines.fit with its 3 px default.
    demand(not git("diff", meta["commit"], "HEAD", "--", *FITTING), "historical fit or scoring code differs")
    demand("baselines.fit" in git("show", f'{meta["commit"]}:scripts/finetune/run.py'),
           "historical fit command unavailable")
    result = summarize(path, split)
    demand(math.isclose(result["auc5"], historical[summary_key]["auc@5"], abs_tol=1e-12, rel_tol=0),
           "recomputed AUC differs from historical summary")
    return dict(result, mode="reuse", prediction=str(path), original_result=str(record), match=meta)


def slot(step, split):
    return CACHE / f"{split}_{step}.json"


def audit():
    training = read(E55 / f"launch/{METHOD}.json")[-1]
    args = training["args"]
    demand(training["status"] == "ok" and not training["dirty"] and args["run"]["seed"] == 0
           and args["model"]["config"] == CONFIG_PATH and not args["model"]["init"]
           and training["config"] == config(), "historical training run does not match #301")
    demand(args["rloo"]["cert_bce"] == args["rloo"]["fine_cert_bce"] == 0,
           "historical training run did not disable both BCE weights")
    weights, missing, pending = {}, {}, []
    for step in STEPS:
        path = TRAIN / f"ckpt_{step}.pt" if step else INIT
        expected = (read(SWEEP / f"scores/{step}.json")["match"]["weights_sha256"]
                    if step else INIT_SHA)
        digest = sha256(path) if path.is_file() else None
        if digest is None or (step and not zipfile.is_zipfile(path)):
            missing[step] = "权重缺失或存档不完整"
        elif digest != expected:
            missing[step] = "权重 SHA-256 与历史记录不符"
        weights[step] = {"path": str(path), "historical_sha256": expected, "sha256": digest}
        if step in missing:
            continue
        for split in COUNTS:
            if slot(step, split).is_file():
                continue
            try:
                write(slot(step, split), reuse(step, split, expected))
                print(f"REUSE {step} {split}", flush=True)
            except (ValueError, FileNotFoundError, KeyError, subprocess.CalledProcessError) as exc:
                reason = "无历史逐对预测" if isinstance(exc, FileNotFoundError) else str(exc)
                pending.append({"step": step, "split": split, "reason": reason})
                print(f"PENDING {step} {split}: {reason}", flush=True)
    write(CACHE / "plan.json", {"weights": weights, "missing": missing, "pending": pending,
                               "training_commit": training["commit"]})
    return {"pending": len(pending), "missing": missing}


def infer(step, split, weight):
    env = {**os.environ, "MOON_DATA": str(DATA_PATH), "MOON_WEIGHTS": str(YGC / "weights"),
           "TORCH_HOME": str(YGC / "weights/torch_home"), "MOON_GIT_TIMEOUT": "240",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    name = f"step{step}"
    with tempfile.TemporaryDirectory(prefix="moon-e61-") as scratch:
        raw = Path(scratch) / name
        match = [PY, "-m", "baselines.match", CONFIG_PATH, "--split", split, "--weights", weight["path"],
                 "--name", name, "--out", scratch]
        subprocess.run(match, cwd=REPO, env=env, check=True, timeout=3_600)
        meta = read(raw / f"{split}.meta.json")
        check_meta(meta, step, split, weight["sha256"])
        raw_rows = rows(raw / f"{split}.jsonl")
        demand(len(raw_rows) == COUNTS[split] and {r["pair"] for r in raw_rows} == labelled(split)
               and all("error" not in r for r in raw_rows), "raw inference coverage or execution error")
        with zipfile.ZipFile(raw / f"{split}.npz") as archive:
            demand({r["pair"].replace("/", "__") + ".npy" for r in raw_rows if r["n"]}
                   <= set(archive.namelist()), "missing raw point arrays")
        fit = [WB, "-m", "baselines.fit", str(raw), "--split", split, "--run", scratch, "--ransac", "3"]
        subprocess.run(fit, cwd=REPO, env=env, check=True, timeout=600)
        path = Path(scratch) / "preds" / name / f"{split}.jsonl"
        result = summarize(path, split)
        return dict(result, mode="inference", match=meta, inference_command=match,
                    fit_command=fit, fit_metadata=read(path.with_suffix(".meta.json")))


def recorded(name, action):
    launch.begin(RUN, name, [sys.executable, *sys.argv], config_file=CONFIG_PATH, config=config())
    try:
        evidence = action()
    except BaseException as exc:
        launch.end(RUN, name, "fail", error=str(exc))
        raise
    launch.end(RUN, name, "ok", **evidence)


def worker(minutes):
    stop_starting = time.monotonic() + 60 * (minutes - 25)
    plan = read(CACHE / "plan.json")
    jobs = RUN / ".claims"
    jobs.mkdir(exist_ok=True)
    errors = []
    for entry in plan["pending"]:
        if time.monotonic() >= stop_starting:
            break
        step, split = entry["step"], entry["split"]
        name = f"{split}_{step}"
        target = slot(step, split)
        if target.exists():
            continue
        lock = jobs / name
        try:
            lock.mkdir()
        except FileExistsError:
            continue
        try:
            if not target.exists():
                recorded(name, lambda: run_one(entry, plan["weights"][str(step)]))
        except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
            errors.append(f"{name}: {exc}")
            print(f"FAILED {errors[-1]}", flush=True)
        finally:
            lock.rmdir()
    if errors:
        raise SystemExit("\n".join(errors))


def run_one(entry, weight):
    step, split = entry["step"], entry["split"]
    print(f"START {step} {split}", flush=True)
    result = infer(step, split, weight)
    result["reason"] = entry["reason"]
    write(slot(step, split), result)
    print(f"DONE {step} {split}: {result['auc5']:.6f}", flush=True)
    return {"step": step, "split": split, "weights": weight, "count": result["covered"],
            "prediction_sha256": result["prediction_sha256"]}


def main():
    os.chdir(REPO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "worker"))
    parser.add_argument("--minutes", type=int, default=220)
    args = parser.parse_args()
    if args.stage == "audit":
        recorded("audit", audit)
    else:
        worker(args.minutes)


if __name__ == "__main__":
    main()
