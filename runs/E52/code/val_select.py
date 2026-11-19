"""E52: score the fixed checkpoint grid on Val, then Test only its winner.

GPU=5 /opt/envs/wb/bin/python runs/E52/code/val_select.py no_screen --scratch /tmp
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from run import D, REPO, RUN
from workbench.dataset import Dataset
from workbench.evaluate import evaluate_preds
from workbench.records import Run

ENTRY = "runs/E52/code/val_select.py"
GRID = (*range(250, 7907, 250), 7907)
METHODS = ("no_screen", "frozen_dino")


def publish_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_predictions(stage, method, split, ds):
    prediction = stage / "preds" / method / f"{split}.jsonl"
    expected = set(ds.labelled(split))
    size = {"val": 652, "test": 953}[split]
    with prediction.open(encoding="utf-8") as stream:
        pairs = [json.loads(line)["pair"] for line in stream]
    if len(expected) != size or len(pairs) != size or set(pairs) != expected:
        raise RuntimeError(f"{split}: expected the exact {size} labelled pair IDs, got {len(set(pairs))}")
    r = Run(stage.name, stage, {}, [method])
    result = evaluate_preds(ds, split, r.preds(method, split))
    if result["coverage"]["missing"]:
        raise RuntimeError(f"{split}: missing labelled predictions")
    return prediction, result


def evaluate(job, ckpt, split, name, stage, ds):
    raw = stage / "match"
    job.sh([D.PY, "-m", "baselines.match", job.infer, "--split", split, "--weights", ckpt,
            "--name", name, "--out", raw], "val_select.log")
    job.sh([D.WB, "-m", "baselines.fit", raw / name, "--split", split, "--run", stage],
           "val_select.log", gpu=False)
    prediction, result = validate_predictions(stage, name, split, ds)
    meta = json.loads((raw / name / f"{split}.meta.json").read_text(encoding="utf-8"))
    return prediction, dict(score=result, inference=meta)


def select_and_test(job, scratch, ds, report, official):
    candidates = []
    for step in GRID:
        ckpt = job.ck / f"ckpt_{step}.pt"
        if not ckpt.is_file() or not ckpt.stat().st_size:
            raise RuntimeError(f"Missing candidate: {ckpt}")
        result_path = report / f"{step}.json"
        if result_path.exists():
            rec = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            with tempfile.TemporaryDirectory(prefix=f"e52-val-{job.m}-{step}-", dir=scratch) as tmp:
                prediction, rec = evaluate(job, ckpt, "val", job.m, Path(tmp), ds)
                shutil.copyfile(prediction, report / f"{step}.val.jsonl")
            rec.update(step=step, checkpoint=ckpt.relative_to(RUN).as_posix())
            publish_json(result_path, rec)
        auc = rec["score"]["summary"]["auc@5"]
        if rec["step"] != step or rec["score"]["summary"]["n"] != 652 or not math.isfinite(auc):
            raise RuntimeError(f"Invalid Val score: {result_path}")
        candidates.append({"step": step, "auc@5": auc})
        print(f"{job.m} Val step={step} auc@5={auc:.9f}", flush=True)

    winner = max(candidates, key=lambda c: (c["auc@5"], -c["step"]))
    step = winner["step"]
    ckpt = job.ck / f"ckpt_{step}.pt"
    if step == GRID[-1]:
        history = json.loads((RUN / "launch" / f"{job.m}_test.json").read_text(encoding="utf-8"))
        if history[-1]["status"] != "ok" or history[-1]["step"] != step:
            raise RuntimeError("Historical terminal Test does not match the selected checkpoint")
        _, test_score = validate_predictions(RUN, job.m, "test", ds)
        test_method = job.m
        test = dict(score=test_score, source="historical_terminal")
    else:
        with tempfile.TemporaryDirectory(prefix=f"e52-test-{job.m}-{step}-", dir=scratch) as tmp:
            pred, test = evaluate(job, ckpt, "test", official, Path(tmp), ds)
            target = RUN / "preds" / official
            pending = target.with_name(f".{official}.publish")
            shutil.copytree(pred.parent, pending)
            pending.rename(target)
        test_method = official
    outcome = dict(method=job.m, criterion="val_auc@5", tie_break="earliest_step",
                   candidates=candidates, winner=winner, checkpoint=ckpt.relative_to(RUN).as_posix(),
                   test_method=test_method, test=test)
    publish_json(report / "selection.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    gpu = os.environ["GPU"]
    if not gpu.isdigit():
        parser.error("GPU must be one physical card index")
    os.chdir(REPO)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    source = RUN / "configs" / f"{args.method}.toml"
    cfg = D.C.load(source)
    job = D.Job(RUN, args.method, cfg, gpu)
    official = f"{job.m}__val_peak"
    report = RUN / "extra" / "data" / "val_selection" / job.m
    report.mkdir(parents=True, exist_ok=True)
    if (report / "selection.json").exists():
        return
    claim = RUN / ".claims" / official
    claim.mkdir()
    infer = json.loads((REPO / job.infer).read_text(encoding="utf-8"))
    D.launch.begin(RUN, official, [sys.executable, *sys.argv], repo=REPO, entry=ENTRY,
                            config_file=source.relative_to(REPO).as_posix(), args=cfg, config=infer,
                            phase="val_peak", candidates=GRID)
    try:
        ds = Dataset(D.ENV["MOON_DATA"])
        outcome = select_and_test(job, args.scratch, ds, report, official)
    except BaseException:
        D.launch.end(RUN, official, "fail")
        raise
    D.launch.end(RUN, official, "ok", checkpoint=outcome["checkpoint"], step=outcome["winner"]["step"],
                 test_method=outcome["test_method"])
    print(json.dumps(outcome["winner"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
