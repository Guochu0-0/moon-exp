"""Run one E57 train or fixed Test job on externally allocated GPUs.

    GPU=0,1,2,3 python runs/E57/code/run.py train no_ema
    GPU=0 python runs/E57/code/run.py test no_ema
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]
ENTRY = "runs/E57/code/run.py"
spec = importlib.util.spec_from_file_location("finetune_driver", REPO / "scripts/finetune/run.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def terminal_checkpoint(job):
    halt_file = job.ck / "halt.json"
    halt = json.loads(halt_file.read_text(encoding="utf-8")) if halt_file.exists() else None
    step = halt["step"] if halt else job.cfg["optim"]["steps"]
    ckpt = job.ck / f"ckpt_{step}.pt"
    if not ckpt.is_file() or not ckpt.stat().st_size:
        raise RuntimeError(f"Missing terminal checkpoint: {ckpt}")
    return ckpt, step, halt


def train(job, source):
    count = job.cfg["run"]["ddp"]
    cmd = [D.PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={count}",
           "-m", "finetune.train", source, "--out", job.ck, "--tb", RUN / "tb" / job.m / "scalars"]
    job.sh(cmd, "train.log")


def test(job, ckpt, scratch=None):
    stage = scratch if scratch is not None else RUN
    raw = stage / "match" if scratch is not None else job.sw / "match"
    job.sh([D.PY, "-m", "baselines.match", job.infer, "--split", "test", "--weights", ckpt,
            "--name", job.m, "--out", raw], "test.log")
    job.sh([D.WB, "-m", "baselines.fit", raw / job.m, "--split", "test", "--run", stage], "test.log", gpu=False)
    predictions = stage / "preds" / job.m / "test.jsonl"
    with predictions.open(encoding="utf-8") as stream:
        pairs = {json.loads(line)["pair"] for line in stream}
    reference = REPO / "runs/E41/preds/main__screen0.8/test.jsonl"
    with reference.open(encoding="utf-8") as stream:
        expected = {json.loads(line)["pair"] for line in stream}
    if len(expected) != 953 or pairs != expected:
        raise RuntimeError(f"Expected the 953 labelled Test pairs from E41, got {len(pairs)}")
    if scratch is not None:
        target = RUN / "preds" / job.m
        pending = target.with_name(f".{job.m}.publish")
        shutil.copytree(predictions.parent, pending)
        pending.rename(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("train", "test"))
    ap.add_argument("method", choices=D.methods(RUN))
    ap.add_argument("--retry-in", type=Path, help="retry a failed Test and stage its files in this new local directory")
    args = ap.parse_args()
    if args.retry_in and args.phase != "test":
        ap.error("--retry-in requires test")
    source = RUN / "configs" / f"{args.method}.toml"
    configs = {m: D.C.load(RUN / "configs" / f"{m}.toml") for m in D.methods(RUN)}
    cfg = configs[args.method]
    expected_gpu_count = cfg["run"]["ddp"] if args.phase == "train" else 1
    try:
        gpus = [int(index) for index in os.environ.get("GPU", "").split(",")]
        if len(gpus) != expected_gpu_count or len(set(gpus)) != expected_gpu_count or min(gpus) < 0:
            raise ValueError("wrong number of distinct nonnegative indices")
    except ValueError:
        ap.error(f"{args.phase} requires {expected_gpu_count} distinct GPU indices in GPU")
    gpu = ",".join(map(str, gpus))
    os.chdir(REPO)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    D.launch.require_clean(REPO)
    job = D.Job(RUN, args.method, cfg, gpu)
    record = args.method if args.phase == "train" else f"{args.method}_test"
    if args.phase == "test" and not (job.ck / "train.done").is_file():
        ap.error(f"Training must finish first: {job.ck / 'train.done'}")
    claims = RUN / ".claims"
    claims.mkdir(exist_ok=True)
    if args.retry_in:
        history = json.loads((RUN / "launch" / f"{record}.json").read_text(encoding="utf-8"))
        if history[-1]["status"] != "fail" or (RUN / "preds" / args.method).exists():
            ap.error("--retry-in requires an earlier failed Test and no published predictions")
        args.retry_in.mkdir()
        (claims / f"{record}_retry").mkdir()
    else:
        (claims / record).mkdir()
    job.ck.mkdir(parents=True, exist_ok=True)
    infer = json.loads((REPO / job.infer).read_text(encoding="utf-8"))
    D.launch.begin(RUN, record, [sys.executable, *sys.argv], repo=REPO, entry=ENTRY,
                   config_file=source.relative_to(REPO).as_posix(), args=cfg, config=infer, phase=args.phase)
    try:
        if args.phase == "train":
            train(job, source)
        ckpt, step, halt = terminal_checkpoint(job)
        if args.phase == "test":
            test(job, ckpt, args.retry_in)
    except BaseException:
        D.launch.end(RUN, record, "fail")
        raise
    D.launch.end(RUN, record, "ok", checkpoint=ckpt.relative_to(RUN).as_posix(), step=step, halt=halt)
    if args.phase == "train":
        (job.ck / "train.done").touch()
    D.log(f"DONE {args.phase} {args.method} step={step} halt={halt}")


if __name__ == "__main__":
    main()
