"""Score one saved E55 weight, or evaluate Test after Val selection."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from scripts.finetune.run import C, ENV, Job, WB  # noqa: E402
from workbench import launch  # noqa: E402
from workbench.dataset import Dataset  # noqa: E402
from workbench.evaluate import evaluate_preds  # noqa: E402
from workbench.records import Run  # noqa: E402

STEPS = (*range(250, 7907, 250), 7907)
NO_BCE_SERIES = tuple(f"k{k}_no_bce" for k in range(6))
METHODS = ("k0", "k1", "k2", "k3", "k4", "k5", "k4_seed1", *NO_BCE_SERIES, "k3_no_screen", "k3_no_screen_no_bce")
E41_BEST_HASH = "052af2db0c18dea9f1f562749ccb1673a4aa63f8487925df06d56062e8b57019"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def score(run, method, split, expected):
    path = run / "preds" / method / f"{split}.jsonl"
    ids = [json.loads(line)["pair"] for line in path.read_text(encoding="utf-8").splitlines()]
    ds = Dataset(ENV["MOON_DATA"])
    if len(ids) != expected or set(ids) != set(ds.labelled(split)):
        raise ValueError(f"{path}: expected exactly the {expected} labelled {split} pairs")
    return evaluate_preds(ds, split, Run(run.name, run, {}, [method]).preds(method, split))


def evaluate(method, step, split, scratch):
    config = ROOT / "configs" / f"{method}.toml"
    cfg = C.load(config)
    job = Job(ROOT, method, cfg, os.environ["GPU"])
    checkpoint = job.ck / f"ckpt_{step}.pt"
    if not zipfile.is_zipfile(checkpoint):
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    job.sw, job.S = scratch, scratch / "S"
    job.sh([WB, "-m", "workbench", "--runs", scratch, "new", "S", "--title", f"E55 {method} {step}"], "sweep.log", gpu=False)
    job.evaluate(split, str(step))
    meta = read_json(scratch / "match" / f"step{step}" / f"{split}.meta.json")
    digest = hash_file(checkpoint)
    if meta["weights_sha256"] != digest:
        raise ValueError(f"Mismatched evaluated weights for {method} step {step}")
    result = score(job.S, f"step{step}", split, 652 if split == "val" else 953)
    return result["summary"], meta, job.S / "preds" / f"step{step}"


def hash_file(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(method):
    scores = ROOT / "sweep" / method / "scores"
    if method == "k5":
        ck = ROOT / "ckpt" / method / "ckpt_4250.pt"
        if hash_file(ck) == E41_BEST_HASH:
            reference = REPO.parent.parent / "moon-exp" / "runs/E41/sweep/main__screen0.8/S/metrics.json"
            ref = read_json(reference)["methods"]
            selected = max(STEPS, key=lambda step: (ref[f"step{step}"]["val"]["summary"]["auc@5"], -step))
            if selected != 4250:
                raise ValueError("E41 validation grid changed")
            return selected, ref["step4250"]["val"]["summary"], "E41 exact checkpoint SHA-256"
    candidates = {step: read_json(scores / f"{step}.json")["summary"] for step in STEPS}
    step = max(STEPS, key=lambda i: (candidates[i]["auc@5"], -i))
    return step, candidates[step], "E55 validation grid"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("val", "test"))
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--step", type=int, choices=STEPS)
    args = parser.parse_args()
    method = args.method
    sweep = ROOT / "sweep" / method
    if args.split == "val":
        if args.step is None:
            parser.error("val requires --step")
        with tempfile.TemporaryDirectory(prefix="moon-e55-") as tmp:
            summary, meta, preds = evaluate(method, args.step, "val", Path(tmp))
            val = sweep / "val" / str(args.step)
            val.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preds / "val.jsonl", val / "val.jsonl")
            write_json(sweep / "scores" / f"{args.step}.json", {"step": args.step, "summary": summary, "match": meta})
        return
    if args.step is not None:
        parser.error("test reads the selected step")
    step, val, source = select(method)
    selected = {"step": step, "val": val, "source": source, "tie_break": "earliest"}
    write_json(sweep / "selection.json", selected)
    launch.begin(ROOT, f"{method}_eval", sys.argv, repo=REPO, selection=selected, phase="test")
    try:
        with tempfile.TemporaryDirectory(prefix="moon-e55-") as tmp:
            summary, meta, preds = evaluate(method, step, "test", Path(tmp))
            dst = ROOT / "preds" / method
            dst.mkdir(parents=True, exist_ok=True)
            for file in preds.iterdir():
                shutil.copy2(file, dst / file.name)
            val_path = sweep / "val" / str(step) / "val.jsonl"
            if source.startswith("E41"):
                val_path = REPO.parent.parent / "moon-exp/runs/E41/sweep/main__screen0.8/S/preds/step4250/val.jsonl"
            shutil.copy2(val_path, dst / "val.jsonl")
            write_json(sweep / "peak.json", {**selected, "test": summary, "match": meta})
    except Exception:
        launch.end(ROOT, f"{method}_eval", "fail")
        raise
    launch.end(ROOT, f"{method}_eval", "ok")


if __name__ == "__main__":
    main()
