"""Run the next unfinished RIPE++ phases within one checked GPU reservation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ENTRY = "runs.E60.code.run"
STAGES = ("extractor_final", "matcher_final")


def completed(method, phase, *, split=None, observation=8, extractor_sha=None):
    name = f"{method}_sweep" if phase == "infer" and observation != 8 else method
    path = RUN / "launch" / f"{name}.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    for row in reversed(records):
        if row["phase"] != phase or row["split"] != split or row["observation_pass"] != observation:
            continue
        if extractor_sha and row["identity"]["checkpoints"].get("extractor_final") != extractor_sha:
            continue
        if row["end"] and row["status"] == "ok" and (phase != "train" or row["complete"]):
            return row
        return None
    return None


def gpu_available():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    command = ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"]
    return not subprocess.check_output(command, text=True, timeout=10).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=13_200)
    args = parser.parse_args()
    if not 500 <= args.seconds <= 13_200:
        parser.error("seconds must be within one GPU reservation (500–13,200)")
    end = time.monotonic() + args.seconds

    def run(phase, method, *, split=None, observation=8, extractor_sha=None):
        existing = completed(method, phase, split=split, observation=observation, extractor_sha=extractor_sha)
        if existing:
            return existing
        remaining = int(end - time.monotonic())
        if remaining < 420 or not gpu_available():
            print(f"Pause before {phase} {method}: segment budget or GPU occupancy", flush=True)
            return None
        command = [sys.executable, "-u", "-m", ENTRY, phase, method, "--seconds", str(remaining - 150)]
        if split:
            command.extend(["--split", split, "--pass", str(observation)])
        result = subprocess.run(command, timeout=remaining + 40)
        result.check_returncode()
        return completed(method, phase, split=split, observation=observation, extractor_sha=extractor_sha)

    for method in ("init", *STAGES):
        if not run("preflight", method):
            return
    extractor = run("train", STAGES[0])
    if not extractor:
        return
    checksum = extractor["final_sha256"]
    if not run("preflight", STAGES[1], extractor_sha=checksum):
        return
    if not run("train", STAGES[1], extractor_sha=checksum):
        return

    for method in ("init", *STAGES):
        for split in ("val", "test"):
            if not run("infer", method, split=split):
                return
    for method in STAGES:
        for observation in (1, 2, 4, 6):
            if not run("infer", method, split="val", observation=observation):
                return
    print("RIPE++ training and fixed checkpoint prediction complete", flush=True)


if __name__ == "__main__":
    main()
