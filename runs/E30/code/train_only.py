"""E30 只用（#216）：只训练、不评测的驱动，评测交给别的卡上的 code/eval_last.py，训完一个就交出去，训练卡不排评测。
领方法、占位（.claims/<方法>）、启动记录与 scripts/finetune/run.py 相同，只是 Job 训完（train.done）就去领下一个；
启动记录的 end / status 由 eval_last.py 评完后写（训练失败时这里写 fail）。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E30/code/train_only.py > runs/E30/ckpt/train_<host>_g<GPU>.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("driver", REPO / "scripts" / "finetune" / "run.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
from finetune import config as C  # noqa: E402
from workbench import launch  # noqa: E402

ENTRY = "runs/E30/code/train_only.py"


def train(job):
    job.ck.mkdir(parents=True, exist_ok=True)
    if not (job.ck / "train.done").exists():
        D.log(f"TRAIN {job.m}")
        job.sh([D.PY, "-m", "finetune.train", R / "configs" / f"{job.m}.toml", "--out", job.ck,
                "--tb", R / "tb" / job.m / "scalars"], "train.log")
        (job.ck / "train.done").touch()


def main():
    gpu = os.environ["GPU"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    launch.require_clean(REPO)
    for m in D.methods(R):
        C.load(R / "configs" / f"{m}.toml")
    claims = R / ".claims"
    claims.mkdir(exist_ok=True)
    while True:
        picked = None
        for m in D.methods(R):
            try:
                (claims / m).mkdir()
            except FileExistsError:
                continue
            picked = m
            break
        if picked is None:
            D.log("queue empty")
            return
        m = picked
        (claims / m / "host").write_text(f"{socket.gethostname()} gpu{gpu} {time.strftime('%F %T %z')}\n")
        D.log(f"START {R.name}/{m} gpu{gpu}")
        src = R / "configs" / f"{m}.toml"
        cfg = C.load(src)
        infer = json.loads((REPO / cfg["model"]["config"]).read_text(encoding="utf-8"))
        launch.begin(R, m, [sys.executable, *sys.argv], repo=REPO, entry=ENTRY,
                     config_file=src.relative_to(REPO).as_posix(), args=cfg, config=infer)
        try:
            train(D.Job(R, m, cfg, gpu))
            D.log(f"TRAINED {m}")
        except subprocess.CalledProcessError as e:
            launch.end(R, m, "fail")
            D.log(f"FAIL {m}: {e}")


if __name__ == "__main__":
    main()
