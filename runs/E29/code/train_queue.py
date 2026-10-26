"""E29 只用（#215）：训练队列。照 scripts/finetune/run.py 的 main 领取 configs/ 里的方法（同一套 .claims/ 占位、
同样先检查已提交、同样写启动记录），但只训练、不评测：训练放在 V100 上，b4_* 的 Val 由 code/eval_queue.py 在别的卡上评，
rand_* 按票不评 Val。训练命令与产物同驱动（ckpt/<方法>/、tb/<方法>/scalars/、train.done）。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E29/code/train_queue.py > runs/E29/ckpt/train_<host>_g<GPU>.log 2>&1 < /dev/null &
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

ENTRY = "runs/E29/code/train_queue.py"


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
        D.log(f"START {R.name}/{m} gpu{gpu} (train only)")
        src = R / "configs" / f"{m}.toml"
        cfg = C.load(src)
        infer = json.loads((REPO / cfg["model"]["config"]).read_text(encoding="utf-8"))
        launch.begin(R, m, [sys.executable, *sys.argv], repo=REPO, entry=ENTRY,
                     config_file=src.relative_to(REPO).as_posix(), args=cfg, config=infer)
        job = D.Job(R, m, cfg, gpu)
        job.ck.mkdir(parents=True, exist_ok=True)
        try:
            if not (job.ck / "train.done").exists():
                job.sh([D.PY, "-m", "finetune.train", src, "--out", job.ck, "--tb", R / "tb" / m / "scalars"],
                       "train.log")
                (job.ck / "train.done").touch()
            launch.end(R, m, "ok")
            D.log(f"OK {m}")
        except subprocess.CalledProcessError as e:
            launch.end(R, m, "fail")
            D.log(f"FAIL {m}: {e}")


if __name__ == "__main__":
    main()
