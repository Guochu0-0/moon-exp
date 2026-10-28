"""E34 只用（#228）：训练与评测分开放卡（同 E31 的 train_queue.py）。本脚本只训练：照 scripts/finetune/run.py 的驱动领取方法
（.claims/<方法>）、写启动记录、跑 finetune.train、训完（含提前停，见 [run] halt_*）写 train.done，然后接着领下一个；不评测。
评测由 eval_queue.py 在另一张卡上接手，它评完再给启动记录写上 end 与 status。训练失败时这里直接记 fail。
METHODS 给出时只领这几个方法（逗号分隔），用来把指定的方法放到指定的卡上。

    GPU=<空卡> [METHODS=acc4] nohup /opt/envs/loftr/bin/python runs/E34/code/train_queue.py > runs/E34/ckpt/train_<host>_g<GPU>.log 2>&1 < /dev/null &
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


def main():
    gpu = os.environ["GPU"]
    only = [x for x in os.environ.get("METHODS", "").split(",") if x]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    launch.require_clean(REPO)
    for m in D.methods(R):
        C.load(R / "configs" / f"{m}.toml")
    claims = R / ".claims"
    claims.mkdir(exist_ok=True)
    while True:
        picked = None
        for m in D.methods(R):
            if only and m not in only:
                continue
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
        launch.begin(R, m, [sys.executable, *sys.argv], repo=REPO, entry="runs/E34/code/train_queue.py",
                     config_file=src.relative_to(REPO).as_posix(), args=cfg, config=infer)
        job = D.Job(R, m, cfg, gpu)
        job.ck.mkdir(parents=True, exist_ok=True)
        try:
            D.log(f"TRAIN {m} [{', '.join(C.parts_of(cfg))}]")
            job.sh([D.PY, "-m", "finetune.train", src, "--out", job.ck, "--tb", R / "tb" / m / "scalars"], "train.log")
            (job.ck / "train.done").touch()
            D.log(f"TRAINED {m}（评测交给 eval_queue.py）")
        except subprocess.CalledProcessError as e:
            launch.end(R, m, "fail")
            D.log(f"FAIL {m}: {e}")


if __name__ == "__main__":
    main()
