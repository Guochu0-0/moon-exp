"""E30 只用（#216）：评测进程。等 code/train_only.py 训完的方法（ckpt/<方法>/train.done），领走（mkdir .evalclaims/<方法>），
照 scripts/finetune/run.py 的 Job 评第 1200 步的 Val（配置 [run] sweep = "last"：只评最后一个 ckpt 的 Val、不评 Test），
写 sweep/<方法>/peak.json、preds/<方法>/val.jsonl，跑 workbench eval E30，再给这个方法的启动记录写 end 与 status。
评失败的放回去（删掉占位）、记 FAIL，下一轮再试；同一方法连续失败 2 次就把启动记录记为 fail、不再领。
全部方法都已评完（或记了 fail）就退出。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E30/code/eval_last.py > runs/E30/ckpt/eval_<host>_g<GPU>.log 2>&1 < /dev/null &
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


def finished(m):
    items = json.loads((R / "launch" / f"{m}.json").read_text(encoding="utf-8")) if (R / "launch" / f"{m}.json").exists() else []
    return bool(items) and items[-1].get("status") is not None


def main():
    gpu = os.environ["GPU"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    claims = R / ".evalclaims"
    claims.mkdir(exist_ok=True)
    fails = {}
    while True:
        busy = False
        for m in D.methods(R):
            if finished(m):
                continue
            busy = True
            if not (R / "ckpt" / m / "train.done").exists():
                continue
            try:
                (claims / m).mkdir()
            except FileExistsError:
                continue
            (claims / m / "host").write_text(f"{socket.gethostname()} gpu{gpu} {time.strftime('%F %T %z')}\n")
            D.log(f"EVAL {m} gpu{gpu}")
            cfg = C.load(R / "configs" / f"{m}.toml")
            try:
                D.Job(R, m, cfg, gpu)()      # train.done 已在：跳过训练，只评最后一个 ckpt 的 Val
                launch.end(R, m, "ok")
                D.log(f"OK {m}")
            except subprocess.CalledProcessError as e:
                fails[m] = fails.get(m, 0) + 1
                D.log(f"FAIL {m} ({fails[m]}): {e}")
                if fails[m] >= 2:
                    launch.end(R, m, "fail")
                else:
                    for f in (claims / m).iterdir():
                        f.unlink()
                    (claims / m).rmdir()
        if not busy:
            D.log("all evaluated")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
