"""E31 只用（#217）：在另一张卡上评测 train_queue.py 训完的方法（参照 E28 的 eval_ahead.py）。见到 ckpt/<方法>/train.done
且还没有 preds/<方法>/val.jsonl 的方法，靠 mkdir runs/E31/.evalclaims/<方法> 领取，照驱动调用 Job（train.done 已在，
跳过训练；配置 [run] sweep = "last"：只评最后一个 ckpt 的 Val，写 sweep/<方法>/peak.json、preds/<方法>/val.jsonl、
workbench eval E31），再给训练时写下的启动记录记上 end 与 status。全部方法都有结果（或训练失败）就退出。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E31/code/eval_queue.py > runs/E31/ckpt/eval_<host>_g<GPU>.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import importlib.util
import json
import os
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


def last_status(m):
    f = R / "launch" / f"{m}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))[-1].get("status")


def main():
    gpu = os.environ["GPU"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    claims = R / ".evalclaims"
    claims.mkdir(exist_ok=True)
    while True:
        busy = False
        for m in D.methods(R):
            if (R / "preds" / m / "val.jsonl").exists() or last_status(m) == "fail":
                continue
            cfg = C.load(R / "configs" / f"{m}.toml")
            job = D.Job(R, m, cfg, gpu)
            if not (job.ck / "train.done").exists():
                busy = True
                continue
            try:
                (claims / m).mkdir()
            except FileExistsError:
                continue
            D.log(f"EVAL {m} gpu{gpu}")
            try:
                job()
                launch.end(R, m, "ok")
                D.log(f"OK {m}")
            except subprocess.CalledProcessError as e:
                launch.end(R, m, "fail")
                D.log(f"FAIL {m}: {e}")
        if not busy:
            D.log("all done")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
