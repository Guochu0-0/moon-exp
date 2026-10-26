"""E29 只用（#215）：评测队列。等 code/train_queue.py 训完 b4_*（train.done 出现），在别的空卡上照驱动的
[run] sweep = "last" 评最后一个存档（第 1200 次更新）的 Val、不评 Test：直接调用 scripts/finetune/run.py 的 Job，
train.done 已在所以跳过训练，产物同驱动（sweep/<方法>/、peak.json、preds/<方法>/val.jsonl、workbench eval E29）。
每个方法另记一条启动记录（入口为本脚本）。多个进程靠 mkdir runs/E29/.evalclaims/<方法> 分工；rand_* 按票不评。
b4_* 都评完或已被领走就退出。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E29/code/eval_queue.py > runs/E29/ckpt/eval_<host>_g<GPU>.log 2>&1 < /dev/null &
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

ENTRY = "runs/E29/code/eval_queue.py"
SETTLE = 60      # train.done 出现后再等这么多秒（gpfs 上存档写完）


def main():
    gpu = os.environ["GPU"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    launch.require_clean(REPO)
    claims = R / ".evalclaims"
    claims.mkdir(exist_ok=True)
    while True:
        left = False
        for m in [x for x in D.methods(R) if x.startswith("b4_")]:
            if (claims / m).exists():
                continue
            left = True
            done = R / "ckpt" / m / "train.done"
            if not done.exists() or time.time() - done.stat().st_mtime < SETTLE:
                continue
            try:
                (claims / m).mkdir()
            except FileExistsError:
                continue
            src = R / "configs" / f"{m}.toml"
            cfg = C.load(src)
            infer = json.loads((REPO / cfg["model"]["config"]).read_text(encoding="utf-8"))
            launch.begin(R, m, [sys.executable, *sys.argv], repo=REPO, entry=ENTRY,
                         config_file=src.relative_to(REPO).as_posix(), args=cfg, config=infer)
            D.log(f"VAL-LAST {m} gpu{gpu}")
            try:
                D.Job(R, m, cfg, gpu)()
                launch.end(R, m, "ok")
                D.log(f"OK {m}")
            except subprocess.CalledProcessError as e:
                launch.end(R, m, "fail")
                D.log(f"FAIL {m}: {e}")
        if not left:
            D.log("all b4 evaluated or claimed")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
