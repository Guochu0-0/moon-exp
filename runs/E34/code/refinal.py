"""E34 只用（#228）：给已记了结束状态的方法重做一遍收尾（同驱动 Job.__call__：存档已评过的跳过，重写 peak.json、
preds/<方法>/、workbench eval E34），不动启动记录。用于修 eval_queue.py 的并发问题留下的错结果
（2026-10-28 lr3e-6：正式结果抄成了空文件）。先重算 sweep 的 metrics.json。

    GPU=<卡> /opt/envs/loftr/bin/python runs/E34/code/refinal.py lr3e-6
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("driver", REPO / "scripts" / "finetune" / "run.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
from finetune import config as C  # noqa: E402
from workbench import launch  # noqa: E402

if __name__ == "__main__":
    gpu = os.environ["GPU"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    launch.require_clean(REPO)
    for m in sys.argv[1:]:
        job = D.Job(R, m, C.load(R / "configs" / f"{m}.toml"), gpu)
        D.log(f"REFINAL {m}")
        job()
