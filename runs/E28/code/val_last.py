"""E28 只用（#212）：训练结束后中途改为只评第 2000 步（最后一个 ckpt）的 Val，不评中间 ckpt、不评 Test。
用户 2026-10-26 决定，为了提前出结果；驱动 scripts/finetune/run.py 训完后原本的逐 ckpt 评测与 Test 已被停掉。
做法：载入驱动模块，把方法配置的 [run] sweep 改成 "last" 后照常调用 Job（train.done 已在，跳过训练），
产物与驱动的 sweep = "last" 相同（sweep/<方法>/peak.json、preds/<方法>/val.jsonl、workbench eval E28）。
驱动那次启动记录记为 stopped，这次另记一条启动。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E28/code/val_last.py <方法> > runs/E28/ckpt/val_last_<方法>.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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


def main():
    m = sys.argv[1]
    gpu = os.environ["GPU"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=D.ENV["CUDA_DEVICE_ORDER"])
    if not (R / "ckpt" / m / "train.done").exists():
        raise SystemExit(f"{m} 还没训完")
    src = R / "configs" / f"{m}.toml"
    cfg = C.load(src)
    cfg["run"]["sweep"] = "last"
    launch.end(R, m, "stopped")   # 驱动那次：训练完成，评测被停掉
    infer = json.loads((REPO / cfg["model"]["config"]).read_text(encoding="utf-8"))
    launch.begin(R, m, [sys.executable, *sys.argv], repo=REPO, entry="runs/E28/code/val_last.py",
                 config_file=src.relative_to(REPO).as_posix(), args=cfg, config=infer)
    D.log(f"VAL-LAST {m} gpu{gpu}")
    try:
        D.Job(R, m, cfg, gpu)()
        launch.end(R, m, "ok")
        D.log(f"OK {m}")
    except subprocess.CalledProcessError as e:
        launch.end(R, m, "fail")
        D.log(f"FAIL {m}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
