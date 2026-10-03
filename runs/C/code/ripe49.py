# 已停用（#76）：产物写到 results/finetune/，不符合 docs/agents/experiments.md。新实验用 scripts/finetune/run.py；本文件留待「历史代码与结果整理」（#77）归位。
"""粗级闭式期望塌缩测试的任务队列（「【类 RIPE】粗级闭式期望从 zero-shot 单独训练：是否塌缩」#49）。

与 queue.sh + scenes.sh 同一套产物布局，另加负样本对监控与塌缩汇总；写成 Python 是为了能从 worktree 会话里直接起远端任务。

    GPU=<空卡> /opt/envs/loftr/bin/python scripts/finetune/ripe49.py scripts/finetune/jobs/ripe49.txt

jobs 文件每行 `<name> <finetune.train 参数...>`，# 开头为注释；每张空卡起一个，靠 mkdir 占位（_claims/<name>）。
每个任务：训练 → 逐 ckpt 在 Val 上匹配、拟合（sweep/S）→ 负样本对监控（neg/）→ workbench eval →
scripts/finetune/collapse.py 汇总（collapse.txt / collapse.json）。产物在 $MOON_RESULTS/finetune/<name>/。
"""
from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Y = "/remote-home/xufang/YGC"
ENV = {"MOON_DATA": f"{Y}/dataset/Moon", "MOON_WEIGHTS": f"{Y}/weights", "MOON_RESULTS": f"{Y}/results",
       "TORCH_HOME": "/opt/torch_home", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}   # GPU 编号与 nvidia-smi 一致
F = Path(ENV["MOON_RESULTS"]) / "finetune"
CFG = "configs/baselines/anymatch_loftr.json"
PY, WB = "/opt/envs/loftr/bin/python", "/opt/envs/wb/bin/python"
NICE = ["nice", "-n", "10", "ionice", "-c3"]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run(cmd, logfile, gpu=None):
    env = {**os.environ, **ENV}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(logfile, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n")
        f.flush()
        subprocess.run(NICE + [str(c) for c in cmd], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, check=True)


def job(name, args, gpu):
    R = F / name
    R.mkdir(parents=True, exist_ok=True)
    if not (R / "train.done").exists():
        log(f"TRAIN {name} {' '.join(args)}")
        run([PY, "-m", "finetune.train", CFG, "--out", R, *args], R / "train.log", gpu)
        (R / "train.done").touch()
    log(f"SWEEP {name}")
    if not (R / "sweep/S").exists():
        run([WB, "-m", "workbench", "--runs", R / "sweep", "new", "S", "--title", f"{name} ckpt sweep"], R / "sweep.log")
    ckpts = sorted(R.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:]))
    for c in ckpts:
        st = c.stem[5:]
        if not (R / f"sweep/S/preds/step{st}/val.jsonl").exists():
            run([PY, "-m", "baselines.match", CFG, "--split", "val", "--weights", c, "--name", f"step{st}",
                 "--out", R / "match"], R / "sweep.log", gpu)
            run([WB, "-m", "baselines.fit", R / "match" / f"step{st}", "--split", "val", "--run", R / "sweep/S"],
                R / "sweep.log")
        done = (R / "neg" / f"step{st}.jsonl").exists()
        if not done:
            run([PY, "-m", "finetune.negpairs", CFG, "--split", "val", "--n", "200", "--weights", f"step{st}={c}",
                 "--out", R / "neg"], R / "neg.log", gpu)
    run([WB, "-m", "workbench", "--runs", R / "sweep", "eval", "S"], R / "sweep.log")
    run([WB, "scripts/finetune/collapse.py", R, "--json", R / "collapse.json"], R / "collapse.txt")
    log(f"DONE {name}")


def main():
    jobs, gpu = Path(sys.argv[1]).resolve(), os.environ["GPU"]
    (F / "_claims").mkdir(parents=True, exist_ok=True)
    while True:
        picked = None
        for line in jobs.read_text(encoding="utf-8").splitlines():   # 每轮重读，中途追加的任务也会被领走
            parts = shlex.split(line)
            if not parts or parts[0].startswith("#"):
                continue
            try:
                (F / "_claims" / parts[0]).mkdir()
            except FileExistsError:
                continue
            picked = parts
            break
        if picked is None:
            log("queue empty")
            return
        name, args = picked[0], picked[1:]
        (F / "_claims" / name / "host").write_text(f"{socket.gethostname()} gpu{gpu} {time.strftime('%T')}\n")
        log(f"START {name} gpu{gpu}")
        try:
            job(name, args, gpu)
            log(f"OK {name}")
        except subprocess.CalledProcessError as e:
            log(f"FAIL {name}: {e}")


if __name__ == "__main__":
    main()
