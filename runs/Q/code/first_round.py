# 已停用（#76）：产物写到 results/finetune/，不符合 docs/agents/experiments.md。新实验用 scripts/finetune/run.py；本文件留待「历史代码与结果整理」（#77）归位。
"""两条路线首轮实验的任务队列（「【类 RIPE】首轮实验设计」#51、「【伪标签】首轮实验设计」#53）。

在 scripts/finetune/ripe49.py 上改：负样本对监控只对在线信号的任务做（参数含 --w-cexp 或 --neg）；
Val sweep 之后按 Val AUC@5 峰值（不含 step 0）选 ckpt，自动在 Test 上补评这一个 ckpt。

    GPU=<空卡> /opt/envs/loftr/bin/python scripts/finetune/first_round.py scripts/finetune/jobs/first-round.txt

jobs 文件每行 `<name> <finetune.train 参数...>`，# 开头为注释；每张空卡起一个，靠 mkdir 占位（_claims/<name>）。
产物在 $MOON_RESULTS/finetune/<name>/：train.log、log.jsonl、ckpt_*.pt、match/、sweep/S（metrics.json 含峰值 ckpt 的 Test）、
neg/ 与 collapse.json（在线信号任务）、peak.json（选中的 step 与 Val / Test 指标）。
"""
from __future__ import annotations

import json
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


def evaluate(R, split, st, c, gpu):
    if not (R / f"sweep/S/preds/step{st}/{split}.jsonl").exists():
        run([PY, "-m", "baselines.match", CFG, "--split", split, "--weights", c, "--name", f"step{st}",
             "--out", R / "match"], R / "sweep.log", gpu)
        run([WB, "-m", "baselines.fit", R / "match" / f"step{st}", "--split", split, "--run", R / "sweep/S"],
            R / "sweep.log")


def peak(R, metric="auc@5"):
    m = json.loads((R / "sweep/S/metrics.json").read_text(encoding="utf-8"))["methods"]
    pts = [(int(k[4:]), v["val"]["summary"][metric]) for k, v in m.items()
           if k.startswith("step") and "val" in v and int(k[4:]) > 0]
    return max(pts, key=lambda p: p[1])[0]


def job(name, args, gpu):
    R = F / name
    R.mkdir(parents=True, exist_ok=True)
    online = "--w-cexp" in args or "--neg" in args
    if not (R / "train.done").exists():
        log(f"TRAIN {name} {' '.join(args)}")
        run([PY, "-m", "finetune.train", CFG, "--out", R, *args], R / "train.log", gpu)
        (R / "train.done").touch()
    log(f"SWEEP {name}")
    if not (R / "sweep/S").exists():
        run([WB, "-m", "workbench", "--runs", R / "sweep", "new", "S", "--title", f"{name} ckpt sweep"], R / "sweep.log")
    for c in sorted(R.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:])):
        st = c.stem[5:]
        evaluate(R, "val", st, c, gpu)
        if online and not (R / "neg" / f"step{st}.jsonl").exists():
            run([PY, "-m", "finetune.negpairs", CFG, "--split", "val", "--n", "200", "--weights", f"step{st}={c}",
                 "--out", R / "neg"], R / "neg.log", gpu)
    run([WB, "-m", "workbench", "--runs", R / "sweep", "eval", "S"], R / "sweep.log")
    st = peak(R)
    log(f"TEST {name} step{st}")
    evaluate(R, "test", st, R / f"ckpt_{st}.pt", gpu)
    run([WB, "-m", "workbench", "--runs", R / "sweep", "eval", "S"], R / "sweep.log")
    m = json.loads((R / "sweep/S/metrics.json").read_text(encoding="utf-8"))["methods"][f"step{st}"]
    (R / "peak.json").write_text(json.dumps({"step": st, "val": m["val"]["summary"], "test": m["test"]["summary"]},
                                            indent=1), encoding="utf-8")
    if online:
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
