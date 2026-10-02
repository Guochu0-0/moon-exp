"""微调实验的任务队列：训练 → 逐 ckpt 在 Val 上评测 → 按 Val AUC@5 峰值选 ckpt 补评 Test → 写成正式记录（#76）。
规矩见 docs/agents/experiments.md，产物布局见 workbench/RECORDS.md「训练产物」。

    GPU=<空卡> /opt/envs/loftr/bin/python scripts/finetune/run.py runs/P/code/jobs.txt [--trainer roma]

- 任务清单放在实验自己的 runs/<id>/code/ 下，实验 id 取清单所在的实验目录。每行 `<方法> <训练参数...>`，# 开头为注释。
  每跑完一个就从头重读清单，所以中途追加的任务也会被领走；每张空卡起一个，靠 mkdir runs/<id>/.claims/<方法> 占位。
- 启动前检查代码已提交（workbench.launch），每个方法的启动记录写进 runs/<id>/launch/<方法>.json。
- --trainer loftr（默认，finetune.train，AnyMatch-LoFTR）/ roma（finetune.train_roma，AnyMatch-RoMa + minmax）。

每个方法 <m> 的产物（相对 runs/<id>/）：
  ckpt/<m>/            ckpt_*.pt、log.jsonl、args.json、train.log、sweep.log（不进 git，不同步）
  tb/<m>/scalars/      训练曲线
  sweep/<m>/S/         逐 ckpt 的工作台记录（方法名 step<N>）；只有 exp.toml、metrics.json 进 git
  sweep/<m>/match/     逐 ckpt 的原始点对（不进 git）
  sweep/<m>/peak.json  选中的 step 与 Val / Test 指标
  sweep/<m>/neg/、collapse.json   负样本对监控（LoFTR，参数含 --w-cexp 或 --neg 时）
  preds/<m>/           选中 step 的 Val、Test 预测，即这个方法的正式结果
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workbench import launch  # noqa: E402

Y = "/remote-home/xufang/YGC"
PY, WB = "/opt/envs/loftr/bin/python", "/opt/envs/wb/bin/python"
NICE = ["nice", "-n", "10", "ionice", "-c3"]
ENV = {"MOON_DATA": f"{Y}/dataset/Moon", "MOON_WEIGHTS": f"{Y}/weights", "MOON_RESULTS": f"{Y}/results",
       "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}   # GPU 编号与 nvidia-smi 一致（A6000 上默认顺序不同）
TRAINERS = {
    "loftr": {"module": "finetune.train", "cfg": "configs/baselines/anymatch_loftr.json",
              "env": {"TORCH_HOME": "/opt/torch_home"}},
    "roma": {"module": "finetune.train_roma", "cfg": "configs/baselines/anymatch_roma__minmax.json",
             "env": {"TORCH_HOME": f"{Y}/weights/torch_home"}},
}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run(cmd, logfile, env, gpu=None):
    env = {**os.environ, **ENV, **env}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(logfile, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n")
        f.flush()
        subprocess.run(NICE + [str(c) for c in cmd], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, check=True)


class Job:
    def __init__(self, R: Path, m: str, args: list[str], trainer: str, gpu):
        self.R, self.m, self.args, self.gpu = R, m, args, gpu
        self.t = TRAINERS[trainer]
        self.ck, self.sw = R / "ckpt" / m, R / "sweep" / m
        self.S = self.sw / "S"

    def sh(self, cmd, logname, gpu=True):
        run(cmd, self.ck / logname, self.t["env"], self.gpu if gpu else None)

    def evaluate(self, split, st):
        if not (self.S / "preds" / f"step{st}" / f"{split}.jsonl").exists():
            self.sh([PY, "-m", "baselines.match", self.t["cfg"], "--split", split, "--weights",
                     self.ck / f"ckpt_{st}.pt", "--name", f"step{st}", "--out", self.sw / "match"], "sweep.log")
            self.sh([WB, "-m", "baselines.fit", self.sw / "match" / f"step{st}", "--split", split, "--run", self.S],
                    "sweep.log", gpu=False)

    def metrics(self):
        self.sh([WB, "-m", "workbench", "--runs", self.sw, "eval", "S"], "sweep.log", gpu=False)
        return json.loads((self.S / "metrics.json").read_text(encoding="utf-8"))["methods"]

    def peak(self, metric="auc@5"):
        pts = [(int(k[4:]), v["val"]["summary"][metric]) for k, v in self.metrics().items()
               if k.startswith("step") and "val" in v and int(k[4:]) > 0]
        return max(pts, key=lambda p: p[1])[0]

    def __call__(self):
        R, m, ck = self.R, self.m, self.ck
        ck.mkdir(parents=True, exist_ok=True)
        online = "--w-cexp" in self.args or "--neg" in self.args
        if not (ck / "train.done").exists():
            log(f"TRAIN {m} {' '.join(self.args)}")
            self.sh([PY, "-m", self.t["module"], self.t["cfg"], "--out", ck, "--tb", R / "tb" / m / "scalars",
                     *self.args], "train.log")
            (ck / "train.done").touch()
        log(f"SWEEP {m}")
        if not self.S.exists():
            self.sh([WB, "-m", "workbench", "--runs", self.sw, "new", "S", "--title", f"{R.name}/{m} ckpt sweep"],
                    "sweep.log", gpu=False)
        for c in sorted(ck.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:])):
            st = c.stem[5:]
            self.evaluate("val", st)
            if online and self.t["module"] == "finetune.train" and not (self.sw / "neg" / f"step{st}.jsonl").exists():
                self.sh([PY, "-m", "finetune.negpairs", self.t["cfg"], "--split", "val", "--n", "200",
                         "--weights", f"step{st}={c}", "--out", self.sw / "neg"], "neg.log")
        st = self.peak()
        log(f"TEST {m} step{st}")
        self.evaluate("test", st)
        s = self.metrics()[f"step{st}"]
        (self.sw / "peak.json").write_text(json.dumps({"step": st, "metric": "val auc@5",
                                                       "val": s["val"]["summary"], "test": s["test"]["summary"]},
                                                      indent=1) + "\n", encoding="utf-8")
        dst = R / "preds" / m   # 选中的 step 即这个方法的正式结果
        dst.mkdir(parents=True, exist_ok=True)
        for f in (self.S / "preds" / f"step{st}").iterdir():
            shutil.copy2(f, dst / f.name)
        if online and (self.sw / "neg").exists():
            self.sh([WB, "scripts/finetune/collapse.py", self.sw, "--json", self.sw / "collapse.json"], "collapse.txt",
                    gpu=False)
        self.sh([WB, "-m", "workbench", "eval", R.name], "sweep.log", gpu=False)
        log(f"DONE {m}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jobs", help="runs/<id>/code/ 下的任务清单")
    ap.add_argument("--trainer", default="loftr", choices=sorted(TRAINERS))
    a = ap.parse_args(argv)
    jobs = Path(a.jobs).resolve()
    if jobs.parent.name != "code" or jobs.parents[2] != REPO / "runs":
        ap.error(f"任务清单应放在本仓库的 runs/<id>/code/ 下：{jobs}")
    R = jobs.parents[1]
    gpu = os.environ["GPU"]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=ENV["CUDA_DEVICE_ORDER"])   # launch 记录里也要有
    launch.require_clean(REPO)   # 清单本身也必须已提交
    claims = R / ".claims"
    claims.mkdir(exist_ok=True)
    while True:
        picked = None
        for line in jobs.read_text(encoding="utf-8").splitlines():   # 每轮重读，中途追加的任务也会被领走
            parts = shlex.split(line)
            if not parts or parts[0].startswith("#"):
                continue
            try:
                (claims / parts[0]).mkdir()
            except FileExistsError:
                continue
            picked = parts
            break
        if picked is None:
            log("queue empty")
            return
        m, args = picked[0], picked[1:]
        (claims / m / "host").write_text(f"{socket.gethostname()} gpu{gpu} {time.strftime('%F %T %z')}\n")
        log(f"START {R.name}/{m} gpu{gpu}")
        launch.begin(R, m, [sys.executable, *sys.argv, "::", m, *args], repo=REPO)
        try:
            Job(R, m, args, a.trainer, gpu)()
            launch.end(R, m, "ok")
            log(f"OK {m}")
        except subprocess.CalledProcessError as e:
            launch.end(R, m, "fail")
            log(f"FAIL {m}: {e}")


if __name__ == "__main__":
    main()
