"""微调实验的驱动：训练 → 逐 ckpt 在 Val 上评测 → 按 Val AUC@5 峰值选 ckpt 补评 Test → 写成正式记录（#76、#86）。
规矩见 .claude/skills/server/SKILL.md，配置格式见 finetune/config.py，产物布局见 .claude/skills/records/formats.md「训练产物」。

    GPU=<空卡[,空卡...]> /opt/envs/loftr/bin/python scripts/finetune/run.py runs/<id> [--method <方法>]

- 每个方法一份 runs/<id>/configs/<方法>.toml；文件名以 _ 开头的只给别的配置当 base，不单独跑。
  每跑完一个就重新列一遍配置，所以中途新增（并提交）的方法也会被领走；每张空卡起一个，靠 mkdir runs/<id>/.claims/<方法> 占位。
- 启动前检查代码与配置已提交（workbench.launch），每个方法的启动记录（commit、完整展开的配置等）写进 runs/<id>/launch/<方法>.json。

每个方法 <m> 的产物（相对 runs/<id>/）：
  ckpt/<m>/            ckpt_*.pt、log.jsonl、args.json、train.log、sweep.log（不进 git，不同步）
  tb/<m>/scalars/      训练曲线
  sweep/<m>/S/         逐 ckpt 的工作台记录（方法名 step<N>）；只有 exp.toml、metrics.json 进 git
  sweep/<m>/match/     逐 ckpt 的原始点对（不进 git）
  sweep/<m>/peak.json  选中的 step 与 Val / Test 指标（[run] sweep = "last" 时只评最后一个 ckpt 的 Val、不评 Test，#209）
  sweep/<m>/neg/、collapse.json   负样本对监控（LoFTR，配置含 [cexp] 或 [neg] 时）
  preds/<m>/           选中 step 的 Val、Test 预测，即这个方法的正式结果
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from finetune import config as C  # noqa: E402
from finetune.models import MODELS  # noqa: E402
from workbench import launch  # noqa: E402

Y = "/remote-home/xufang/YGC"
PY, WB = "/opt/envs/loftr/bin/python", "/opt/envs/wb/bin/python"
NICE = ["nice", "-n", "10", "ionice", "-c3"]
ENV = {"MOON_DATA": f"{Y}/dataset/Moon", "MOON_WEIGHTS": f"{Y}/weights", "MOON_RESULTS": f"{Y}/results",
       "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}   # GPU 编号与 nvidia-smi 一致（A6000 上默认顺序不同）
ENTRY = "scripts/finetune/run.py"


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


def methods(R: Path) -> list[str]:
    """runs/<id>/configs/ 下要跑的方法，按文件名排序。"""
    return sorted(p.stem for p in (R / "configs").glob("*.toml") if not p.stem.startswith("_"))


class Job:
    def __init__(self, R: Path, m: str, cfg: dict, gpu):
        self.R, self.m, self.cfg, self.gpu = R, m, cfg, gpu
        self.infer = cfg["model"]["config"]          # 推理配置，评测用
        self.env = {"TORCH_HOME": MODELS[cfg["model"]["name"]].TORCH_HOME}
        self.ck, self.sw = R / "ckpt" / m, R / "sweep" / m
        self.S = self.sw / "S"

    def sh(self, cmd, logname, gpu=True):
        if gpu and (self.cfg["run"]["segment_steps"] or self.cfg["run"]["eval_gpus"]):
            # 子进程结束后 CUDA 上下文已释放；继续前重新等空卡。
            while True:
                mem = subprocess.check_output(["timeout", "15", "nvidia-smi", "-i", str(self.gpu),
                       "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                if int(mem.strip()) < 100:
                    break
                time.sleep(30)
        run(cmd, self.ck / logname, self.env, self.gpu if gpu else None)

    def evaluate(self, split, st):
        if not (self.S / "preds" / f"step{st}" / f"{split}.jsonl").exists():
            self.sh([PY, "-m", "baselines.match", self.infer, "--split", split, "--weights",
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

    def train(self):
        """默认单次训练；显式分段时保存并恢复优化器、数据顺序及全部随机状态。"""
        import copy
        src = self.R / "configs" / f"{self.m}.toml"
        ddp = self.cfg["run"]["ddp"]
        launcher = [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ddp}"] if ddp else [PY]
        base = launcher + ["-m", "finetune.train", src, "--out", self.ck, "--tb", self.R / "tb" / self.m / "scalars"]
        interval = self.cfg["run"]["segment_steps"]
        if not interval:
            self.sh(base, "train.log")
            return
        total = self.cfg["optim"]["steps"]
        existing = sorted(self.ck.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:]))
        done = int(existing[-1].stem[5:]) if existing else 0
        while done < total:
            target = min(done + interval, total)
            resume = self.ck / f"ckpt_{done}.pt" if done else None
            cmd = base + ["--stop-at", str(target)] + (["--resume", resume] if resume else [])
            cfg = copy.deepcopy(self.cfg)
            cfg["run"].update(resume=str(resume) if resume else "")
            method = f"{self.m}_train"
            launch.begin(self.R, method, list(map(str, cmd)), entry="finetune.train", config_file=src.relative_to(REPO).as_posix(),
                         args=cfg, stop_at=target, config=json.loads((REPO / self.infer).read_text()))
            try:
                self.sh(cmd, "train.log")
            except BaseException:
                launch.end(self.R, method, "fail")
                raise
            extra = {}
            if self.cfg["run"]["record_hashes"]:
                runtime = json.loads((self.ck / "args.json").read_text())
                extra = {"source_hashes": runtime["source_hashes"], "runtime_commit": runtime["commit"]}
            launch.end(self.R, method, "ok", **extra)
            done = target
            if done < total:
                time.sleep(30)

    def sweep_val(self, ckpts, online):
        """每个存档的原始Val推理独立，结果目录按step隔离；不在工作线程选模或评Test。"""
        def evaluate(job, c):
            st = c.stem[5:]
            job.evaluate("val", st)
            if online and not (job.sw / "neg" / f"step{st}.jsonl").exists():
                job.sh([PY, "-m", "finetune.negpairs", job.infer, "--split", "val", "--n", "200",
                        "--weights", f"step{st}={c}", "--out", job.sw / "neg"], "neg.log")

        gpus = self.cfg["run"]["eval_gpus"]
        if not gpus:
            for c in ckpts:
                evaluate(self, c)
            return
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue, Empty
        queue = Queue()
        for c in ckpts:
            queue.put(c)

        def worker(gpu):
            job = Job(self.R, self.m, self.cfg, gpu)
            while True:
                try:
                    c = queue.get_nowait()
                except Empty:
                    return
                evaluate(job, c)

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for f in futures:
                f.result()

    def __call__(self):
        R, m, ck = self.R, self.m, self.ck
        ck.mkdir(parents=True, exist_ok=True)
        online = self.cfg["model"]["name"] == "loftr" and ("cexp" in self.cfg or "neg" in self.cfg)
        if not (ck / "train.done").exists():
            log(f"TRAIN {m} [{', '.join(C.parts_of(self.cfg))}]")
            self.train()
            (ck / "train.done").touch()
        log(f"SWEEP {m}")
        if not self.S.exists():
            self.sh([WB, "-m", "workbench", "--runs", self.sw, "new", "S", "--title", f"{R.name}/{m} ckpt sweep"],
                    "sweep.log", gpu=False)
        ckpts = sorted(ck.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:]))
        sweep = self.cfg["run"]["sweep"]
        val_only = sweep == "last"
        last_only = sweep != "all"
        self.sweep_val(ckpts[-1:] if last_only else ckpts, online)
        last = self.cfg["run"]["select"] == "last"   # 在 Test 上训练时不能按 Val 选（#98）
        st = int(ckpts[-1].stem[5:]) if last or last_only else self.peak()
        if not val_only:
            log(f"TEST {m} step{st}")
            self.evaluate("test", st)
        s = self.metrics()[f"step{st}"]
        metric = "last (val only)" if val_only else "last" if last or last_only else "val auc@5"
        (self.sw / "peak.json").write_text(json.dumps({"step": st, "metric": metric, "val": s["val"]["summary"],
                                                       "test": None if val_only else s["test"]["summary"]},
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
    ap.add_argument("run", help="实验目录 runs/<id>")
    ap.add_argument("--method", help="只运行指定方法")
    a = ap.parse_args(argv)
    R = Path(a.run).resolve()
    if R.parent != REPO / "runs" or not (R / "configs").is_dir():
        ap.error(f"应给本仓库的实验目录 runs/<id>，且其下有 configs/：{R}")
    gpu = os.environ["GPU"]
    devices = gpu.split(",")
    if any(not d.isdecimal() for d in devices) or len(devices) != len(set(devices)):
        ap.error("GPU 应是互不重复的 GPU 编号，以逗号分隔")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER=ENV["CUDA_DEVICE_ORDER"])   # launch 记录里也要有
    launch.require_clean(REPO)   # 配置本身也必须已提交
    if a.method and a.method not in methods(R):
        ap.error(f"未知方法 {a.method}")

    def candidates():
        return [a.method] if a.method else methods(R)

    def load(m):
        cfg = C.load(R / "configs" / f"{m}.toml")
        expected = cfg["run"]["ddp"] or 1
        if len(devices) != expected:
            raise C.ConfigError(f"{m} 需要 {expected} 张 GPU，得到 GPU={gpu}")
        return cfg

    for m in candidates():       # 先把全部配置展开一遍，有错就不开跑
        load(m)
    claims = R / ".claims"
    claims.mkdir(exist_ok=True)
    while True:
        picked = None
        for m in candidates():   # 每轮重新列，中途新增的方法也会被领走
            try:
                (claims / m).mkdir()
            except FileExistsError:
                continue
            picked = m
            break
        if picked is None:
            log("queue empty")
            return
        m = picked
        (claims / m / "host").write_text(f"{socket.gethostname()} gpu{gpu} {time.strftime('%F %T %z')}\n")
        log(f"START {R.name}/{m} gpu{gpu}")
        src = R / "configs" / f"{m}.toml"
        try:
            cfg = load(m)
        except C.ConfigError as e:
            log(f"FAIL {m}: {e}")
            continue
        infer = json.loads((REPO / cfg["model"]["config"]).read_text(encoding="utf-8"))
        launch.begin(R, m, [sys.executable, *sys.argv], repo=REPO, entry=ENTRY,      # 字段同 .claude/skills/records/formats.md「启动记录」
                     config_file=src.relative_to(REPO).as_posix(), args=cfg, config=infer)
        try:
            Job(R, m, cfg, gpu)()
            launch.end(R, m, "ok")
            log(f"OK {m}")
        except subprocess.CalledProcessError as e:
            launch.end(R, m, "fail")
            log(f"FAIL {m}: {e}")


if __name__ == "__main__":
    main()
