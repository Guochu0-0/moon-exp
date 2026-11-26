"""微调实验的驱动：训练 → 逐 ckpt 在 Val 上评测 → 按实验的数据版本选 ckpt 补评 Test → 写成正式记录。
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
from moonlib.data_version import (CURRENT, INLIER_SELECTION, LEGACY, ROOT_ENV, experiment_version,
                                  manifest_sha256, prediction_sha256, root_error, root_for, sha256)  # noqa: E402
from workbench import launch  # noqa: E402

Y = "/remote-home/xufang/YGC"
PY, WB = "/opt/envs/loftr/bin/python", "/opt/envs/wb/bin/python"
NICE = ["nice", "-n", "10", "ionice", "-c3"]
ENV = {"MOON_WEIGHTS": f"{Y}/weights", "MOON_RESULTS": f"{Y}/results",
       "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}   # GPU 编号与 nvidia-smi 一致（A6000 上默认顺序不同）
DATA_DEFAULT = {LEGACY: f"{Y}/dataset/Moon", CURRENT: f"{Y}/dataset/Moon-v2"}
ENTRY = "scripts/finetune/run.py"


class EvaluationError(RuntimeError):
    pass


def validate_selection(cfg, version):
    opts = cfg["run"]
    inlier = opts["select"] == INLIER_SELECTION
    if version == CURRENT and (not inlier or opts["sweep"] != "all" or opts["split"] != "train"):
        raise C.ConfigError('Moon-v2 要求 [run] select = "val_inlier_mean"、sweep = "all"，且仅使用 Train 训练')
    if version == LEGACY and inlier:
        raise C.ConfigError('select = "val_inlier_mean" 适用于 Moon-v2')


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
        self.version = experiment_version(R)
        validate_selection(cfg, self.version)
        self.infer = cfg["model"]["config"]          # 推理配置，评测用
        self.roots = {v: root_for(v, default=DATA_DEFAULT[v]) for v in ROOT_ENV}
        self.env = {"TORCH_HOME": MODELS[cfg["model"]["name"]].TORCH_HOME,
                    **{name: str(self.roots[v]) for v, name in ROOT_ENV.items()}}
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
        if self.version == CURRENT:
            return self.evaluate_versioned(split, int(st))
        if not (self.S / "preds" / f"step{st}" / f"{split}.jsonl").exists():
            self.sh([PY, "-m", "baselines.match", self.infer, "--split", split, "--weights",
                     self.ck / f"ckpt_{st}.pt", "--name", f"step{st}", "--out", self.sw / "match"], "sweep.log")
            self.sh([WB, "-m", "baselines.fit", self.sw / "match" / f"step{st}", "--split", split, "--run", self.S],
                    "sweep.log", gpu=False)

    def evaluate_versioned(self, split, step):
        from workbench.dataset import Dataset
        from workbench.evaluate import evaluate_preds
        from workbench.protocol import RANSAC_PX
        from workbench.records import Run
        from workbench.validation import evaluate_val, inlier_rate

        ds = Dataset(self.roots[self.version])
        if not ds.has_split(split):
            raise EvaluationError(root_error(self.version, ds.root, split))
        if split == "val" and ds.labelled(split):
            raise EvaluationError("Moon-v2 Val 应为无标注验证集，请核对 MOON_DATA_V2")
        pairs = ds.pairs(split) if split == "val" else ds.labelled(split)
        if not pairs:
            raise EvaluationError(f"Moon-v2 {split} 评测清单为空")
        method = f"step{step}"
        weight = self.ck / f"ckpt_{step}.pt"
        src = self.S / "preds" / method
        prediction = src / f"{split}.jsonl"
        meta_path = src / f"{split}.meta.json"
        expected = {"data_version": self.version, "split": split, "step": step, "ransac_px": RANSAC_PX,
                    "manifest_sha256": manifest_sha256(pairs), "weights_sha256": sha256(weight),
                    "config_sha256": sha256(REPO / self.infer)}
        run = Run("S", self.S, {"data_version": self.version}, [method])

        def result():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (any(meta.get(k) != v for k, v in expected.items())
                        or meta.get("prediction_sha256") != prediction_sha256(prediction)):
                    return None
                preds = run.preds(method, split)
                if set(preds) != set(pairs):
                    return None
                if split == "val":
                    evaluation = evaluate_val(pairs, preds)
                    if evaluation["summary"]["inlier_mean"] is None:
                        return evaluation
                if any(inlier_rate(p) is None for p in preds.values()):
                    return None
                if meta.get("matches_sha256") != sha256(src / f"{split}_matches.npz"):
                    return None
                return evaluation if split == "val" else evaluate_preds(ds, split, preds)
            except (OSError, ValueError, KeyError):
                return None

        cached = result()
        if cached and (split == "test" or cached["summary"]["inlier_mean"] is not None):
            return cached
        # 匹配或拟合异常后保留逐对原因，只有符合本次清单的完整候选可以复用。
        meta_path.unlink(missing_ok=True)
        self.sh([PY, "-m", "baselines.match", self.infer, "--split", split, "--weights", weight,
                 "--name", method, "--out", self.sw / "match", "--run", self.R], "sweep.log")
        self.sh([WB, "-m", "baselines.fit", self.sw / "match" / method, "--split", split, "--run", self.S],
                "sweep.log", gpu=False)
        fresh = result()
        if fresh is None:
            raise EvaluationError(f"{method}/{split} 的预测来源或逐对覆盖不完整，详见 sweep.log")
        return fresh

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
        if self.version == CURRENT:
            base += ["--data", self.roots[self.version]]
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
        candidates = {}

        def evaluate(job, c):
            st = c.stem[5:]
            try:
                result = job.evaluate("val", st)
            except (EvaluationError, subprocess.CalledProcessError, OSError, ValueError) as e:
                if self.version != CURRENT:
                    raise
                candidates[int(st)] = {"step": int(st), "error": str(e)}
                return
            if self.version == CURRENT:
                candidates[int(st)] = {"step": int(st), "summary": result["summary"], "coverage": result["coverage"]}
            if online and not (job.sw / "neg" / f"step{st}.jsonl").exists():
                job.sh([PY, "-m", "finetune.negpairs", job.infer, "--split", "val", "--n", "200",
                        "--weights", f"step{st}={c}", "--out", job.sw / "neg"], "neg.log")

        gpus = self.cfg["run"]["eval_gpus"]
        if not gpus:
            for c in ckpts:
                evaluate(self, c)
            return [candidates[k] for k in sorted(candidates)]
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
        return [candidates[k] for k in sorted(candidates)]

    def __call__(self):
        self.run_training()
        self.run_evaluation()

    def run_training(self):
        self.ck.mkdir(parents=True, exist_ok=True)
        if not (self.ck / "train.done").exists():
            log(f"TRAIN {self.m} [{', '.join(C.parts_of(self.cfg))}]")
            self.train()
            (self.ck / "train.done").touch()

    def run_evaluation(self):
        if self.version == LEGACY:
            self.evaluate_checkpoints()
            return
        # 全流程和 --eval-only 使用不同的队列占位，评测产物共用同一把锁。
        lock = self.ck / ".evaluating"
        try:
            lock.mkdir()
        except FileExistsError:
            raise EvaluationError(f"{self.m} 已有评测进程") from None
        try:
            self.evaluate_checkpoints()
        finally:
            lock.rmdir()

    def evaluate_checkpoints(self):
        R, m, ck = self.R, self.m, self.ck
        online = self.cfg["model"]["name"] == "loftr" and ("cexp" in self.cfg or "neg" in self.cfg)
        log(f"SWEEP {m}")
        if not self.S.exists():
            self.sh([WB, "-m", "workbench", "--runs", self.sw, "new", "S", "--title", f"{R.name}/{m} ckpt sweep",
                     "--data-version", self.version],
                    "sweep.log", gpu=False)
        ckpts = sorted(ck.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:]))
        if self.version == CURRENT:
            self.select_by_inlier_rate(ckpts, online)
            return
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

    def select_by_inlier_rate(self, ckpts, online):
        from tempfile import TemporaryDirectory

        from workbench.evaluate import jsonable

        if experiment_version(self.S) != self.version:
            raise EvaluationError(f"{self.S}/exp.toml 的数据版本与实验不符")
        candidates = self.sweep_val(ckpts, False)
        rated = [c for c in candidates if c.get("summary", {}).get("inlier_mean") is not None]
        chosen = max(rated, key=lambda c: c["summary"]["inlier_mean"], default=None)
        step = chosen["step"] if chosen else None
        peak = {"data_version": self.version, "metric": INLIER_SELECTION, "step": step, "candidates": candidates,
                "val": chosen["summary"] if chosen else None, "test": None}
        def write_peak(name):
            with TemporaryDirectory(dir=self.sw) as tmp:
                path = Path(tmp) / name
                path.write_text(json.dumps(jsonable(peak), indent=1, allow_nan=False) + "\n", encoding="utf-8")
                path.replace(self.sw / name)

        try:
            if step is None:
                raise EvaluationError("所有 Moon-v2 Val 候选均缺少完整有效的逐对预测")
            log(f"TEST {self.m} step{step}")
            peak["test"] = self.evaluate("test", step)["summary"]
            src = self.S / "preds" / f"step{step}"
            meta = json.loads((src / "val.meta.json").read_text(encoding="utf-8"))
            peak.update(weights_sha256=meta["weights_sha256"], val_manifest_sha256=meta["manifest_sha256"])
            dst = self.R / "preds" / self.m
            dst.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=dst) as tmp:
                for split in ("val", "test"):
                    for suffix in (".jsonl", "_matches.npz", ".meta.json"):
                        name = split + suffix
                        shutil.copy2(src / name, Path(tmp) / name)
                for split in ("val", "test"):
                    for suffix in (".jsonl", "_matches.npz", ".meta.json"):
                        name = split + suffix
                        (Path(tmp) / name).replace(dst / name)
        except (EvaluationError, subprocess.CalledProcessError, OSError, ValueError) as e:
            peak["error"] = str(e)
            write_peak("peak_attempt.json")
            raise
        write_peak("peak.json")
        (self.sw / "peak_attempt.json").unlink(missing_ok=True)
        if online:
            log("Moon-v2 Val 无标注，跳过旧版基于有标注 Val 的负样本诊断")
        self.sh([WB, "-m", "workbench", "eval", self.R.name], "sweep.log", gpu=False)
        log(f"DONE {self.m}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="实验目录 runs/<id>")
    ap.add_argument("--method", help="只运行指定方法")
    phase = ap.add_mutually_exclusive_group()
    phase.add_argument("--train-only", action="store_true", help="只训练，另起 eval-only 评测")
    phase.add_argument("--eval-only", action="store_true", help="在单卡上评测已完成的训练")
    a = ap.parse_args(argv)
    if (a.train_only or a.eval_only) and not a.method:
        ap.error("分开运行训练、评测时须用 --method 指定方法")
    R = Path(a.run).resolve()
    if R.parent != REPO / "runs" or not (R / "configs").is_dir():
        ap.error(f"应给本仓库的实验目录 runs/<id>，且其下有 configs/：{R}")
    version = experiment_version(R)
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
        validate_selection(cfg, version)
        expected = 1 if a.eval_only else (cfg["run"]["ddp"] or 1)
        if len(devices) != expected:
            raise C.ConfigError(f"{m} 需要 {expected} 张 GPU，得到 GPU={gpu}")
        return cfg

    for m in candidates():       # 先把全部配置展开一遍，有错就不开跑
        load(m)
        if a.eval_only and not (R / "ckpt" / m / "train.done").is_file():
            ap.error(f"{m} 还没有完成训练")
    phase_name = "test" if a.eval_only else "train" if a.train_only else "train_eval"
    claims = R / ".claims"
    claims.mkdir(exist_ok=True)
    attempted = set()
    failed = False
    while True:
        picked = None
        for m in candidates():   # 每轮重新列，中途新增的方法也会被领走
            if m in attempted:
                continue
            try:
                record = f"{m}_eval" if a.eval_only else m
                (claims / record).mkdir()
            except FileExistsError:
                continue
            picked = m
            break
        if picked is None:
            log("queue empty")
            if failed:
                raise SystemExit(1)
            return
        m = picked
        attempted.add(m)
        (claims / record / "host").write_text(f"{socket.gethostname()} gpu{gpu} {time.strftime('%F %T %z')}\n")
        log(f"START {R.name}/{m} gpu{gpu}")
        src = R / "configs" / f"{m}.toml"
        try:
            cfg = load(m)
        except C.ConfigError as e:
            log(f"FAIL {m}: {e}")
            continue
        infer = json.loads((REPO / cfg["model"]["config"]).read_text(encoding="utf-8"))
        launch.begin(R, record, [sys.executable, *sys.argv], repo=REPO, entry=ENTRY,
                     config_file=src.relative_to(REPO).as_posix(), args=cfg, config=infer, phase=phase_name,
                     data_version=version,
                     batch_transport=os.environ.get("MOON_BATCH_TRANSPORT", "shared_memory"))
        try:
            job = Job(R, m, cfg, gpu)
            if a.train_only:
                job.run_training()
            elif a.eval_only:
                job.run_evaluation()
            else:
                job()
            launch.end(R, record, "ok")
            log(f"OK {m}")
        except (subprocess.CalledProcessError, EvaluationError, OSError, ValueError) as e:
            launch.end(R, record, "fail")
            log(f"FAIL {m}: {e}")
            if version == CURRENT:
                failed = True
                (claims / record / "host").unlink()
                (claims / record).rmdir()


if __name__ == "__main__":
    main()
