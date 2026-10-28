"""E36 只用（#233，照抄 E35 的 eval_queue.py）：在另一台机器（150）的空卡上评测 train_queue.py 训出的存档。
- 训练还在跑时：每个方法新存下的 ckpt（写完至少 SETTLE 秒）先评 Val（同驱动 Job.evaluate，写 sweep/<方法>/S/preds/step<N>/），
  靠 mkdir runs/E36/.evalclaims/<方法>_<step> 分工。
- 见到 ckpt/<方法>/train.done 后照驱动调用 Job（配置 [run] sweep = "last"：已评过的跳过，最后一个 ckpt 的 Val 即这个方法的
  正式结果，写 peak.json、preds/<方法>/val.jsonl、workbench eval E36），再给训练时写下的启动记录记上 end 与 status：
  训完为 ok；提前停（ckpt/<方法>/halt.json）也是 ok，停在哪一步、哪个量越线看 halt.json；提前停在第一个存档之前、
  一个 ckpt 都没有的记 halted_no_ckpt。不评 Test。全部方法都有结果（或训练失败）就退出。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E36/code/eval_queue.py > runs/E36/ckpt/eval_<host>_g<GPU>.log 2>&1 < /dev/null &
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

SETTLE = 120     # ckpt 写完后至少放这么多秒再读（torch.save 不是原子写）
QUIET = 300      # 没有完成标记的 Val 预测，最后一次写入后至少过这么多秒才算写完



def now():
    """文件系统上的当前时间：新写一个文件，取它的修改时间。A6000 的 NFS 给文件记的时间比容器时钟快约 8 小时，
    拿 time.time() 和文件的修改时间相减会是负数，存档永远等不满 SETTLE（2026-10-28）。"""
    f = R / ".evalclaims" / ".now"
    f.touch()
    return f.stat().st_mtime


def evaluated(job, claims, m, st):
    """这个存档的 Val 已评完：评它的进程写了完成标记（.evalclaims/<方法>_<step>/done），或（标记加上之前评的）预测文件
    已有一段时间没再写。只看 val.jsonl 在不在不够：baselines.fit 一开始就建好空文件，别的卡上的进程会把半截文件当成结果
    （2026-10-28 lr3e-6 第 1000 次存档：正式结果抄成了空文件，AUC@5 记成 0）。"""
    f = job.S / "preds" / f"step{st}" / "val.jsonl"
    if (claims / f"{m}_{st}" / "done").exists():
        return True
    return f.exists() and f.stat().st_size > 0 and now() - f.stat().st_mtime > QUIET


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
            if last_status(m) is not None:
                continue
            busy = True
            cfg = C.load(R / "configs" / f"{m}.toml")
            job = D.Job(R, m, cfg, gpu)
            done = (job.ck / "train.done").exists()
            ckpts = sorted(job.ck.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:]))
            for c in ckpts:                     # 中间存档先评
                st = c.stem[5:]
                if (job.S / "preds" / f"step{st}" / "val.jsonl").exists() or now() - c.stat().st_mtime < SETTLE:
                    continue
                try:
                    (claims / f"{m}_{st}").mkdir()
                except FileExistsError:
                    continue
                D.log(f"VAL {m} step{st} gpu{gpu}")
                try:
                    if not job.S.exists():
                        job.sh([D.WB, "-m", "workbench", "--runs", job.sw, "new", "S", "--title",
                                f"{R.name}/{m} ckpt sweep"], "sweep.log", gpu=False)
                    job.evaluate("val", st)
                    (claims / f"{m}_{st}" / "done").touch()
                except subprocess.CalledProcessError as e:
                    D.log(f"FAIL {m} step{st}: {e}")
                    (claims / f"{m}_{st}").rmdir()
            if not done or any(not evaluated(job, claims, m, c.stem[5:]) for c in ckpts):
                continue
            try:
                (claims / m).mkdir()
            except FileExistsError:
                continue
            if not ckpts:
                launch.end(R, m, "halted_no_ckpt")
                D.log(f"NO CKPT {m}")
                continue
            D.log(f"FINAL {m} gpu{gpu}")
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
