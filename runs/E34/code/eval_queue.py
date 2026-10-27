"""E34 只用（#228）：在另一张卡上评测 train_queue.py 训出的存档（参照 E28 的 eval_ahead.py、E31 的 eval_queue.py）。
- 训练还在跑时：每个方法新存下的 ckpt（写完至少 SETTLE 秒）先评 Val（同驱动 Job.evaluate，写 sweep/<方法>/S/preds/step<N>/），
  靠 mkdir runs/E34/.evalclaims/<方法>_<step> 分工。
- 见到 ckpt/<方法>/train.done 后照驱动调用 Job（配置 [run] sweep = "last"：已评过的跳过，最后一个 ckpt 的 Val 即这个方法的
  正式结果，写 peak.json、preds/<方法>/val.jsonl、workbench eval E34），再给训练时写下的启动记录记上 end 与 status：
  训完为 ok；提前停（ckpt/<方法>/halt.json）也是 ok，停在哪一步、哪个量越线看 halt.json；提前停在第一个存档之前、
  一个 ckpt 都没有的记 halted_no_ckpt。不评 Test。全部方法都有结果（或训练失败）就退出。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E34/code/eval_queue.py > runs/E34/ckpt/eval_<host>_g<GPU>.log 2>&1 < /dev/null &
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
                if (job.S / "preds" / f"step{st}" / "val.jsonl").exists() or time.time() - c.stat().st_mtime < SETTLE:
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
                except subprocess.CalledProcessError as e:
                    D.log(f"FAIL {m} step{st}: {e}")
                    (claims / f"{m}_{st}").rmdir()
            if not done or any(not (job.S / "preds" / f"step{c.stem[5:]}" / "val.jsonl").exists() for c in ckpts):
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
