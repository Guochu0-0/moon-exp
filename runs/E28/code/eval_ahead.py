"""E28 只用（#212）：训练还在跑时，用别的空卡先评已经存下的中间 ckpt 的 Val，不等训练结束后在训练卡上逐个排队。
命令与写的位置同 scripts/finetune/run.py 的 Job.evaluate（sweep/<方法>/S/preds/step<N>/val.jsonl），驱动训完后见到已有的
Val 预测就跳过，只剩最后一个 ckpt 的 Val 与峰值的 Test。最后一个 ckpt（stop_at）留给驱动，免得两边同时评。
多个进程靠 mkdir runs/E28/.evalclaims/<方法>_<step> 分工；评失败的放回去，由驱动照常补评。全部方法都训完、没有可评的就退出。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E28/code/eval_ahead.py > runs/E28/ckpt/eval_<host>_g<GPU>.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import importlib.util
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

SETTLE = 120     # ckpt 写完后至少放这么多秒再读（torch.save 不是原子写）


def main():
    gpu = os.environ["GPU"]
    claims = R / ".evalclaims"
    claims.mkdir(exist_ok=True)
    while True:
        busy = False
        for m in D.methods(R):
            cfg = C.load(R / "configs" / f"{m}.toml")
            job = D.Job(R, m, cfg, gpu)
            if not (job.ck / "train.done").exists():
                busy = True
            for c in sorted(job.ck.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:])):
                st = int(c.stem[5:])
                if st == 0 or st >= cfg["run"]["stop_at"] or (job.S / "preds" / f"step{st}" / "val.jsonl").exists():
                    continue
                if time.time() - c.stat().st_mtime < SETTLE:
                    busy = True
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
                    job.evaluate("val", str(st))
                except subprocess.CalledProcessError as e:
                    D.log(f"FAIL {m} step{st}: {e}")
                    (claims / f"{m}_{st}").rmdir()
        if not busy:
            D.log("all trained, nothing left")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
