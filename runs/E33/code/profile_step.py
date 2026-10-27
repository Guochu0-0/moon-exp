"""E33 只用（#226）：测 RoMa RL 训练一步里各部分的耗时、GPU 利用率与显存。

两种测法，都直接调用 finetune.train.main（同一份配置、同一套代码）：
- plain：不插桩，跑 --stop-at N 步，读 log.jsonl 里每步的 sec（不含取数据）与取数据的时间；同时用 nvidia-smi 每 200 ms
  采一次这张卡的利用率与显存。
- ops：用 torch.profiler 记第 skip+1 … skip+5 步，按 GPU 时间与 CPU 时间各列前 40 个算子（看大头落在哪些算子上）。
- detail：在各部分的函数外面包一层计时（进出时 torch.cuda.synchronize），记每部分的独占时间（嵌套的子部分不重复计）。
  插桩本身会多出同步，所以 detail 的合计比 plain 略长，只用来看比例。

    GPU=<卡> python runs/E33/code/profile_step.py <配置> --out <目录> --mode plain|detail|ops [--steps 60] [--skip 10]

结果写 <out>/profile_<mode>.json，并打印一张表。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from finetune import optim as O  # noqa: E402
from finetune import train as T  # noqa: E402
from finetune.parts import rloo as RL  # noqa: E402

ACC = collections.defaultdict(list)     # 部分 → 每步的独占时间
CUR = collections.defaultdict(float)    # 本步累计
STACK = []                              # [(名字, 开始时间, 子部分合计)]
STATE = {"last_refine": "cur"}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(name_fn, f):
    def g(*a, **k):
        name = name_fn(a, k) if callable(name_fn) else name_fn
        _sync()
        t0 = time.perf_counter()
        STACK.append([name, t0, 0.0])
        try:
            return f(*a, **k)
        finally:
            _sync()
            t1 = time.perf_counter()
            _, s, child = STACK.pop()
            CUR[name] += (t1 - s) - child
            if STACK:
                STACK[-1][2] += t1 - s
    return g


def is_ema(a):
    return type(a[0]).__name__ == "SimpleNamespace"


def patch():
    def feat_name(a, k):
        return "EMA 模型推理（编码与粗级）" if is_ema(a) else "当前模型前向（编码与粗级，带梯度）"

    def coarse_name(a, k):
        return "EMA 模型推理（编码与粗级）" if is_ema(a) else "当前模型前向（编码与粗级，带梯度）"

    def refine_name(a, k):
        STATE["last_refine"] = "ema" if is_ema(a) else "cur"
        return "EMA 模型推理（细化）" if is_ema(a) else "当前模型正常推理（细化）"

    def fit_name(a, k):
        return "EMA 模型推理（抽点与 RANSAC）" if STATE["last_refine"] == "ema" else "当前模型正常推理（抽点与 RANSAC）"

    def bw_name(a, k):
        return "细化三级的反向" if any(s[0] == "细化三级抽签（不含反向）" for s in STACK) else "主反向"

    RL.features = timed(feat_name, RL.features)
    RL.coarse = timed(coarse_name, RL.coarse)
    RL.refine = timed(refine_name, RL.refine)
    RL.fit_many = timed(fit_name, RL.fit_many)
    RL.in_frame = timed("画面内掩码", RL.in_frame)
    RL.fine_direct = timed("细化三级抽签（不含反向）", RL.fine_direct)
    RL.SlowRef.update = timed("EMA 更新", RL.SlowRef.update)
    RL.Term.__call__ = timed("粗匹配抽签、打分与其余", RL.Term.__call__)
    O.Optim.backward = timed(bw_name, O.Optim.backward)
    O.Optim.step = timed("优化器", O.Optim.step)


PROF = {"p": None}


class TimedLoader(T.Loader):
    """取数据的时间（主进程等 DataLoader 的时间），并在每步开始时结算上一步的累计。"""

    def __iter__(self):
        it = super().__iter__()
        while True:
            if CUR:
                for k, v in CUR.items():
                    ACC[k].append(v)
                CUR.clear()
            if PROF["p"] is not None:
                PROF["p"].step()
            t0 = time.perf_counter()
            b = next(it)
            CUR["取数据"] += time.perf_counter() - t0
            yield b


def smi(gpu, path):
    return subprocess.Popen(["nvidia-smi", "-i", str(gpu), "--query-gpu=utilization.gpu,memory.used",
                             "--format=csv,noheader,nounits", "-lms", "200"], stdout=open(path, "w"),
                            stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("plain", "detail", "ops"), default="plain")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--skip", type=int, default=10, help="前几步不计（热身）")
    a, rest = ap.parse_known_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("log.jsonl",):
        (out / f).unlink(missing_ok=True)
    T.Loader = TimedLoader
    if a.mode == "detail":
        patch()
    gpu = os.environ.get("GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0]
    p = smi(gpu, out / f"smi_{a.mode}.csv")
    t0 = time.time()
    if a.mode == "ops":
        from torch.profiler import ProfilerActivity, profile, schedule
        PROF["p"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            schedule=schedule(wait=a.skip, warmup=1, active=5, repeat=1))
        PROF["p"].__enter__()
    try:
        T.main([a.config, "--out", str(out), "--stop-at", str(a.steps), *rest])
    finally:
        p.terminate()
    if a.mode == "ops":
        PROF["p"].__exit__(None, None, None)
        ka = PROF["p"].key_averages()
        txt = (ka.table(sort_by="cuda_time_total", row_limit=40) + "\n\n"
               + ka.table(sort_by="self_cpu_time_total", row_limit=40))
        (out / "ops.txt").write_text(txt, encoding="utf-8")
        print(txt)
        return
    for k, v in CUR.items():
        ACC[k].append(v)
    wall = time.time() - t0
    log = [json.loads(x) for x in open(out / "log.jsonl", encoding="utf-8")]
    sec = [r["sec"] for r in log[a.skip:]]
    res = {"mode": a.mode, "steps": a.steps, "skip": a.skip, "wall_s": round(wall, 1),
           "step_sec_mean": round(float(np.mean(sec)), 4), "step_sec_median": round(float(np.median(sec)), 4),
           "mem_gb": log[0].get("mem_gb"), "parts": {}}
    data = ACC.get("取数据", [])
    n = len(data)
    for k, v in ACC.items():
        v = v[a.skip:] if len(v) == n else v
        res["parts"][k] = round(float(np.sum(v)) / max(1, len(data) - a.skip), 4)
    try:
        sm = np.loadtxt(out / f"smi_{a.mode}.csv", delimiter=",", ndmin=2)
        busy = np.flatnonzero(sm[:, 0] >= 30)       # 训练段：第一个到最后一个利用率 ≥ 30% 的采样（去掉加载模型与收尾）
        if len(busy):
            res["gpu_util_mean"] = round(float(sm[busy[0]:busy[-1] + 1, 0].mean()), 1)
            res["gpu_mem_mb_max"] = int(sm[:, 1].max())
    except Exception as e:      # noqa: BLE001
        res["smi_error"] = str(e)
    (out / f"profile_{a.mode}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
