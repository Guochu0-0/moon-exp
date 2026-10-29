"""E37 只用（#236）：启动前在一对训练图上实测每次更新的耗时与显存，并核对各级细化层是否收到梯度。

单卡、不起多卡（[run] ddp 不生效）：照 runs/E33/code/grad_check.py 搭好模型、训练成分与优化器，在训练集第一对图上
连续做 --steps 次完整的更新（损失、反向、参数更新、EMA 参照走一步），每次前后 torch.cuda.synchronize 计时；
显存取 torch.cuda.max_memory_allocated。第一次更新时记下各级细化层（decoder.conv_refiner[s]）梯度的范数
（除以 GradScaler 的 scale，即真实梯度；没收到梯度记 None）。4 卡时每张卡每次更新也只算 1 对图，单卡的耗时
即 4 卡每次更新的耗时减去梯度同步。

    GPU=<卡> python runs/E37/code/probe.py runs/E37/configs/all.toml --out runs/E37/extra/probe_all.json [--steps 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from baselines.match import seed_all  # noqa: E402
from finetune import config as C  # noqa: E402
from finetune.data import PairSet  # noqa: E402
from finetune.models import MODELS  # noqa: E402
from finetune.optim import Optim  # noqa: E402
from finetune.parts import PARTS  # noqa: E402
from finetune.train import Step  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--skip", type=int, default=2, help="前几次更新不计时（cudnn 选算法等）")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
    cfg = C.load(a.config)
    o, r, mname = cfg["optim"], cfg["run"], cfg["model"]["name"]
    seed_all(r["seed"])
    mod = MODELS[mname]
    os.environ.setdefault("TORCH_HOME", mod.TORCH_HOME)
    model = mod.Model(cfg, os.environ["MOON_WEIGHTS"], device="cuda")
    names = C.parts_of(cfg)
    kw = {}
    for n in names:
        kw.update(PARTS[n].data_kw(cfg[n]))
    ds = PairSet(os.environ["MOON_DATA"], r["split"], model.resize, **model.input, seed=r["seed"],
                 resize_hi=getattr(model, "resize_hi", None), **kw)
    terms = [(n, PARTS[n].Term(cfg[n], model, ds)) for n in names if PARTS[n].Term is not None]
    optim = Optim(model.params, o, model.amp, getattr(model, "groups", None))
    refiners = model.model.decoder.conv_refiner
    torch.manual_seed(12345)
    batch = torch.utils.data.default_collate([ds[0]])
    secs, gnorm, logs = [], None, []
    torch.cuda.reset_peak_memory_stats()
    for j in range(a.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        st = Step(model, mname, batch, "cuda", False, np.random.default_rng(j), optim)
        loss = torch.zeros((), device="cuda")
        stats = {}
        for _, term in terms:
            l_, s_, _ = term(st)
            loss = loss + l_
            stats.update(s_)
        optim.backward(loss)
        if gnorm is None:
            sc = optim.factor()
            gnorm = {}
            for k, ref in refiners.items():
                gs = [p.grad for p in ref.parameters() if p.grad is not None]
                gnorm[k] = None if not gs else float(torch.cat([g.float().reshape(-1) for g in gs]).norm() / sc)
        optim.step(o["accum"])
        for _, t in terms:
            if hasattr(t, "after_update"):
                t.after_update(None)
        torch.cuda.synchronize()
        secs.append(time.perf_counter() - t0)
        logs.append({"loss": float(loss), **{k: stats[k] for k in ("d_ref", "in_frame", "l_fine") if k in stats}})
    res = {"config": a.config, "levels": cfg["rloo"]["levels"], "gpu": torch.cuda.get_device_name(0),
           "sec_per_update": float(np.mean(secs[a.skip:])), "secs": [round(s, 3) for s in secs],
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           "refiner_grad_norm": gnorm, "steps": logs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "steps"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
