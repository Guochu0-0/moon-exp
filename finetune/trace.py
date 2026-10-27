"""训练循环的逐步记录与续训（#209）。

- TraceWriter：训练成分每步放进 step.trace 的数组（{名字: ndarray}，第 0 维是 batch 里的第几对），每 every 步沿新的
  第 0 维按步叠起，写成 <out>/trace/train_<首步>-<末步>.npz（另有 step 一列）；观察集每次一个 <out>/trace/obs_<step>.npz。
- Loader：DataLoader 外包一层，记下 epoch 与 epoch 内已取的 batch 数；续训时恢复数据顺序的发生器，从同一个 batch 接着取。
- rng_state / set_rng_state：训练用到的全部随机数发生器的状态。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def stack_rows(rows: list[dict]) -> dict:
    """[{名字: 数组}] → {名字: 沿新的第 0 维叠起的数组}。某行缺的量用同形状的填充值补上：浮点为 NaN，整数为 −1。"""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = {}
    for k in keys:
        ref = np.asarray(next(r[k] for r in rows if k in r))
        fill = np.nan if np.issubdtype(ref.dtype, np.floating) else -1
        out[k] = np.stack([np.asarray(r[k]) if k in r else np.full(ref.shape, fill, ref.dtype) for r in rows])
    return out


class TraceWriter:
    def __init__(self, out, every=50):
        self.dir, self.every, self.rows = Path(out) / "trace", every, []

    def add(self, step, trace: dict):
        if trace:
            self.rows.append({"step": np.int32(step), **trace})
        if len(self.rows) >= self.every:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        a, b = int(self.rows[0]["step"]), int(self.rows[-1]["step"])
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.dir / f"train_{a:05d}-{b:05d}.npz", **stack_rows(self.rows))
        self.rows = []

    def obs(self, step, data: dict):
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.dir / f"obs_{step:05d}.npz", **data)


class Loader:
    """无限地逐个产出 batch。state() 存下当前 epoch、epoch 内已取的 batch 数和这一 epoch 开始时数据发生器的状态；
    load() 之后从同一个 batch 接着取（前面的照常取出再丢掉，所以 worker 里的随机扰动也和原来一样）。
    只支持第一个 epoch 内的状态：persistent_workers 时 worker 的种子在第一个 epoch 取定，之后的 epoch 无法单独重建。"""

    def __init__(self, dl, g: torch.Generator):
        self.dl, self.g = dl, g
        self.epoch, self.i, self.skip = 0, 0, 0
        self.g0 = g.get_state()

    def state(self) -> dict:
        return {"epoch": self.epoch, "batch": self.i, "g": self.g0}

    def load(self, s: dict):
        if s["epoch"] > 0:
            raise ValueError(f"续训只支持第一个 epoch 内的存档，这个存档在第 {s['epoch']} 个 epoch")
        self.g.set_state(s["g"])
        self.skip = s["batch"]

    def __iter__(self):
        while True:
            self.g0, self.i = self.g.get_state(), 0
            if hasattr(self.dl.sampler, "set_epoch"):    # 多卡的 DistributedSampler 按 epoch 换顺序（#226）
                self.dl.sampler.set_epoch(self.epoch)
            for b in self.dl:
                self.i += 1
                if self.skip:
                    self.skip -= 1
                    continue
                yield b
            self.epoch += 1


def rng_state(rng: np.random.Generator) -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate(), "rng": rng.bit_generator.state}


def set_rng_state(s: dict, rng: np.random.Generator):
    torch.set_rng_state(s["torch"])
    if s["cuda"]:
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    rng.bit_generator.state = s["rng"]
