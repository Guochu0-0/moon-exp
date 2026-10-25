"""读 E27 各方法的逐步记录（finetune/trace.py、finetune/parts/rloo_trace.py 写的 ckpt/<m>/trace/）与训练日志。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
CELL = 512 / 40          # 粗格边长（原网格 px）


def log(m, root=R):
    return [json.loads(l) for l in (root / "ckpt" / m / "log.jsonl").read_text(encoding="utf-8").splitlines()]


def series(rows, key):
    st = np.array([r["step"] for r in rows])
    v = np.array([r.get(key, np.nan) if r.get(key) is not None else np.nan for r in rows], float)
    return st, v


def rolling_median(v, w=50):
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        x = v[max(0, i - w + 1): i + 1]
        x = x[np.isfinite(x)]
        if len(x):
            out[i] = np.median(x)
    return out


def train_trace(m, keys=None, root=R):
    """各 train_*.npz 按步拼起来，去掉 batch 维（B = 1）。"""
    fs = sorted((root / "ckpt" / m / "trace").glob("train_*.npz"))
    parts = [np.load(f) for f in fs]
    keys = keys or [k for k in parts[0].files]
    out = {}
    for k in keys:
        xs = [p[k] for p in parts if k in p.files]
        if xs:
            x = np.concatenate(xs)
            out[k] = x[:, 0] if k != "step" and x.ndim > 1 and x.shape[1] == 1 else x
    return out


def obs_steps(m, root=R):
    return sorted(int(f.stem[4:]) for f in (root / "ckpt" / m / "trace").glob("obs_*.npz"))


def obs(m, step, root=R):
    return np.load(root / "ckpt" / m / "trace" / f"obs_{step:05d}.npz")


def cell_centers():
    j = (np.arange(40) + 0.5) * CELL
    return np.meshgrid(j, j)                       # (x, y)，各 40×40


def anchor_xy(k):
    k = np.asarray(k).astype(int)
    return (k % 64 + 0.5) * 8, (k // 64 + 0.5) * 8


def first(st, v, cond):
    i = np.flatnonzero(cond(v))
    return int(st[i[0]]) if len(i) else None
