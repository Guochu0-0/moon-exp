"""难版评测的已知几何扰动（「在 SAR 侧叠加已知平移 / 旋转 / 缩放的难版 Val/Test 上重测主要方法」#138）。

只扰动 SAR 一侧：每对一个已知仿射 T（原网格角点约定 2×3，与 finetune/geom.py 同），SAR 影像按 T warp（图外补 0），
真值随之精确变为 T∘A，检查点 SAR 一侧变为 T(q)。各轴单独扰动，三档：

    平移  shift12 / shift64 / shift128   方向随机，幅度固定（原网格 px）
    旋转  rot3 / rot10 / rot20           绕 patch 中心，正负随机（度）
    缩放  scale3 / scale10 / scale20     绕 patch 中心，各向同性，放大或缩小随机（1 ± 百分比）

随机量只由 (轴, split, pair) 决定，同一轴的三档共用一个方向 / 符号，只是幅度不同。
T 由 write 一次写成 runs/<id>/perturb/<变体>/<split>.json（进 git），推理（baselines.match --perturb）和工作台都只读这份文件。

    python -m moonlib.perturb runs/E7 --data $MOON_DATA

只依赖 numpy 与标准库，兼容 py3.8。
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np

HW = 512
VARIANTS = {f"{axis}{lvl}": (axis, lvl) for axis, lvls in
            (("shift", (12, 64, 128)), ("rot", (3, 10, 20)), ("scale", (3, 10, 20))) for lvl in lvls}


def sample(variant: str, split: str, pair: str, hw: int = HW) -> np.ndarray:
    """变体 variant 下这一对的 T（角点约定）：q' = L·(q − c) + c + t，c = patch 中心。"""
    axis, lvl = VARIANTS[variant]
    rng = np.random.default_rng(zlib.crc32(f"{axis}/{split}/{pair}".encode()))
    u, sign = rng.uniform(0, 2 * np.pi), (1.0 if rng.random() < 0.5 else -1.0)
    L, t = np.eye(2), np.zeros(2)
    if axis == "shift":
        t = lvl * np.array([np.cos(u), np.sin(u)])
    elif axis == "rot":
        th = np.deg2rad(sign * lvl)
        L = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    else:
        L = (1.0 + sign * lvl / 100) * np.eye(2)
    c = np.array([hw / 2, hw / 2])
    return np.c_[L, c - L @ c + t]


def path(run_dir, variant: str, split: str) -> Path:
    return Path(run_dir) / "perturb" / variant / f"{split}.json"


def load(run_dir, variant: str, split: str) -> dict:
    """{pair: 2×3 ndarray}。"""
    d = json.loads(path(run_dir, variant, split).read_text(encoding="utf-8"))
    return {p: np.asarray(T, np.float64) for p, T in d["T"].items()}


def write(run_dir, pairs: dict, variants=tuple(VARIANTS)):
    """pairs: {split: [pair, ...]}。每个变体、每个 split 写一份。"""
    for v in variants:
        for split, ps in pairs.items():
            f = path(run_dir, v, split)
            f.parent.mkdir(parents=True, exist_ok=True)
            T = {p: np.round(sample(v, split, p), 12).tolist() for p in ps}
            body = {"variant": v, "axis": VARIANTS[v][0], "level": VARIANTS[v][1], "split": split,
                    "convention": "原网格角点约定 2×3，SAR 原坐标 → 扰动后坐标；真值 A 变为 T∘A"}
            head = json.dumps(body, ensure_ascii=False, indent=1)[:-2]          # 去掉结尾的 "\n}"
            rows = ",\n".join(f"  {json.dumps(p)}: {json.dumps(t)}" for p, t in T.items())   # 一对一行
            f.write_text(f'{head},\n "T": {{\n{rows}\n }}\n}}\n', encoding="utf-8")


def main(argv=None):
    from baselines.data import Data

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="实验目录 runs/<id>")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--splits", default="val,test")
    a = ap.parse_args(argv)
    data = Data(a.data)
    write(a.run, {s: data.pairs(s) for s in a.splits.split(",")})


if __name__ == "__main__":
    main()
