"""核对 geom.py 的闭式（#195）：细化公式对比 4×10⁶ 次抽样的估计，粗级公式对比 0.05 间距格点上的离散求和
（π_j = 密度 × 格点面积，锚点面积取同一格点面积）。

    python runs/E21/code/check_geom.py

产物：runs/E21/extra/data/check_geom.json（每组几何参数下闭式与数值的值）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geom  # noqa: E402

CASES = [((3., 1.), (5., 0.), 2., 1.5), ((1., -2.), (4., 1.), 3., 3.), ((2., .5), (4., 0.), 1., 2.),
         ((6., 0.), (4., 0.), 1., 1.), ((0.5, 0.2), (8., 3.), 2., 0.5)]       # (M, R, s, δ)，T 在原点


def main():
    torch.manual_seed(0)
    dt = torch.float64
    h = 0.05
    g = torch.arange(-40, 40, h, dtype=dt)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    Pg = torch.stack([X, Y], -1).reshape(-1, 2)
    out = []
    for M, R, s, d in CASES:
        M, R = torch.tensor(M, dtype=dt), torch.tensor(R, dtype=dt)
        T = torch.zeros(2, dtype=dt)
        a = M + s * torch.randn(4_000_000, 2, dtype=dt)
        r, e = torch.exp(-0.5 * ((a - R).norm(dim=-1) / d) ** 2), (a - T).norm(dim=-1)
        gr = ((r - r.mean())[:, None] * (a - M)).mean(0) / s ** 2
        ge = ((e - e.mean())[:, None] * (a - M)).mean(0) / s ** 2
        p = torch.exp(-((Pg - M) ** 2).sum(-1) / (2 * s * s)) / (2 * math.pi * s * s) * h * h
        r, e = torch.exp(-0.5 * ((Pg - R).norm(dim=-1) / d) ** 2), (Pg - T).norm(dim=-1)
        rb, eb = (p * r).sum(), (p * e).sum()
        out.append({"M": M.tolist(), "R": R.tolist(), "s": s, "delta": d,
                    "fine_closed": float(geom.fine_a(M, T, R, s, d)), "fine_mc": float(gr @ ge),
                    "coarse_closed": float(geom.coarse_a(M, T, R, torch.tensor(s, dtype=dt), d, area=h * h)),
                    "coarse_grid": float((p * p * (r - rb) * (e - eb)).sum())})
        print(out[-1], flush=True)
    dst = HERE.parent / "extra" / "data"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "check_geom.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
