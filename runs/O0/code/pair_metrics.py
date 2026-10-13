"""baselines.match 存下的原始点对 → 每对的伪仿射和若干可信度度量（#131，度量清单见调研 #133 的 K2、K8）。

    python runs/O0/code/pair_metrics.py <a.npz> [<b.npz> ...] --out runs/O0/extra/data/<名字>.jsonl

每行 {"pair", "A", "n_match", "n_inliers", "inlier_ratio", "resid_med", "conf_mean", "conf_med", "cover", "boot_spread"}：
- A、n_inliers：同 finetune.label_from_raw（RANSAC 3 px），A 为原 512 网格、角点约定；
- inlier_ratio：内点数 / 匹配数；
- resid_med：内点到伪仿射的残差中位数（px）；
- conf_mean / conf_med：全部匹配点的 certainty 均值 / 中位数；
- cover：把光学 patch 分成 16×16 格，有内点落入的格子所占比例；
- boot_spread：从已存的匹配点里不放回地随机取 --sub 个、共 --boot 次，各拟合一个仿射，512 网格四个角点
  被映到的位置到各自平均位置的平均距离（px）；任一次拟合失败记 inf。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from moonlib.ransac import fit_affine  # noqa: E402

CORNERS = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], float)
GRID = 16


def metrics(M, thr, boot, sub, rng):
    A, inl, _ = fit_affine(M, thr)
    out = {"n_match": len(M), "conf_mean": float(M[:, 4].mean()) if len(M) else math.nan,
           "conf_med": float(np.median(M[:, 4])) if len(M) else math.nan}
    if A is None:
        return dict(out, A=None, n_inliers=0, inlier_ratio=0.0, resid_med=math.inf, cover=0.0, boot_spread=math.inf)
    src, dst = M[inl, :2].astype(np.float64) + 0.5, M[inl, 2:4].astype(np.float64) + 0.5
    res = np.linalg.norm(src @ A[:, :2].T + A[:, 2] - dst, axis=1)
    cell = np.clip((src // (512 / GRID)).astype(int), 0, GRID - 1)
    cover = len(np.unique(cell[:, 0] * GRID + cell[:, 1])) / GRID ** 2
    pts = []
    for _ in range(boot):
        B, _, _ = fit_affine(M[rng.choice(len(M), size=min(sub, len(M)), replace=False)], thr)
        if B is None:
            pts = None
            break
        pts.append(CORNERS @ B.T)
    if pts is None:
        spread = math.inf
    else:
        P = np.stack(pts)                                    # boot × 4 × 2
        spread = float(np.linalg.norm(P - P.mean(0), axis=2).mean())
    n = int(inl.sum())
    return dict(out, A=np.round(A, 6).tolist(), n_inliers=n, inlier_ratio=n / len(M), resid_med=float(np.median(res)),
                cover=cover, boot_spread=spread)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--boot", type=int, default=8)
    ap.add_argument("--sub", type=int, default=1000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for p in a.npz:
            with np.load(p) as z:
                for k in z.files:
                    rng = np.random.default_rng(n)
                    d = metrics(z[k].astype(np.float32), a.ransac, a.boot, a.sub, rng)
                    f.write(json.dumps(dict(pair=k.replace("__", "/"), **d)) + "\n")
                    n += 1
                    if n % 500 == 0:
                        print(n, flush=True)
    print(f"done: {n} pairs -> {out}")


if __name__ == "__main__":
    main()
