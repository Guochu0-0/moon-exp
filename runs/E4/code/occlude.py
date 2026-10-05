"""遮挡敏感性：在 60 对上，把光学图 16×16 网格（每块 32 px）逐块遮住，SAR 图同时遮住这块经标注拟合仿射映射后的位置，
看估出的仿射变化多少。

    GPU=1 /opt/envs/loftr/bin/python runs/E4/code/occlude.py loftr_zs [--limit N]

- 遮挡值：该图（已按模型的输入映射到 [0,1]）的均值。SAR 侧遮的是一个平行四边形（块的四个角经仿射映射）。
- 变化量：遮挡前后两个仿射在整张 patch 上（17×17 网格点）映射结果的平均距离（common.affine_change）；遮后失败记 nan。
  仿射有两种算法，都存：delta 用 common.refit（以不遮挡的 RANSAC 仿射为起点，3 px 内迭代最小二乘；RoMa 用不抽样的稠密
  对应、按 certainty 加权），是主分析用的量；delta_ransac 是评测口径的 RANSAC 仿射，它对微小扰动本身就会跳 1–3 px。
- 噪声底：不遮挡、给两张图各加 4 次独立的微小高斯噪声（σ = 0.5/255），同样算变化量，衡量「什么都没遮时仿射本来会晃多少」。
- 60 对由 common.pick_occlusion_pairs 按 zero-shot 的整体偏移分层抽取，第一次运行时写进 extra/occlusion_pairs.json。

产物 runs/E4/raw/<模型>/occl.npz：pairs、delta / delta_ransac (P,16,16)、noise / noise_ransac (P,4)、A0、B0（refit 基准）、G (P,2,3)。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from common import MODELS, RUN, Runner, affine_change, checkpoints, fit_gt, pick_occlusion_pairs, refit

from workbench.launch import begin, end

B, NB = 32, 16
NOISE_SIGMA, NOISE_N = 0.5 / 255, 4


def selection() -> list:
    f = RUN / "extra" / "occlusion_pairs.json"
    sel = pick_occlusion_pairs()
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(sel, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return [p for g in sel["groups"].values() for p in g]


def occlude(opt, sar, G, bx, by):
    o, s = opt.copy(), sar.copy()
    o[by * B:(by + 1) * B, bx * B:(bx + 1) * B] = opt.mean()
    x0, y0, x1, y1 = bx * B, by * B, (bx + 1) * B, (by + 1) * B
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    q = corners @ G[:, :2].T + G[:, 2] - 0.5               # 角点约定 → 像素下标（像素中心为整数）
    cv2.fillPoly(s, [np.round(q * 16).astype(np.int32)], float(sar.mean()), shift=4)
    return o, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    tag = f"occlude_{a.model}"
    begin(RUN, tag, ["occlude.py", a.model] + (["--limit", str(a.limit)] if a.limit else []),
          entry="runs/E4/code/occlude.py", model=MODELS[a.model])
    status = "fail"
    try:
        out = RUN / "raw" / a.model
        out.mkdir(parents=True, exist_ok=True)
        pairs = selection()[: a.limit or None]
        r = Runner(a.model, record=False)
        P = len(pairs)
        delta, delta_r = (np.full((P, NB, NB), np.nan, np.float32) for _ in range(2))
        noise, noise_r = (np.full((P, NOISE_N), np.nan, np.float32) for _ in range(2))
        A0s, B0s, Gs = np.full((P, 2, 3), np.nan), np.full((P, 2, 3), np.nan), np.zeros((P, 2, 3))
        t0 = time.time()
        for i, p in enumerate(pairs):
            opt, sar = r.load_pair(p)
            G = fit_gt(*checkpoints(p))
            Gs[i] = G
            A0, *_, corr = r(opt, sar)
            B0 = None if A0 is None else refit(corr, A0)
            if B0 is None:
                print(f"{p}: 不遮挡也失败，跳过", flush=True)
                continue
            A0s[i], B0s[i] = A0, B0

            def change(o, s):
                A, *_, c = r(o, s)
                return affine_change(refit(c, A0), B0), affine_change(A, A0)

            rng = np.random.default_rng(i)
            for j in range(NOISE_N):
                no = np.clip(opt + rng.normal(0, NOISE_SIGMA, opt.shape), 0, 1).astype(np.float32)
                ns = np.clip(sar + rng.normal(0, NOISE_SIGMA, sar.shape), 0, 1).astype(np.float32)
                noise[i, j], noise_r[i, j] = change(no, ns)
            for by in range(NB):
                for bx in range(NB):
                    delta[i, by, bx], delta_r[i, by, bx] = change(*occlude(opt, sar, G, bx, by))
            print(f"{a.model} {i + 1}/{P} {time.time() - t0:.0f}s", flush=True)
            np.savez(out / "occl.npz", pairs=np.array(pairs), delta=delta, delta_ransac=delta_r, noise=noise,
                     noise_ransac=noise_r, A0=A0s, B0=B0s, G=Gs, done=i + 1)
        status = "ok"
    finally:
        end(RUN, tag, status)


if __name__ == "__main__":
    main()
