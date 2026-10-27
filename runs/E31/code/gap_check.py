"""E31 的只读检查（#217）：在 E26 的坍缩存档与健康存档上算训练用的同一个核对距离，看它能否把两者分开。不训练、不影响训练。

存档与载入照 E26（runs/E26/code/methods.py、probe.load）：七个坍缩运行各取第 4000 步，三个健康对照的第 4000 步，未训练的模型 zs。
数据：Val 有标注的对里均匀取 N 对；SAR 先施加一个训练同分布的扰动 T0（难版：平移半径 128 px、旋转 ±20°、缩放 ±20%，
固定种子，各存档相同），模拟训练时的已扰动 SAR。每对：
- A0：正常推理（粗级取最高分位置、细化到尺度 1，按可信度抽 5000 点、RANSAC 3 px，同训练拟合参照）；
- 核对：finetune.parts.rloo.multi_ref，再施加 2 个已知扰动（平移半径 64 px、旋转 ±10°、缩放 ±10%，以对的序号为种子），
  换算回来后两两格心平均距离的最大值 gap（px），> 16 px 记为被拦；
- err0：A0 相对标注的误差（workbench.protocol.pair_error，标注点同样经 T0 变换），> 20 px 记为这对配错。
zs 这一行同时是「对同一张图施加已知 T 后换算回来与原估计一致」的真实模型检查。

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E31/code/gap_check.py > runs/E31/ckpt/gap_check.log 2>&1 < /dev/null &
产物：runs/E31/extra/data/gap_check.csv（逐存档逐对）。
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "runs" / "E26" / "code"))
os.environ.setdefault("MOON_DATA", "/remote-home/xufang/YGC/dataset/Moon")
os.environ.setdefault("TORCH_HOME", "/remote-home/xufang/YGC/weights/torch_home")
WEIGHTS = "/remote-home/xufang/YGC/weights"

from finetune.augment import sample_T          # noqa: E402
from finetune.geom import HW                   # noqa: E402
from finetune.parts import rloo                # noqa: E402
from workbench import protocol                 # noqa: E402
import probe                                    # noqa: E402  runs/E26/code/probe.py
from methods import COLLAPSED, HEALTHY          # noqa: E402

N = 100
MP = dict(shift=64.0, rot=10.0, scale=0.1)
THR = 16.0


def main():
    torch.manual_seed(0)
    names = [f"{e.lower()}_{r}_4000" for e, r in COLLAPSED + HEALTHY] + ["zs"]
    out = R / "extra" / "data" / "gap_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        t = time.time()
        model, cfg, _ = probe.load(name, WEIGHTS)
        model.model.eval()
        ds = probe.val_set(model)
        pairs = ds.pairs[:: max(1, len(ds.pairs) // N)][:N]

        def estimate(opt, sar):
            return rloo.infer_affine(model, opt.cuda(), sar.cuda(), 5000, 3.0,
                                     torch.Generator("cuda").manual_seed(0))

        for i, pair in enumerate(pairs):
            j = ds.pairs.index(pair)
            rng = np.random.default_rng([0, j])
            T0 = sample_T(rng, 128.0, 20.0, 0.2, HW, disk=True)
            item = ds[j]
            opt = item["image0"]
            sar = rloo.perturb_sar(item["image1"], T0)
            A0 = estimate(opt, sar)
            Ts = [sample_T(np.random.default_rng([1, j, k]), MP["shift"], MP["rot"], MP["scale"], HW, disk=True)
                  for k in range(2)]
            _, gap, As = rloo.multi_ref(lambda x: estimate(opt, x), sar, A0, Ts, 40, 40)
            po, qo = probe.labels(ds.data, ds.src[pair], pair)
            qo = qo @ T0[:, :2].T + T0[:, 2]
            rows.append({"ckpt": name, "pair": pair, "gap": gap, "blocked": int(not gap <= THR),
                         "fail": int(any(A is None for A in As)), "err0": protocol.pair_error(A0, po, qo)})
        g = np.array([r["gap"] for r in rows if r["ckpt"] == name])
        print(f"{name}: gap 中位数 {np.median(g):.1f}，被拦 {np.mean(~(g <= THR)):.2f}，{time.time() - t:.0f} s", flush=True)
        del model
        torch.cuda.empty_cache()
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
