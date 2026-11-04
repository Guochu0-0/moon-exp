"""E41（#245）：定样本筛选的门槛。不属于方法本身，只为定这个常数。

zero-shot 模型（训练起点，也是 EMA 参照的起点）在训练集上的参照内点比例：用训练同一份配置（configs/_e37all.toml，
即 E37 `all`）照 finetune/train.py 建模型、训练集（同样的难版扰动）与 [rloo] 成分，固定种子 0 从训练集随机取 2000 对
（不放回），每对经成分的 ref_fit（训练中 screen 调用的同一个函数：参照模型正常推理、按可信度抽 5000 个点、RANSAC 3 px）
拟合参照仿射，内点比例 = 内点数 ÷ 5000，拟合失败记 0。门槛 = 中位数四舍五入到 0.05。

写 extra/data/threshold_pairs.csv（每对：pair、内点数、内点比例）与 extra/data/threshold.json（分位数、门槛），并打印。
在 gpfs worktree 里单卡跑：

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E41/code/threshold.py > runs/E41/ckpt/threshold.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
Y = "/remote-home/xufang/YGC"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if "GPU" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.match import seed_all  # noqa: E402
from finetune import config as C  # noqa: E402
from finetune.data import PairSet  # noqa: E402
from finetune.models import MODELS  # noqa: E402
from finetune.parts import PARTS  # noqa: E402
from workbench.launch import require_clean  # noqa: E402

N, SEED, STEP = 2000, 0, 0.05
QS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def main():
    require_clean(REPO)
    cfg = C.load(R / "configs" / "_e37all.toml")
    seed_all(cfg["run"]["seed"])
    mod = MODELS[cfg["model"]["name"]]
    os.environ.setdefault("TORCH_HOME", mod.TORCH_HOME)
    model = mod.Model(cfg, f"{Y}/weights", device="cuda")
    kw = {}
    for n in C.parts_of(cfg):
        kw.update(PARTS[n].data_kw(cfg[n]))
    ds = PairSet(f"{Y}/dataset/Moon", cfg["run"]["split"], model.resize, **model.input, seed=cfg["run"]["seed"],
                 resize_hi=getattr(model, "resize_hi", None), **kw)
    term = PARTS["rloo"].Term(cfg["rloo"], model, ds)
    idx = np.random.default_rng(SEED).choice(len(ds), N, replace=False)
    num = cfg["rloo"]["num"]
    rows, t0 = [], time.time()
    for j, i in enumerate(idx):
        b = ds[int(i)]
        A, n = term.ref_fit(b["image0"][None].cuda(), b["image1"][None].cuda())
        rows.append({"pair": b["pair"], "n_inl": n, "ratio": n / num, "fit": A is not None})
        if (j + 1) % 200 == 0:
            print(f"{j + 1}/{N} {time.time() - t0:.0f}s 中位数 {np.median([r['ratio'] for r in rows]):.3f}", flush=True)
    out = R / "extra" / "data"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "threshold_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    x = np.array([r["ratio"] for r in rows])
    med = float(np.median(x))
    res = {"n": N, "seed": SEED, "num": num, "ransac": cfg["rloo"]["ransac"],
           "fit_fail": int(sum(not r["fit"] for r in rows)), "mean": round(float(x.mean()), 4),
           "quantiles": {str(q): round(float(np.quantile(x, q)), 4) for q in QS},
           "median": round(med, 4), "threshold": round(round(med / STEP) * STEP, 2),
           "pass_frac_at_threshold": round(float((x >= round(round(med / STEP) * STEP, 2)).mean()), 4)}
    (out / "threshold.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
