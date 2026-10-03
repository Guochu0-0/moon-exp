"""真值上限的留一法版本（第二轮，#58 #59）。

「真值最小二乘」上限用每对的全部检查点拟合仿射（6 个参数）再在同样的点上评测，点少（多为 8 个）时偏乐观。
这里对每个检查点用其余点拟合、在它上面算误差，逐对取均值；另报只拟合平移（线性部分取单位阵之外的
全局平均不合适，所以只拟合平移 + 用其余点的仿射）的对照不做。输出与 err_decomp.py 同格式。

    python scripts/finetune/loo_ceiling.py [--data $MOON_DATA]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from baselines.diagnose_reward import fit_gt
from workbench import protocol
from workbench.dataset import Dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    args = ap.parse_args(argv)
    ds = Dataset(args.data)
    E = {"gt_fit": [], "gt_fit_loo": []}
    npts = []
    for pair in ds.labelled("val"):
        o, s = ds.checkpoints("val", pair)
        npts.append(len(o))
        E["gt_fit"].append(protocol.pair_error(fit_gt(o, s), o, s))
        d = []
        for i in range(len(o)):
            m = np.arange(len(o)) != i
            A = fit_gt(o[m], s[m])
            d.append(np.linalg.norm(o[i] @ A[:, :2].T + A[:, 2] - s[i]))
        E["gt_fit_loo"].append(float(np.mean(d)))
    out = {"n": len(npts), "n_points": {str(k): int(v) for k, v in zip(*np.unique(npts, return_counts=True))}}
    for k, e in E.items():
        e = np.asarray(e, float)
        out[k] = {**{f"auc@{t}": round(protocol.auc(e, t), 4) for t in (3, 5, 10)},
                  **{f"sr@{t}": round(float(np.mean(e <= t)), 4) for t in (2, 3, 5)},
                  "median": round(float(np.median(e)), 3)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
