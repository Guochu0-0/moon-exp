"""预测误差拆成平移与线性部分（第二轮，#58 #59）：剩下的误差在哪一部分？

对每个有标注的 Val pair（A_p 预测，A_g 检查点最小二乘）：
- pred：原误差；
- oracle_t：保留 A_p 的线性部分，平移取给定线性部分下对检查点的最小二乘（= 只修平移能到哪）；
- oracle_L：线性部分换成 A_g 的，并保持 A_p 对 patch 中心的映射不变（= 只修旋转 / 缩放 / 剪切能到哪）；
- gt_fit：A_g 本身（上限）。

    python scripts/finetune/err_decomp.py <workbench 预测 val.jsonl> [--data $MOON_DATA] [--center 256]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from baselines.diagnose_reward import fit_gt
from workbench import protocol
from workbench.dataset import Dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--center", type=float, default=256.0, help="patch 中心（原网格 px）")
    args = ap.parse_args(argv)
    ds = Dataset(args.data)
    preds = {}
    for line in Path(args.pred).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            preds[d["pair"]] = None if d.get("A") is None else np.asarray(d["A"], float)
    c = np.array([args.center, args.center])
    E = {k: [] for k in ("pred", "oracle_t", "oracle_L", "gt_fit")}
    for pair in ds.labelled("val"):
        o, s = ds.checkpoints("val", pair)
        Ag, Ap = fit_gt(o, s), preds.get(pair)
        E["gt_fit"].append(protocol.pair_error(Ag, o, s))
        if Ap is None:
            for k in ("pred", "oracle_t", "oracle_L"):
                E[k].append(np.inf)
            continue
        E["pred"].append(protocol.pair_error(Ap, o, s))
        At = Ap.copy()
        At[:, 2] = (s - o @ Ap[:, :2].T).mean(0)
        E["oracle_t"].append(protocol.pair_error(At, o, s))
        AL = Ag.copy()
        AL[:, 2] = Ap[:, :2] @ c + Ap[:, 2] - Ag[:, :2] @ c
        E["oracle_L"].append(protocol.pair_error(AL, o, s))
    out = {}
    for k, e in E.items():
        e = np.asarray(e, float)
        out[k] = {**{f"auc@{t}": round(protocol.auc(e, t), 4) for t in (3, 5, 10)},
                  **{f"sr@{t}": round(float(np.mean(e <= t)), 4) for t in (2, 3, 5)},
                  "median": round(float(np.median(e)), 3)}
    print(json.dumps({"n": len(E["pred"]), **out}, indent=1))


if __name__ == "__main__":
    main()
