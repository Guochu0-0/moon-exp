"""带噪真值下理想仿射方法的上限（#62）。

模型：每对 n 个检查点，s_i = A o_i + f(o_i) + ε_i。f 为仿射表达不了的错位场，ε 为标注误差（每轴 σx、σy，两侧合成，
独立同分布高斯）。「理想仿射」取对无噪对应（A o + f）的最小二乘仿射，它在检查点上的残差为 e = (I−H) f + ε。
全点拟合的残差 r = (I−H)(f + ε) 少了 H ε 一项，所以偏乐观；留一法多了外推方差，偏悲观。
由于 H ε 与 (I−H) ε 在高斯下独立，e 与 r + H ε'（ε' 为新抽的噪声）同分布，不需要对 f 做任何假设：

- noise_only：f = 0，e = ε（世界若是仿射，完美方法的得分）；
- ideal：e = r + H ε'（含错位场的上限）；
- fit / loo：现有两个有偏估计，作对照。

另报每轴的矩估计：E[RSS] = ‖(I−H) f‖² + (n−3) σ²，即错位场的方差份额。

    python scripts/finetune/ceiling_mc.py [--data $MOON_DATA] [--split val] [--sigma 1.0 1.0] [--preds name=path ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from workbench import protocol
from workbench.dataset import Dataset

AUC_T, SR_T = (3, 5, 10), (2, 3, 5, 10)


def summarize(err):
    err = np.asarray(err, float)
    return {**{f"auc@{t}": protocol.auc(err, t) for t in AUC_T},
            **{f"sr@{t}": float(np.mean(err <= t)) for t in SR_T}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--sigma", nargs=2, type=float, default=[1.0, 1.0], metavar=("SX", "SY"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000, help="按 pair 自助重采样次数（只用于 ideal 的区间）")
    ap.add_argument("--preds", nargs="*", default=[], help="name=workbench 预测 jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    sig = np.array(args.sigma)
    ds = Dataset(args.data)
    pairs = ds.labelled(args.split)

    fit, loo, noise, ideal = [], [], [], []
    rss, dof = np.zeros(2), 0
    for pair in pairs:
        o, s = ds.checkpoints(args.split, pair)
        n = len(o)
        X = np.c_[o, np.ones(n)]
        H = X @ np.linalg.pinv(X)
        r = s - H @ s                                        # 全点拟合残差（两轴共用设计矩阵）
        h = np.diag(H)
        fit.append(np.linalg.norm(r, axis=1).mean())
        loo.append(np.linalg.norm(r / (1 - h)[:, None], axis=1).mean())   # 留一残差的闭式
        rss += (r ** 2).sum(0)
        dof += n - 3
        eps = rng.standard_normal((args.draws, n, 2)) * sig
        noise.append(np.linalg.norm(eps, axis=2).mean(1))
        ideal.append(np.linalg.norm(r + np.einsum("ij,djk->dik", H, eps), axis=2).mean(1))
    noise, ideal = np.array(noise), np.array(ideal)                      # (pairs, draws)

    def mc(E):
        per = [summarize(E[:, d]) for d in range(E.shape[1])]
        return {k: round(float(np.mean([p[k] for p in per])), 4) for k in per[0]}

    out = {"split": args.split, "n_pairs": len(pairs), "sigma": list(sig),
           "noise_only": mc(noise), "ideal": mc(ideal),
           "fit": {k: round(v, 4) for k, v in summarize(fit).items()},
           "loo": {k: round(v, 4) for k, v in summarize(loo).items()}}
    boots = []
    for _ in range(args.boot):
        idx = rng.integers(0, len(pairs), len(pairs))
        boots.append(summarize(ideal[idx, rng.integers(0, ideal.shape[1])]))
    out["ideal_ci95"] = {k: [round(float(np.percentile([b[k] for b in boots], q)), 4) for q in (2.5, 97.5)]
                         for k in boots[0]}
    s2 = rss / dof
    out["moment"] = {"rss_per_dof": [round(float(v), 3) for v in s2],
                     "field_var": [round(float(v), 3) for v in s2 - sig ** 2]}
    for spec in args.preds:
        name, path = spec.split("=", 1)
        P = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                P[d["pair"]] = d.get("A")
        e = [protocol.pair_error(P.get(p), *ds.checkpoints(args.split, p)) for p in pairs]
        out[name] = {k: round(v, 4) for k, v in summarize(e).items()}
    rows = ["noise_only", "ideal", "fit", "loo"] + [s.split("=", 1)[0] for s in args.preds]
    keys = list(out["ideal"])
    print(f"{args.split}  n={len(pairs)}  sigma={list(sig)}  moment={out['moment']}")
    print("| | " + " | ".join(keys) + " |")
    for k in rows:
        print(f"| {k} | " + " | ".join(f"{out[k][c]:.3f}" for c in keys) + " |")
    print("| ideal 95% | " + " | ".join(f"{a:.3f}–{b:.3f}" for a, b in out["ideal_ci95"].values()) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
