"""带噪真值下理想仿射方法的上限：错位场按高斯过程建模（#62，ceiling_mc.py 的收窄版）。

ceiling_mc.py 的 ideal（检查点上最小二乘的理想仿射）偏乐观：真实方法看不到检查点，最好也只能拿到对整片真实 warp
最优的仿射。二者之差取决于错位场 f 在 patch 内的空间相关：f 越平滑，两者越接近；f 在检查点之间越不相关，
越接近「白场」的下界。

做法：每轴 f 为零均值高斯过程，协方差 c·exp(−d²/2ℓ²)。ℓ 全局一个；幅度 c 逐对取，使该对在检查点上的
最小二乘残差平方和的期望等于观测值（扣掉标注噪声 (n−3)σ²）。ℓ 用留一残差平方和与全点残差平方和之比
（全体合计）匹配观测来定。定好之后模拟：在 32×32 网格与检查点上联合抽 f，对网格拟合整片最优仿射，
在检查点上加 ε 评测。

    python scripts/finetune/ceiling_gp.py [--data $MOON_DATA] [--split val] [--sigma 1 1] [--ells 16 32 64 128 256 512]
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
G = np.stack(np.meshgrid(np.linspace(8, 504, 32), np.linspace(8, 504, 32)), -1).reshape(-1, 2)


def summarize(err):
    err = np.asarray(err, float)
    return {**{f"auc@{t}": protocol.auc(err, t) for t in AUC_T},
            **{f"sr@{t}": float(np.mean(err <= t)) for t in SR_T}}


def hat(o):
    X = np.c_[o, np.ones(len(o))]
    return X @ np.linalg.pinv(X)


def sim_pair(o, ell, amp, sig, draws, rng):
    """返回 (draws,) 的 pair_error、以及检查点上 LS 残差平方和与留一残差平方和（每轴，draws 平均）。"""
    n = len(o)
    P = np.r_[o, G]
    d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * ell ** 2)) + 1e-6 * np.eye(len(P))
    L = np.linalg.cholesky(K)
    z = rng.standard_normal((draws, len(P), 2))
    f = np.einsum("ij,djk->dik", L, z) * np.sqrt(amp)                     # (draws, n+grid, 2)
    fo, fg = f[:, :n], f[:, n:]
    Xg = np.c_[G, np.ones(len(G))]
    B = np.einsum("ij,djk->dik", np.linalg.pinv(Xg), fg)                  # 整片最优仿射 (draws, 3, 2)
    eps = rng.standard_normal((draws, n, 2)) * sig
    e = fo - np.einsum("ij,djk->dik", np.c_[o, np.ones(n)], B) + eps
    H = hat(o)
    obs = fo + eps
    r = obs - np.einsum("ij,djk->dik", H, obs)
    h = np.diag(H)[None, :, None]
    return np.linalg.norm(e, axis=2).mean(1), (r ** 2).sum(1).mean(0), ((r / (1 - h)) ** 2).sum(1).mean(0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--sigma", nargs=2, type=float, default=[1.0, 1.0])
    ap.add_argument("--ells", nargs="+", type=float, default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    sig = np.array(args.sigma)
    ds = Dataset(args.data)
    pairs = ds.labelled(args.split)
    data = []
    obs_rss, obs_loo = np.zeros(2), np.zeros(2)
    for p in pairs:
        o, s = ds.checkpoints(args.split, p)
        H = hat(o)
        r = s - H @ s
        h = np.diag(H)[:, None]
        rss = (r ** 2).sum(0)
        obs_rss += rss
        obs_loo += ((r / (1 - h)) ** 2).sum(0)
        data.append((o, rss))
    out = {"split": args.split, "sigma": list(sig), "obs_loo_over_rss": list(np.round(obs_loo / obs_rss, 3)), "ells": {}}
    print(f"{args.split} n={len(pairs)} 观测 留一/全点 残差平方和之比 {out['obs_loo_over_rss']}")
    for ell in args.ells:
        errs, s_rss, s_loo = [], np.zeros(2), np.zeros(2)
        for o, rss in data:
            n = len(o)
            # 单位幅度下 LS 残差平方和的期望（每轴），用来把观测残差折算成逐对幅度
            _, unit_rss, _ = sim_pair(o, ell, np.ones(2), np.zeros(2), 32, rng)
            amp = np.maximum(rss - (n - 3) * sig ** 2, 0) / np.maximum(unit_rss, 1e-9)
            e, rs, lo = sim_pair(o, ell, amp, sig, args.draws, rng)
            errs.append(e)
            s_rss += rs
            s_loo += lo
        E = np.array(errs)
        per = [summarize(E[:, d]) for d in range(E.shape[1])]
        m = {k: round(float(np.mean([q[k] for q in per])), 4) for k in per[0]}
        ratio = list(np.round(s_loo / s_rss, 3))
        out["ells"][str(ell)] = {"loo_over_rss": ratio, **m}
        print(f"ell={ell:>5g} 留一/全点 {ratio}  " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
