"""细化奖励的设计依据（#197）：zero-shot AnyMatch-RoMa 在训练设置下（Train、难版扰动 + 擦除，粗级温度 0.3 抽签，
尺度 16、8、4 加 σ = 0.5 × s 的高斯，每对 8 条链，同 configs/_rl.toml），各级抽到位置离参照仿射的距离 d 的分布，
以及两种细化奖励（gauss exp(−½(d/δ)²)、cauchy 1/(1 + ½(d/δ)²)，δ = s）在单精度下：
- zero：奖励为 0 的像素比例；
- tie：8 条链的奖励完全相同（这一像素这一步没有 RLOO 信号）的像素比例；
- sep：按 d 排序后相邻两条链 d 不同、奖励却相同的比例（相邻距离分不开）。
另记各级 d 的分位数（按 d/δ 也记一份）。

    python runs/E22/code/dist.py --out runs/E22/extra/data/dist.json [--n 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from finetune import config as C                # noqa: E402
from finetune.data import PairSet               # noqa: E402
from finetune.geom import HW                    # noqa: E402
from finetune.models import roma                # noqa: E402
from finetune.parts import aug, rloo            # noqa: E402

LEVELS = (16, 8, 4)
Q = [0.5, 0.9, 0.99, 0.999, 1.0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    a = ap.parse_args(argv)
    cfg = C.load(REPO / "runs" / "E22" / "configs" / "_rl.toml")
    p = cfg["rloo"]
    torch.manual_seed(0)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    model = roma.Model(cfg, a.weights_root)
    ds = PairSet(a.data, "train", model.resize, **model.input, seed=0, **aug.data_kw(cfg["aug"]))
    ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // a.n)][: a.n]
    sig = {s: p["k"] * s / (HW / 2) for s in LEVELS}
    D = {s: [] for s in LEVELS}
    cnt = {(s, sh): np.zeros(4) for s in LEVELS for sh in ("gauss", "cauchy")}   # 像素数、zero、tie、相邻对数 / 分不开
    sep = {(s, sh): np.zeros(2) for s in LEVELS for sh in ("gauss", "cauchy")}
    n_fail = 0
    for i in range(len(ds)):
        it = ds[i]
        with torch.no_grad():
            i0, i1 = it["image0"][None].cuda(), it["image1"][None].cuda()
            f1, f2 = rloo.features(model, i0, i1)
            cls, cert = rloo.coarse(model, f1, f2)
            cls = cls.float()
            P = cls.softmax(1)
            C_, h, w = cls.shape[1:]
            q = (cls[0] / p["temp"]).softmax(0)
            k = torch.multinomial(q.reshape(C_, -1).T, p["group"], replacement=True).T.reshape(-1, h, w)
            flow, ce = rloo.refine(model, f1, f2, rloo.anchor_flow(P, P.argmax(1)), cert.float())
            A_ref = rloo.fit(flow[0], ce[0], p["num"], p["ransac"])[0]
            if A_ref is None:
                n_fail += 1
                continue
            rec, _ = rloo.rollout_noisy(model, f1, f2, P, cert.float(), k, sig, p["chunk"])
            for s in LEVELS:
                for sh in ("gauss", "cauchy"):
                    r, d = rloo.level_reward(rec[s][2], A_ref, s, {**p, "fine_shape": sh})   # 单精度
                    c = cnt[(s, sh)]
                    c += [r[0].numel(), float((r == 0).float().sum()) / len(r), float((r.std(0) == 0).float().sum()), 0]
                    o = d.argsort(0)
                    ds_, rs = d.gather(0, o), r.gather(0, o)
                    diff = ds_[1:] > ds_[:-1]
                    sep[(s, sh)] += [float(diff.sum()), float((diff & (rs[1:] == rs[:-1])).sum())]
                D[s].append(d.flatten().cpu().numpy())
        if i % 20 == 0:
            print(f"{i}/{len(ds)}", flush=True)
    out = {"n_pairs": len(ds), "n_ref_fail": n_fail, "levels": {}}
    for s in LEVELS:
        d = np.concatenate(D[s])
        lv = {"quantiles_px": {str(x): float(np.quantile(d, x)) for x in Q},
              "frac_over_px": {str(t): float((d > t * s).mean()) for t in (4, 8, 14, 20, 50)}}   # d > t·δ 的比例
        for sh in ("gauss", "cauchy"):
            c, sp = cnt[(s, sh)], sep[(s, sh)]
            lv[sh] = {"zero": c[1] / c[0], "tie": c[2] / c[0], "adjacent_unresolved": sp[1] / max(sp[0], 1)}
        out["levels"][str(s)] = lv
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
