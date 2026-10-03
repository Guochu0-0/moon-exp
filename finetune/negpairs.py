"""负样本对上的恒等匹配诊断（#48 建议的诊断 ⓪，#49 也用它逐 ckpt 监控）。

负样本对 = 光学取自 pair p，SAR 取自另一个 ROI 的 pair（按 p 播种，各权重用同一组）。同一批 p 也跑正样本对作参照。
每对：推理同款前向（无梯度）→ 细级匹配 → 仿射 RANSAC（3 px）。记录匹配数、内点数、恒等匹配数（|k1 − k0| < 3 原网格 px）、
仿射相对 [I|0] 的角点平均位移、粗级 conf[i,i] 均值。

    python -m finetune.negpairs configs/baselines/anymatch_loftr.json --split val --n 200 \
        --weights zs=<ckpt> s2=<ckpt> ... --out <dir>

产物 <out>/<name>.jsonl（逐对）与 <out>/summary.json（各权重 × 正/负的中位数与均值）。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import PairSet, neg_partner
from .models.loftr import Model
from .parts.cexp import diag_mean, pair_stats

KEYS = ("n_match", "n_inl", "n_ident", "dist_I", "diag")


def summarize(rows):
    out = {}
    for kind in ("pos", "neg"):
        xs = [r for r in rows if r["kind"] == kind]
        d = {"n": len(xs)}
        for k in KEYS:
            v = np.array([r[k] for r in xs if r[k] is not None], float)
            d[k] = {"median": round(float(np.median(v)), 4), "mean": round(float(v.mean()), 4)} if len(v) else None
        d["ransac_fail"] = sum(r["dist_I"] is None for r in xs)
        out[kind] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--weights", nargs="+", required=True, help="name=ckpt；ckpt 写 default 表示配置里的底座权重")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for spec in args.weights:
        name, ck = spec.split("=", 1)
        base = Model({"model": {"config": args.config, "init": "" if ck == "default" else ck}}, args.weights_root,
                     device=args.device)
        ck, s = base.weights, base.s
        ds = PairSet(args.data, args.split, base.resize, **base.input)
        idx = np.linspace(0, len(ds.pairs) - 1, args.n).round().astype(int)
        rows = []
        for i in idx:
            p = ds.pairs[i]
            q = neg_partner(ds.pairs, p, np.random.default_rng([args.seed, int(i)]))
            it = ds[int(i)]
            sar_neg = torch.from_numpy(base.resize(ds.map_sar(ds.data.sar(args.split, q))).astype(np.float32))[None]
            i0 = torch.stack([it["image0"], it["image0"]]).to(args.device)
            i1 = torch.stack([it["image1"], sar_neg]).to(args.device)
            with torch.no_grad():
                data = base.forward(i0, i1)
            b = data["b_ids"].cpu().numpy()
            k0, k1 = data["mkpts0_f"].cpu().numpy().astype(np.float64), data["mkpts1_f"].cpu().numpy().astype(np.float64)
            dg = diag_mean(data["conf_matrix"])
            for bb, kind, sp in ((0, "pos", p), (1, "neg", q)):
                A, _, st = pair_stats(k0[b == bb], k1[b == bb], s)
                rows.append({"opt": p, "sar": sp, "kind": kind, **st, "diag": round(float(dg[bb]), 6),
                             "A": None if A is None else np.round(A, 5).tolist()})
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summ[name] = {"weights": ck, **summarize(rows)}
        (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
        sm = summ[name]
        print(name, " | ".join(f"{k}: " + " ".join(f"{m}={sm[k][m]['median'] if sm[k][m] else None}" for m in KEYS)
                               + f" fail={sm[k]['ransac_fail']}" for k in ("pos", "neg")), flush=True)
        del base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
