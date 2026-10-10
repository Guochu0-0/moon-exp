"""SCENES 式离线伪标签：用起点模型在 Train 上推理一遍，每对估一个伪仿射（「复现 SCENES 式伪标签微调 baseline」#26）。

    python -m finetune.label configs/baselines/anymatch_loftr.json --out $MOON_RESULTS/finetune/labels_b0.jsonl

筛选照 SCENES（§4.2）：匹配数 ≥ 100 且内点 ≥ 20，否则 keep=false，训练时不监督。
每行 {"pair", "A"（原网格 2×3，角点约定，同 workbench）, "n_match", "n_inliers", "keep"}。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from moonlib.ransac import fit_affine

from .data import PairSet
from .geom import in_to_orig
from .models.loftr import Model

MIN_MATCHES, MIN_INLIERS = 100, 20


def fit_pseudo(kp0, kp1, s, thr=3.0):
    """输入网格上的一对匹配点集 → 原网格上的伪仿射。返回 (A 或 None, 内点数)。"""
    M = np.c_[in_to_orig(kp0, s), in_to_orig(kp1, s), np.ones(len(kp0))].astype(np.float32)
    A, inl, _ = fit_affine(M, thr)
    return A, int(inl.sum()) if inl is not None else 0


def load_labels(path, top=1.0):
    """label.py 的输出 → {pair: A（list）或 None}。keep=false 的记 None。
    top < 1：只留 keep 的对里内点数最多的前 top 比例（课程式取子集，#53 调研 C4），其余记 None。"""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = sorted((d for d in rows if d["keep"]), key=lambda d: -d["n_inliers"])
    ok = {d["pair"] for d in kept[: int(round(len(kept) * top))]}
    return {d["pair"]: d["A"] if d["pair"] in ok else None for d in rows}


def pair_weights(path, pairs, kind="inlier_ratio"):
    """每对的损失权重（#127，DAFormer 2111.14887 式 (4) 的伪标签质量权重，对级）：inlier_ratio = 内点数 / 匹配数，
    在 pairs 上归一化到均值 1。"""
    rows = {d["pair"]: d for d in (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                                   if line.strip())}
    if kind != "inlier_ratio":
        raise ValueError(f"未知的对权重 {kind!r}")
    w = {q: rows[q]["n_inliers"] / max(rows[q]["n_match"], 1) for q in pairs}
    m = sum(w.values()) / len(w)
    return {q: v / m for q, v in w.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--init", help="起点 ckpt，默认配置里的底座权重")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    base = Model({"model": {"config": args.config, "init": args.init or ""}}, args.weights_root, device=args.device)
    s = base.s
    ds = PairSet(args.data, "train", base.resize, limit=args.limit, **base.input)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, t0 = 0, time.time()
    with open(out, "w", encoding="utf-8") as f, torch.no_grad():
        for it, batch in enumerate(dl):
            data = base.forward(batch["image0"].to(args.device), batch["image1"].to(args.device))
            k0 = data["mkpts0_f"].cpu().numpy().astype("float64")
            k1 = data["mkpts1_f"].cpu().numpy().astype("float64")
            A, n = fit_pseudo(k0, k1, s, args.ransac)
            keep = A is not None and len(k0) >= MIN_MATCHES and n >= MIN_INLIERS
            kept += keep
            f.write(json.dumps({"pair": batch["pair"][0], "A": None if A is None else A.round(6).tolist(),
                                "n_match": len(k0), "n_inliers": n, "keep": bool(keep)}) + "\n")
            if (it + 1) % 500 == 0:
                print(f"{it + 1}/{len(ds)} kept={kept} {(time.time() - t0) / 60:.1f}min", flush=True)
    print(f"done: {len(ds)} pairs, kept {kept}")


if __name__ == "__main__":
    main()
