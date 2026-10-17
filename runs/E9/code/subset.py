"""难版重档的分数高于原始档，是不是因为剔除了画面边缘的检查点（#157）。

难版评测剔除扰动后落到 SAR 画面外的检查点（workbench/perturb.py）。对每个模型和变体，取它原始档的预测，只在该变体留下的
检查点上按原协议算 AUC@5（「同子集的原始档」），与该变体的实测 AUC@5 并列：两者接近，说明重档的分数变化主要来自检查点子集，
而不是模型在扰动后的影像上配得更准。

    python runs/E9/code/subset.py      # 在 worktree 根目录，MOON_DATA 指向数据集；先跑 summary.py（读 extra/data/summary.csv）

写 runs/E9/extra/data/subset.csv（实验、方法、变体、split、剔除后检查点数、同子集原始档 AUC@5、实测 AUC@5、全集原始档 AUC@5）。
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from workbench.dataset import Dataset  # noqa: E402
from workbench.perturb import PerturbedView  # noqa: E402
from workbench.protocol import auc, pair_error  # noqa: E402

R = Path("runs/E9")
V = ["shift12", "shift64", "shift128", "rot3", "rot10", "rot20", "scale3", "scale10", "scale20"]
MODELS = [("E7", "anymatch_loftr"), ("E7", "P10"), ("E7", "Q4"), ("E7", "anymatch_roma"), ("E7", "erase_geo__r2"),
          ("E9", "lns_r4"), ("E9", "q4g"), ("E9", "rns_r2")]


def preds(exp, m, split):
    f = Path("runs") / exp / "preds" / m / f"{split}.jsonl"
    if not f.exists():
        return None
    return {d["pair"]: d.get("A") for d in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if d.strip()}


def main():
    ds = Dataset(os.environ["MOON_DATA"])
    got = {}
    with open(R / "extra" / "data" / "summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            got[(r["exp"], r["method"], r["level"], r["split"])] = float(r["auc@5"])
    rows = []
    for split in ("val", "test"):
        pairs = ds.labelled(split)
        for v in V:
            pv = PerturbedView(ds, R / "perturb" / v)
            sub = {p: pv.original(split, p, None)[1] for p in pairs}
            npts = sum(len(c[0]) for c in sub.values() if c is not None)
            for exp, m in MODELS:
                P = preds(exp, m, split)
                if P is None:
                    continue
                full = np.array([pair_error(P.get(p), *ds.checkpoints(split, p)) for p in pairs])
                err = np.array([pair_error(P.get(p), *sub[p]) for p in pairs if sub[p] is not None])
                rows.append([exp, m, v, split, npts, round(auc(err, 5.0), 4), got.get((exp, m, v, split)),
                             round(auc(full, 5.0), 4)])
    with open(R / "extra" / "data" / "subset.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["exp", "method", "level", "split", "n_points", "orig_on_subset_auc5", "level_auc5",
                                  "orig_full_auc5"]] + rows)
    for r in rows:
        if r[3] == "val":
            print(*r)


if __name__ == "__main__":
    main()
