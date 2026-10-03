"""离线伪标签的整体平移修正：把每对的伪仿射平移到梯度幅值 NCC 的峰（伪标签路线第二轮，#53）。

    python -m finetune.refine_labels --labels $MOON_RESULTS/finetune/labels_p2.jsonl \
        --out $MOON_RESULTS/finetune/labels_p2n.jsonl [--feat gradncc] [--workers 16]

动机：老师的 RANSAC 仿射看不见逐对的整体偏移（#23）；测试时把 S1 的预测平移到梯度 NCC 峰，Val AUC@5 0.272 → 0.284（#27）。
这里把同一修正用在 Train 的离线标签上，给学生带进老师自己没有的信息。相似度曲面、峰值插值与
baselines/diagnose_reward.py 相同（±R px 整数平移 + 抛物线亚像素，去掉 MARGIN 边框）。
峰无效（曲面全 NaN）时保留原仿射。输出格式同 finetune/label.py，另加 A0（修正前）、ncc_dx / ncc_dy / ncc_sharp。
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from baselines.data import Data
from baselines.diagnose_reward import FEATS, peak, surface, warp_feat
from moonlib import inputs


def refine(A, opt, sar, feat="gradncc"):
    """A: 原网格 2×3（角点约定）。返回 (A', dx, dy, sharp)；A' = A 的平移减去峰位。"""
    fn = FEATS[feat]
    dx, dy, sharp = peak(surface(warp_feat(fn(opt), A, sar.shape), fn(sar), feat))
    A2 = A.copy()
    if np.isfinite(dx):
        A2[:, 2] -= [dx, dy]
    return A2, dx, dy, sharp


def one(job):
    root, d, feat = job
    if not d["keep"] or d["A"] is None:
        return d
    ds = Data(root)
    opt = inputs.get("optical", "div255")(ds.optical("train", d["pair"]))
    sar = inputs.get("sar", "p2p98")(ds.sar("train", d["pair"]))
    A = np.asarray(d["A"], float)
    A2, dx, dy, sharp = refine(A, opt, sar, feat)
    f = lambda v: None if not np.isfinite(v) else round(float(v), 4)
    return {**d, "A": A2.round(6).tolist(), "A0": d["A"], "ncc_dx": f(dx), "ncc_dy": f(dy), "ncc_sharp": f(sharp)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--feat", default="gradncc", choices=tuple(FEATS))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    with Pool(args.workers) as pool:
        recs = pool.map(one, [(args.data, d, args.feat) for d in rows], chunksize=8)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    d = np.array([[r["ncc_dx"], r["ncc_dy"]] for r in recs if r.get("ncc_dx") is not None])
    print(json.dumps({"n": len(recs), "refined": len(d), "dx_median": float(np.median(d[:, 0])),
                      "dx_std": float(d[:, 0].std()), "dy_std": float(d[:, 1].std()),
                      "abs_shift_median": float(np.median(np.hypot(*d.T)))}, indent=1))


if __name__ == "__main__":
    main()
