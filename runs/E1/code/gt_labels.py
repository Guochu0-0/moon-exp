"""每对用全部标注点最小二乘拟合一个仿射（光学 → SAR，与 workbench 的 A 同一约定）：
1. 写成离线伪标签文件 extra/labels_gt.jsonl（finetune.label 的格式），供在 Val+Test 上过拟合训练；
2. 写成参照方法 gtfit 的 Val、Test 预测：网络训到极致，经 RANSAC 估出的仿射也只能逼近它。

    MOON_DATA=/remote-home/xufang/YGC/dataset/Moon /opt/envs/wb/bin/python runs/E1/code/gt_labels.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parents[1]))
from workbench.dataset import Dataset  # noqa: E402
from workbench.records import PredWriter  # noqa: E402


def lsq_affine(p, q):
    """p, q: N×2。返回 2×3（q ≈ A·[p;1]），点少于 3 个或退化时为 None。"""
    X = np.c_[p, np.ones(len(p))]
    if len(p) < 3 or np.linalg.matrix_rank(X) < 3:
        return None
    M, *_ = np.linalg.lstsq(X, q, rcond=None)
    return M.T


def main():
    ds = Dataset(os.environ["MOON_DATA"])
    (R / "extra").mkdir(exist_ok=True)
    seen = set()
    with open(R / "extra" / "labels_gt.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for split in ("val", "test"):
            n_ok = 0
            with PredWriter(R, "gtfit", split) as w:
                for pair in ds.labelled(split):
                    assert pair not in seen, f"{pair} 在 Val 和 Test 中重名"
                    seen.add(pair)
                    p, q = ds.checkpoints(split, pair)
                    A = lsq_affine(p, q)
                    n_ok += A is not None
                    w.write(pair, A, fail=None if A is not None else "few_points", n_points=len(p))
                    f.write(json.dumps({"pair": pair, "split": split,
                                        "A": None if A is None else A.round(6).tolist(),
                                        "n_match": len(p), "n_inliers": len(p), "keep": A is not None}) + "\n")
            print(f"{split}: {len(ds.labelled(split))} 对有标注，{n_ok} 对拟合成功")


if __name__ == "__main__":
    main()
