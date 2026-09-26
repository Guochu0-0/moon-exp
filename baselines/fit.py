"""原始点对 → 仿射，写成工作台记录（runs/<id>/preds/<method>/<split>.jsonl）。在工作台环境里跑（需要 cv2）。

    python -m baselines.fit $MOON_RESULTS/baselines/loftr --split val --run runs/B0 [--ransac 3] [--name loftr]

口径（「定义评价协议与指标」#2）：cv2.estimateAffine2D，RANSAC 阈值 3 px，confidence 0.99999，每个 pair 前重置种子；
点对少于 3 或内点少于 3 记失败。有标注但没跑到（或匹配阶段报错）的 pair 也写一行失败。

坐标：matcher 输出「整数 = 像素中心」，而标注（workbench 的检查点）是 ArcGIS 角点原点，像素中心在 c + 0.5，
所以两侧都 +0.5 再估计，得到的 A 直接和检查点同一约定。点对视图存的也是 +0.5 之后的坐标。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from workbench.dataset import Dataset
from workbench.records import PredWriter

CONFIDENCE = 0.99999
MAX_ITERS = 10000
SEED = 0
MATCHES_CAP = 100   # 每个 pair 在记录里最多存多少对内点（按置信度），只供页面画连线


def fit_affine(M: np.ndarray, thr: float):
    """M: N×5 (x0, y0, x1, y1, conf)，中心约定。返回 (A 或 None, 内点掩码 或 None, 失败原因 或 None)。"""
    if len(M) < 3:
        return None, None, "few_matches"
    src = M[:, :2].astype(np.float64) + 0.5
    dst = M[:, 2:4].astype(np.float64) + 0.5
    cv2.setRNGSeed(SEED)
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                  maxIters=MAX_ITERS, confidence=CONFIDENCE)
    if A is None or inl is None or int(inl.sum()) < 3:
        return None, None, "few_inliers"
    return A, inl.ravel().astype(bool), None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="baselines.match 的 <out>/<method> 目录")
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--run", required=True, help="实验目录，例如 runs/B0")
    ap.add_argument("--name", help="方法名，默认取 raw 目录名")
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    name = args.name or raw.name
    log = {}
    for line in (raw / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            log[d["pair"]] = d
    ds = Dataset(args.data)
    counts = {}
    with np.load(raw / f"{args.split}.npz") as z, PredWriter(args.run, name, args.split) as w:
        for pair in ds.labelled(args.split):
            rec = log.get(pair)
            k = pair.replace("/", "__")
            if rec is None:
                A, fail, extra, M, inl = None, "not_run", {}, None, None
            elif "error" in rec:
                A, fail, extra, M, inl = None, "error: " + rec["error"], {"sec": rec.get("sec")}, None, None
            else:
                M = z[k] if k in z.files else np.zeros((0, 5), np.float32)
                A, inl, fail = fit_affine(M, args.ransac)
                extra = {"n_matches": len(M), "sec": rec.get("sec")}
                if inl is not None:
                    extra["n_inliers"] = int(inl.sum())
            matches = None
            if inl is not None:
                I = M[inl]
                I = I[np.argsort(-I[:, 4], kind="stable")[:MATCHES_CAP]]
                matches = I[:, :4].astype(np.float64) + 0.5
            w.write(pair, A, fail=fail, matches=matches, **extra)
            reason = fail.split(":")[0] if fail else "ok"
            counts[reason] = counts.get(reason, 0) + 1
    print(f"[{name}/{args.split}] ransac={args.ransac}", counts)


if __name__ == "__main__":
    main()
