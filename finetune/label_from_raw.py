"""baselines.match 在 Train 上的原始点对 → finetune.label 同格式的离线伪仿射（RoMa 自己当老师，#65）。

    python -m finetune.label_from_raw <out>/<method>/train.npz --out labels_roma0.jsonl

点对已在原网格（中心约定），直接仿射 RANSAC（3 px，与评测同口径）；筛选同 finetune/label.py（匹配 ≥ 100、内点 ≥ 20）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.ransac import fit_affine

from .label import MIN_INLIERS, MIN_MATCHES


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ransac", type=float, default=3.0)
    args = ap.parse_args(argv)
    kept = total = 0
    with np.load(args.npz) as z, open(args.out, "w", encoding="utf-8") as f:
        for k in z.files:
            total += 1
            M = z[k].astype(np.float32)
            A, inl, _ = fit_affine(M, args.ransac)
            n = int(inl.sum()) if inl is not None else 0
            keep = A is not None and len(M) >= MIN_MATCHES and n >= MIN_INLIERS
            kept += keep
            pair = k.replace("__", "/")
            f.write(json.dumps({"pair": pair, "A": None if A is None else np.round(A, 6).tolist(),
                                "n_match": len(M), "n_inliers": n, "keep": bool(keep)}) + "\n")
    print(f"done: {total} pairs, kept {kept}")


if __name__ == "__main__":
    main()
