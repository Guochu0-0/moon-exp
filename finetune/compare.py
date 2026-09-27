"""比较两份 baselines.match 原始点对（<split>.npz），核对「接入没有改变行为」。

    python -m finetune.compare $MOON_RESULTS/baselines/anymatch_loftr/val.npz $MOON_RESULTS/finetune/verify/anymatch_loftr_ft/val.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def compare(a: dict, b: dict, atol: float = 1e-3) -> dict:
    """a、b：pair 键 → N×5。返回汇总：键集合差异、点数不同的 pair、坐标/置信度最大差。"""
    common = sorted(set(a) & set(b))
    diff_n, max_xy, max_conf = [], 0.0, 0.0
    for k in common:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            diff_n.append({"pair": k, "n": [len(x), len(y)]})
            continue
        if len(x):
            max_xy = max(max_xy, float(np.abs(x[:, :4] - y[:, :4]).max()))
            max_conf = max(max_conf, float(np.abs(x[:, 4] - y[:, 4]).max()))
    return {"n_a": len(a), "n_b": len(b), "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a)),
            "n_common": len(common), "n_diff_count": len(diff_n), "diff_count": diff_n[:20],
            "max_abs_xy": max_xy, "max_abs_conf": max_conf,
            "identical": not diff_n and set(a) == set(b) and max_xy <= atol and max_conf <= atol}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args(argv)
    with np.load(args.a) as za, np.load(args.b) as zb:
        r = compare({k: za[k] for k in za.files}, {k: zb[k] for k in zb.files}, args.atol)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
