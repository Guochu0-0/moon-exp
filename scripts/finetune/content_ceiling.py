"""内容最优对齐的指标（第二轮，#58 #59）：把真值仿射（检查点最小二乘）平移到它自己的相似度峰，按标注评测。

只靠影像内容的信号（NCC 类 reward、NCC 修正标签）最多把预测推到「内容峰」。内容峰与标注有约 1.3 px 的出入（#23），
所以这个分数近似「内容信号能达到的上限」，与「真值最小二乘」上限（extra/full_metrics.md）并列看。

    python scripts/finetune/content_ceiling.py <diagnose_reward 的输出目录>/pairs.jsonl [--data $MOON_DATA]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from baselines.diagnose_reward import FEATS, fit_gt
from workbench import protocol
from workbench.dataset import Dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    args = ap.parse_args(argv)
    ds = Dataset(args.data)
    recs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    errs = {"gt_fit": []} | {k: [] for k in FEATS}
    for r in recs:
        o, s = ds.checkpoints("val", r["pair"])
        Ag = fit_gt(o, s)
        errs["gt_fit"].append(protocol.pair_error(Ag, o, s))
        for k in FEATS:
            dx, dy = r.get(f"{k}_gt_dx"), r.get(f"{k}_gt_dy")
            A2 = Ag.copy()
            if dx is not None and np.isfinite(dx):
                A2[:, 2] -= [dx, dy]
            errs[k].append(protocol.pair_error(A2, o, s))
    out = {}
    for k, e in errs.items():
        e = np.asarray(e, float)
        out[k] = {**{f"auc@{t}": round(protocol.auc(e, t), 4) for t in (3, 5, 10)},
                  **{f"sr@{t}": round(float(np.mean(e <= t)), 4) for t in (2, 3, 5)},
                  "median": round(float(np.median(e)), 3)}
    print(json.dumps({"n": len(recs), **out}, indent=1))


if __name__ == "__main__":
    main()
