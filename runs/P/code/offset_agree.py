"""两个方法相对真值的逐对偏移是否同向（「【伪标签】首轮实验设计」#53，调研 #52 的 C6 前置检查，零训练成本）。

    python scripts/finetune/offset_agree.py <data 根> <preds A 目录> <preds B 目录> [--split val] [--json out.json]

preds 目录里是 <split>.jsonl（workbench 格式）。每对的偏移 = 检查点经 A 映射后减 SAR 检查点的平均向量（px）。
报告：两方法 dx、dy 的相关系数；|dx| 都 > 1 px 的对里同号的比例；两方法仿射逐元素平均后的误差（异源老师互补的上限信号）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.dataset import Dataset   # noqa: E402


def load(d, split):
    out = {}
    for line in (Path(d) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("A") is not None:
                out[r["pair"]] = np.asarray(r["A"], np.float64).reshape(2, 3)
    return out


def offset(A, o, s):
    return (o @ A[:, :2].T + A[:, 2] - s).mean(0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    ds = Dataset(args.data)
    PA, PB = load(args.a, args.split), load(args.b, args.split)
    rows = []
    for p in ds.labelled(args.split):
        if p in PA and p in PB:
            o, s = ds.checkpoints(args.split, p)
            err = lambda A: float(np.linalg.norm(o @ A[:, :2].T + A[:, 2] - s, axis=1).mean())
            Am = (PA[p] + PB[p]) / 2
            rows.append({"pair": p, "da": offset(PA[p], o, s).tolist(), "db": offset(PB[p], o, s).tolist(),
                         "ea": err(PA[p]), "eb": err(PB[p]), "em": err(Am)})
    da, db = np.array([r["da"] for r in rows]), np.array([r["db"] for r in rows])
    E = {k: np.array([r[k] for r in rows]) for k in ("ea", "eb", "em")}
    ok = np.all(np.isfinite(E["ea"])) and len(rows)
    sane = (np.abs(da).max(1) < 20) & (np.abs(db).max(1) < 20)    # 两者都没粗错的对
    both = sane & (np.abs(da[:, 0]) > 1) & (np.abs(db[:, 0]) > 1)
    res = {"n": len(rows), "n_sane": int(sane.sum()),
           "corr_dx": float(np.corrcoef(da[sane, 0], db[sane, 0])[0, 1]),
           "corr_dy": float(np.corrcoef(da[sane, 1], db[sane, 1])[0, 1]),
           "n_both_dx_gt1": int(both.sum()),
           "same_sign_dx": float((np.sign(da[both, 0]) == np.sign(db[both, 0])).mean()) if both.any() else None,
           "median_err": {k: float(np.median(v)) for k, v in E.items()},
           "sr3": {k: float((v < 3).mean()) for k, v in E.items()},
           "sr5": {k: float((v < 5).mean()) for k, v in E.items()}}
    print(json.dumps(res, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": res, "pairs": rows}, indent=1), encoding="utf-8")
    return ok


if __name__ == "__main__":
    main()
