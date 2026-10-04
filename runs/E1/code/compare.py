"""过拟合后的网络离它的训练目标还有多远：每对在光学检查点上比较方法的仿射与标注自拟合仿射（gtfit）的映射位置，
取平均距离；同时列出逐 ckpt 的 Val AUC@5。结果打印并写 extra/compare.json。

    MOON_DATA=/remote-home/xufang/YGC/dataset/Moon /opt/envs/wb/bin/python runs/E1/code/compare.py
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

METHODS = ("loftr", "loftr_lr5e5", "roma", "roma_vgg")
REFS = {"B0/anymatch_loftr": "AnyMatch-LoFTR zero-shot", "B0m/anymatch_roma__minmax": "AnyMatch-RoMa zero-shot"}


def load(path):
    if not path.exists():
        return None
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {d["pair"]: d.get("A") for d in rows}


def gap(pred, ref, ds, split):
    """每对：检查点经两个仿射映射后的平均距离（px）；失败记 inf。"""
    out = []
    for pair, Ag in ref.items():
        A = pred.get(pair)
        if A is None or Ag is None:
            out.append(np.inf)
            continue
        p, _ = ds.checkpoints(split, pair)
        P = np.c_[p, np.ones(len(p))]
        out.append(float(np.linalg.norm(P @ np.asarray(A).T - P @ np.asarray(Ag).T, axis=1).mean()))
    return np.asarray(out)


def summary(g):
    return {"median": round(float(np.median(g)), 3), "le1": round(float((g <= 1).mean()), 3),
            "le2": round(float((g <= 2).mean()), 3), "le3": round(float((g <= 3).mean()), 3),
            "fail": round(float(np.isinf(g).mean()), 4)}


def main():
    ds = Dataset(os.environ["MOON_DATA"])
    res = {"gap_to_gtfit": {}, "sweep_val_auc5": {}}
    for split in ("val", "test"):
        ref = load(R / "preds" / "gtfit" / f"{split}.jsonl")
        cands = {m: R / "preds" / m / f"{split}.jsonl" for m in METHODS}
        cands.update({k: R.parent / k.split("/")[0] / "preds" / k.split("/")[1] / f"{split}.jsonl" for k in REFS})
        for m, p in cands.items():
            pred = load(p)
            if pred is not None:
                res["gap_to_gtfit"].setdefault(m, {})[split] = summary(gap(pred, ref, ds, split))
    for m in METHODS:
        f = R / "sweep" / m / "S" / "metrics.json"
        if f.exists():
            ms = json.loads(f.read_text(encoding="utf-8"))["methods"]
            res["sweep_val_auc5"][m] = {k[4:]: round(v["val"]["summary"]["auc@5"], 4)
                                        for k, v in sorted(ms.items(), key=lambda kv: int(kv[0][4:])) if "val" in v}
    (R / "extra" / "compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
