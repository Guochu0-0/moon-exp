"""把一个实验的预测按评价协议算成指标。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from . import protocol
from .dataset import Dataset
from .records import SPLITS, Pred, Run

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def pair_errors(ds: Dataset, split: str, preds: dict[str, Pred]) -> np.ndarray:
    """按 ds.labelled(split) 的顺序给出 pair 误差；没有预测的有标注 pair 记为失败。"""
    out = []
    for p in ds.labelled(split):
        pr = preds.get(p)
        out.append(protocol.pair_error(pr.A if pr else None, *ds.checkpoints(split, p)))
    return np.array(out)


def evaluate_preds(ds: Dataset, split: str, preds: dict[str, Pred]) -> dict:
    err = pair_errors(ds, split, preds)
    labelled = set(ds.labelled(split))
    return {
        "summary": protocol.summarize(err),
        "coverage": {
            "n_pairs": len(ds.pairs(split)),
            "n_labelled": len(labelled),
            "n_pred": len(preds),
            "missing": sum(p not in preds for p in labelled),
            "unknown": sorted(p for p in preds if p not in set(ds.pairs(split)))[:20],
        },
        "errors": [None if not math.isfinite(e) else round(float(e), 4) for e in err],
    }


def evaluate_run(ds: Dataset, run: Run) -> dict:
    out = {"protocol": protocol.protocol_info(), "methods": {}}
    for m in run.methods:
        out["methods"][m] = {s: evaluate_preds(ds, s, run.preds(m, s))
                             for s in SPLITS if (run.dir / "preds" / m / f"{s}.jsonl").exists()}
    return out


def evaluate_identity(ds: Dataset) -> dict:
    return {s: evaluate_preds(ds, s, {p: Pred(p, IDENTITY) for p in ds.pairs(s)}) for s in SPLITS}


def write_metrics(run: Run, metrics: dict) -> Path:
    path = run.dir / "metrics.json"
    text = json.dumps(jsonable(metrics), ensure_ascii=False, indent=1, allow_nan=False)
    # 纯数字数组压成一行，逐对误差不至于占上千行
    text = re.sub(r"\[\s*((?:-?[\d.eE+-]+|null)(?:,\s*(?:-?[\d.eE+-]+|null))*)\s*\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def jsonable(o):
    """递归把 NaN/∞ 换成 None，保证输出是合法 JSON。"""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o
