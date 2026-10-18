"""把一个实验的预测按评价协议算成指标。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from . import perturb, protocol
from .dataset import Dataset
from .records import SPLITS, Pred, Run

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def pair_errors(ds: Dataset, split: str, preds: dict[str, Pred]) -> np.ndarray:
    """按 ds.labelled(split) 的顺序给出 pair 误差；没有预测的有标注 pair 记为失败。"""
    out = []
    for p in ds.labelled(split):
        pr = preds.get(p)
        out.append(pair_error(ds, split, p, pr.A if pr else None))
    return np.array(out)


def pair_error(ds, split: str, pair: str, A) -> float | None:
    """一个 pair 的误差；无标注为 None。扰动变体（workbench/perturb.py）先把 A 换回原 SAR 坐标系，误差按原始像素计。"""
    if hasattr(ds, "original"):
        A, cp = ds.original(split, pair, A)
    else:
        cp = ds.checkpoints(split, pair)
    return protocol.pair_error(A, *cp) if cp else None


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
    """扰动变体的方法（workbench/perturb.py）按扰动后的真值评测，另记变体与落到画面外的检查点数。"""
    out = {"protocol": protocol.protocol_info(), "methods": {}}
    for m in run.methods:
        dm = perturb.dataset_for(ds, run, m)
        out["methods"][m] = {s: evaluate_preds(dm, s, run.preds(m, s))
                             for s in SPLITS if (run.dir / "preds" / m / f"{s}.jsonl").exists()}
        if dm is not ds:
            for s, res in out["methods"][m].items():
                res["perturb"] = {"variant": perturb.variant(run, m), "outside": dm.outside(s)}
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
