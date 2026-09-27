"""PROTOTYPE（节点页内容，#20）：把 runs/*/metrics.json + exp.toml 抽成 data.js 快照，外加两个模拟实验。

    py workbench/static/prototype-node-page/gen_data.py

不依赖数据集和 server，生成后双击 index.html 即可。一次性代码，别合进 main。
"""
from __future__ import annotations

import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from workbench import protocol  # noqa: E402


def clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 5)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def errs_out(e):
    return [None if v is None else round(v, 2) for v in e]


runs = {}
for d in sorted((REPO / "runs").iterdir()):
    if not (d / "exp.toml").exists():
        continue
    meta = tomllib.loads((d / "exp.toml").read_text(encoding="utf-8"))
    mj = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    metrics = {m: {s: {"summary": clean(v["summary"]), "errors": errs_out(v["errors"])} for s, v in sp.items()}
               for m, sp in mj["methods"].items()}
    runs[d.name] = {"id": d.name, "meta": meta, "lit": True, "mock": False, "metrics": metrics, "extras": []}

# ---- 模拟实验 ----
b0 = runs["B0"]["metrics"]
best = max(b0, key=lambda m: b0[m]["val"]["summary"]["auc@10"] or -1)
rng = np.random.default_rng(0)


def improve(e):
    out = []
    for v in e:
        if v is None:
            out.append(round(float(rng.uniform(3, 40)), 2) if rng.random() < 0.3 else None)
        else:
            out.append(round(v * float(rng.uniform(0.55, 1.1)), 2))
    return out


e1 = {}
for s in ("val", "test"):
    er = improve(b0[best][s]["errors"])
    arr = np.array([np.inf if v is None else v for v in er])
    e1[s] = {"summary": clean(protocol.summarize(arr)), "errors": er}

steps = list(range(0, 20001, 1000))
runs["E1"] = {
    "id": "E1", "lit": True, "mock": True, "extras": [{
        "name": "train_curve.json", "type": "line", "x": steps, "xlabel": "step",
        "series": {"loss": [round(1.2 * math.exp(-s / 6000) + 0.15 + 0.02 * math.sin(s / 900), 3) for s in steps],
                   "val auc@10": [round(0.2 + 0.25 * (1 - math.exp(-s / 5000)), 3) for s in steps]}}],
    "metrics": {"main": e1},
    "meta": {"id": "E1", "title": f"{best} 伪标签自训练（模拟数据）", "parent": "B0", "init": f"B0/{best}",
             "date": "2026-10-03", "commit": "a1b2c3d", "verdict_tag": "",
             "hypothesis": f"用 {best} 在无标注训练集上的高置信匹配当伪标签微调，能修掉一部分失败 pair。\n",
             "change": "伪标签：RANSAC 内点 ≥ 30 且残差 < 2 px 的 pair；lr 1e-5，20k step。\n",
             "verdict": "", "next": ""},
}
runs["E2"] = {
    "id": "E2", "lit": False, "mock": True, "extras": [], "metrics": {},
    "meta": {"id": "E2", "title": "SAR 先去斑再匹配（想法）", "parent": "E1", "init": "",
             "hypothesis": "E1 剩下的失败 pair 多在强斑点区域；先做 Lee 滤波可能让匹配器出更多点。\n",
             "change": "", "verdict": "", "next": ""},
}
# 真实记录补上 v2 的字段拆分：status → baseline 标记 + 判定
for r in ("B0", "B0m"):
    m = runs[r]["meta"]
    m["baseline"] = True
    m["verdict_tag"] = ""
runs["E1"]["meta"]["baseline"] = False
runs["E2"]["meta"]["baseline"] = False

data = {"protocol": clean(protocol.protocol_info()), "runs": runs, "best_b0": best}
(HERE / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
print("wrote data.js", {k: list(v["metrics"]) [:3] for k, v in runs.items()}, "best B0 =", best)
