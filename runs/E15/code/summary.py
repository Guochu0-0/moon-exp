"""E15 汇总：各方法逐存档的 Val AUC@5 与误差中位数、验证集最好存档的 Test，以及训练日志里概率形状与奖励的窗口均值。
用法（在实验目录的上两级，即仓库根目录）：python runs/E15/code/summary.py
"""
import json
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
METHODS = ["loo_t05", "loo_t03", "loo_t02", "none_t05", "none_t03", "none_t02"]
AT = [1, 500, 1000, 2000, 3000, 3900]          # 从该步起 100 步的窗口
KEYS = ["H", "pmode", "rew", "in_tau", "d_ref", "nonmode"]


def sweep(m):
    d = json.loads((E / "sweep" / m / "S" / "metrics.json").read_text())["methods"]
    out = {}
    for name, v in d.items():
        if name.startswith("step") and "val" in v:
            out[int(name[4:])] = v["val"]["summary"]
    return dict(sorted(out.items()))


def logs(m):
    rows = [json.loads(l) for l in (E / "ckpt" / m / "log.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "step" in r and "H" in r]
    by = {r["step"]: r for r in rows}
    res = {}
    for a in AT:
        win = [by[s] for s in range(a, a + 100) if s in by]
        res[a] = {k: float(np.mean([r[k] for r in win if k in r])) for k in KEYS}
    return res


for m in METHODS:
    s = sweep(m)
    peak = json.loads((E / "sweep" / m / "peak.json").read_text())
    print(f"== {m}")
    print("  val auc@5 :", {k: round(v["auc@5"], 3) for k, v in s.items()})
    print("  val median:", {k: round(v["median"], 1) for k, v in s.items()})
    print(f"  peak step {peak['step']}: val {peak['val']['auc@5']:.3f} ci {np.round(peak['val']['auc@5_ci'], 3)}"
          f"  test {peak['test']['auc@5']:.3f} ci {np.round(peak['test']['auc@5_ci'], 3)} median {peak['test']['median']:.2f}")
    for a, v in logs(m).items():
        print(f"  step {a:4d}: " + " ".join(f"{k}={v[k]:.3f}" for k in KEYS))
