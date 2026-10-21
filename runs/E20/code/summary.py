"""E20 汇总：各方法逐存档的 Val AUC@5 与误差中位数、最好存档、与 E19 gauss8（只训粗级的对照）的配对自助差，
训练日志里粗级与各加噪级的奖励、离参照距离、概率形状的窗口均值，以及每步耗时与第 1 步的显存。
用法（在仓库根目录、wb 环境）：PYTHONPATH=. python runs/E20/code/summary.py
"""
import json
from pathlib import Path

import numpy as np

from workbench.protocol import auc

E = Path(__file__).resolve().parents[1]
E19 = E.parent / "E19"
METHODS = ["l84_k05", "l8421_k05", "l84_k025", "l84_k1", "l84_k05_n16"]
AT = [1, 500, 1000, 2000, 3000, 3900]          # 从该步起 100 步的窗口
KEYS = ["H", "pmode", "in_tau", "d_ref", "rew_8", "d_8", "rew_4", "d_4", "rew_2", "d_2", "rew_1", "d_1",
        "l_coarse", "l_fine", "gnorm"]


def sweep(d):
    m = json.loads((d / "S" / "metrics.json").read_text())["methods"]
    return {int(k[4:]): v["val"] for k, v in sorted(m.items(), key=lambda kv: int(kv[0][4:])) if "val" in v}


def logs(m):
    p = E / "ckpt" / m / "log.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "step" in r]
    by = {r["step"]: r for r in rows}
    win = {a: {k: float(np.mean([by[s][k] for s in range(a, a + 100) if s in by and k in by[s]] or [np.nan]))
               for k in KEYS} for a in AT}
    sec = [r["sec"] for r in rows if r["step"] > 10]
    return win, float(np.median(sec)), rows[0].get("mem_gb"), len(rows), \
        sum(1 for r in rows if "scale" in r and r.get("gnorm") is not None and not np.isfinite(r["gnorm"]))


def paired(e1, e0, n=2000, seed=0):
    """同一批验证图对上 AUC@5(e1) − AUC@5(e0) 的配对自助法 95% 区间。"""
    fix = lambda e: np.array([np.inf if x is None else x for x in e], float)   # 没有输出的图对：误差记为无穷
    e1, e0 = fix(e1), fix(e0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return auc(e1, 5.0) - auc(e0, 5.0), np.percentile(diff, [2.5, 97.5])


o = sweep(E19 / "sweep" / "gauss8")
st = max(o, key=lambda k: o[k]["summary"]["auc@5"])
ref = o[st]
print("== E19 gauss8 val auc@5:", {k: round(v["summary"]["auc@5"], 3) for k, v in o.items()}, "peak", st,
      "median", round(ref["summary"]["median"], 2))
for m in METHODS:
    if not (E / "sweep" / m / "S" / "metrics.json").exists():
        print(f"== {m}: 没有评测结果")
        continue
    s = sweep(E / "sweep" / m)
    st = max(s, key=lambda k: s[k]["summary"]["auc@5"])
    print(f"== {m}")
    print("  val auc@5 :", {k: round(v["summary"]["auc@5"], 3) for k, v in s.items()})
    print("  val median:", {k: v["summary"]["median"] and round(v["summary"]["median"], 1) for k, v in s.items()})
    print("  val fail_rate:", {k: round(v["summary"]["fail_rate"], 3) for k, v in s.items()})
    d, ci = paired(s[st]["errors"], ref["errors"])
    print(f"  peak step {st}: auc@5 {s[st]['summary']['auc@5']:.3f}  − E19 gauss8: {d:+.4f} ci {np.round(ci, 4)}")
    if (E / "ckpt" / m / "log.jsonl").exists():
        win, sec, mem, n, bad = logs(m)
        print(f"  steps {n}  sec/step median {sec:.2f}  mem_gb(step 1) {mem}  非有限梯度步 {bad}")
        for a, v in win.items():
            print(f"  step {a:4d}: " + " ".join(f"{k}={v[k]:.3f}" for k in KEYS if np.isfinite(v[k])))
