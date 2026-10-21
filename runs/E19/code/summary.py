"""E19 汇总：各方法逐存档的 Val AUC@5 与误差中位数、峰值存档及其置信区间、新形式与本批 ripe 复跑的配对自助差，
E15 loo_t03 的原值，以及训练日志里概率形状与离参照距离的窗口均值。
用法（在仓库根目录、wb 环境）：python runs/E19/code/summary.py
"""
import json
from pathlib import Path

import numpy as np

from workbench.protocol import auc

E = Path(__file__).resolve().parents[1]
E15 = E.parent / "E15"
METHODS = ["ripe", "clip", "gauss4", "gauss8", "gauss16"]
AT = [1, 500, 1000, 2000, 3000, 3900]          # 从该步起 100 步的窗口
KEYS = ["H", "pmode", "rew", "in_tau", "d_ref"]


def sweep(d):
    m = json.loads((d / "S" / "metrics.json").read_text())["methods"]
    return {int(k[4:]): v["val"] for k, v in sorted(m.items(), key=lambda kv: int(kv[0][4:])) if "val" in v}


def logs(m):
    rows = [json.loads(l) for l in (E / "ckpt" / m / "log.jsonl").read_text().splitlines() if l.strip()]
    by = {r["step"]: r for r in rows if "step" in r and "H" in r}
    return {a: {k: float(np.mean([by[s][k] for s in range(a, a + 100) if s in by and k in by[s]])) for k in KEYS}
            for a in AT}


def paired(e1, e0, n=2000, seed=0):
    """同一批验证图对上 AUC@5(e1) − AUC@5(e0) 的配对自助法 95% 区间。"""
    e1, e0 = np.asarray(e1), np.asarray(e0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return auc(e1, 5.0) - auc(e0, 5.0), np.percentile(diff, [2.5, 97.5])


peaks = {}
for m in METHODS:
    s = sweep(E / "sweep" / m)
    st = max(s, key=lambda k: s[k]["summary"]["auc@5"])
    peaks[m] = s[st]
    print(f"== {m}")
    print("  val auc@5 :", {k: round(v["summary"]["auc@5"], 3) for k, v in s.items()})
    print("  val median:", {k: round(v["summary"]["median"], 2) for k, v in s.items()})
    print(f"  peak step {st}: auc@5 {s[st]['summary']['auc@5']:.3f} ci {np.round(s[st]['summary']['auc@5_ci'], 3)}")
    for a, v in logs(m).items():
        print(f"  step {a:4d}: " + " ".join(f"{k}={v[k]:.3f}" for k in KEYS))

o = sweep(E15 / "sweep" / "loo_t03")
st = max(o, key=lambda k: o[k]["summary"]["auc@5"])
print("== E15 loo_t03 val auc@5:", {k: round(v["summary"]["auc@5"], 3) for k, v in o.items()}, "peak", st)
peaks["E15"] = o[st]

print("== 峰值存档的 AUC@5 差（减本批 ripe 复跑；减 E15 loo_t03），配对自助法 95% 区间")
for m in METHODS[1:] + ["E15"]:
    d, ci = paired(peaks[m]["errors"], peaks["ripe"]["errors"])
    print(f"  {m:8s} − ripe: {d:+.4f} ci {np.round(ci, 4)}")
for m in METHODS:
    d, ci = paired(peaks[m]["errors"], peaks["E15"]["errors"])
    print(f"  {m:8s} − E15 : {d:+.4f} ci {np.round(ci, 4)}")
