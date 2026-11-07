"""E47 只用（#255）：票里的停止规则。在 gpfs worktree 里跑：

    /opt/envs/loftr/bin/python runs/E47/code/stopcheck.py

第 1 遍训练集的 8 个存档（第 250、500、…、2000 次更新）都评完 Val 后，逐个与 E41 main__screen0.8（门槛 0.8，gpfs 主 checkout）
同一位置的存档比较 AUC@5：差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 E41 summary.py）。
停训条件：8 个差全部 < 0，或 8 个差的平均 < −0.01。打印逐存档的表与判定，写 extra/data/stopcheck.json。
还没评完 8 个时只打印已有的，判定记 pending。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
from workbench.protocol import auc  # noqa: E402

E41 = Path("/remote-home/xufang/YGC/moon-exp/runs/E41")
STEPS = [250 * i for i in range(1, 9)]


def fix(e):
    return np.array([np.inf if x is None else x for x in e], float)


def paired(e1, e0, n=2000, seed=0):
    e1, e0 = fix(e1), fix(e0)
    idx = np.random.default_rng(seed).integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return auc(e1, 5.0) - auc(e0, 5.0), np.percentile(diff, [2.5, 97.5])


def val(d, m):
    f = d / "sweep" / m / "S" / "metrics.json"
    if not f.exists():
        return {}
    return {int(k[4:]): v["val"] for k, v in json.loads(f.read_text(encoding="utf-8"))["methods"].items()
            if k.startswith("step") and "val" in v}


def main():
    a, b = val(R, "main"), val(E41, "main__screen0.8")
    rows = []
    for s in STEPS:
        if s not in a:
            continue
        d, ci = paired(a[s]["errors"], b[s]["errors"])
        rows.append({"update": s, "e47": round(auc(fix(a[s]["errors"]), 5.0), 4),
                     "e41": round(auc(fix(b[s]["errors"]), 5.0), 4), "diff": round(float(d), 4),
                     "lo": round(float(ci[0]), 4), "hi": round(float(ci[1]), 4)})
        print(rows[-1])
    out = {"rows": rows}
    if len(rows) == len(STEPS):
        diffs = [r["diff"] for r in rows]
        out.update(mean_diff=round(float(np.mean(diffs)), 4), all_lower=all(x < 0 for x in diffs))
        out["stop"] = out["all_lower"] or out["mean_diff"] < -0.01
    else:
        out["stop"] = "pending"
    print({k: v for k, v in out.items() if k != "rows"})
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    (R / "extra" / "data" / "stopcheck.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
