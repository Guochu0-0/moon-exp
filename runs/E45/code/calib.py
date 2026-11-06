"""E45 只用（#253）：从定系数的 50 次更新（code/calib.sh 的 ckpt/_calib/log.jsonl）取新项系数。
每次更新的日志量是 4 张卡的平均：main_gn、eqv_gn 为现有项与（系数 1 的）新项在两项共用参数上的梯度范数，
eqv_coef = main_gn / eqv_gn（逐卡算后平均）。系数取 eqv_coef 在 50 次更新上的中位数，保留两位有效数字；另记两个范数的
中位数、四分位与两者中位数之比，写 extra/data/calib.json。

    python runs/E45/code/calib.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]


def main():
    rows = [json.loads(l) for l in (R / "ckpt" / "_calib" / "log.jsonl").read_text(encoding="utf-8").splitlines() if l]
    out = {"n_updates": len(rows)}
    for k in ("main_gn", "eqv_gn", "eqv_coef"):
        v = np.array([r[k] for r in rows if k in r], float)
        out[k] = {"n": int(len(v)), "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)),
                  "q75": float(np.quantile(v, 0.75))}
    out["ratio_of_medians"] = out["main_gn"]["median"] / out["eqv_gn"]["median"]
    out["eqv_w"] = float(f"{out['eqv_coef']['median']:.2g}")
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    (R / "extra" / "data" / "calib.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
