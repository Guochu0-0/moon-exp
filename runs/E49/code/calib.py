"""E49 只用（#261）：从定系数的 100 次更新（code/calib.sh 的 ckpt/_calib/log.jsonl）取强、弱两档的新项系数。
那 100 次更新新项系数 190 真正施加；sim_ratio_avg = 190 × 新项梯度范数 ÷ 现有项梯度范数，两个范数都在两项共用的参数上、
先把各卡梯度取平均（与真正更新参数时相同）再算，4 张卡记的是同一个数。r = sim_ratio_avg 在 100 次更新上的中位数；
强档系数 = 190 / r（实际比例约 1），弱档 = 0.3 × 强档，各取两位有效数字。另记 r 的四分位、按 25 次更新分段的中位数，
以及 E44 口径（各卡比例的平均 sim_ratio、各卡范数的平均）供对照。写 extra/data/calib.json。

    python runs/E49/code/calib.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
W = 190.0


def stat(v):
    return {"n": int(len(v)), "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)),
            "q75": float(np.quantile(v, 0.75))}


def main():
    rows = [json.loads(l) for l in (R / "ckpt" / "_calib" / "log.jsonl").read_text(encoding="utf-8").splitlines() if l]
    out = {"n_updates": len(rows), "sim_w_calib": W}
    for k in ("sim_ratio_avg", "main_gn_avg", "sim_gn_avg", "sim_ratio", "main_gn", "sim_gn"):
        out[k] = stat(np.array([r[k] for r in rows if k in r], float))
    r = np.array([x["sim_ratio_avg"] for x in rows if "sim_ratio_avg" in x], float)
    out["ratio_avg_by_25"] = [float(np.median(r[i:i + 25])) for i in range(0, len(r), 25)]
    strong = W / out["sim_ratio_avg"]["median"]
    out["sim_w_strong"] = float(f"{strong:.2g}")
    out["sim_w_weak"] = float(f"{0.3 * strong:.2g}")
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    (R / "extra" / "data" / "calib.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
