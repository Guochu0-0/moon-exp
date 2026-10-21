"""按不加噪声时整对图是否配准成功，分组汇总 probe.py 的逐对结果（#189）。

    python runs/E18/code/split.py

分组取自 E16（runs/E16/extra/data/pairs.csv，同一模型、同 200 对、同扰动）：不抽签时标注点误差 ≤ 5 px 为成功。
每组、每级、每个 k：逐对比例（至少一次更好、全都更差）与逐对中位数（中位数差、秩相关）在组内各对上的中位数。
产物：runs/E18/extra/data/by_success.json。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
E16 = RUN.parent / "E16" / "extra" / "data" / "pairs.csv"


def main():
    ok = {r["pair"]: float(r["err0"]) <= 5 for r in csv.DictReader(open(E16, encoding="utf-8"))
          if float(r["temp"]) == 1.0}
    rows = list(csv.DictReader(open(RUN / "extra" / "data" / "pairs.csv", encoding="utf-8")))
    out = {"n_success": sum(ok.values()), "n_fail": len(ok) - sum(ok.values()), "groups": {}}
    for name, want in (("success", True), ("fail", False)):
        g = {}
        for r in rows:
            if ok[r["pair"]] != want:
                continue
            g.setdefault((r["level"], r["k"]), []).append(r)
        med = lambda rs, key: round(float(np.nanmedian([float(x[key]) for x in rs])), 4)
        out["groups"][name] = {f"{lv}/{k}": {"n_pairs": len(rs), "e0_med": med(rs, "e0_med"),
                                             "frac_better_any": med(rs, "frac_better_any"),
                                             "frac_all_worse": med(rs, "frac_all_worse"),
                                             "dmed_med": med(rs, "dmed_med"), "rho_med": med(rs, "rho_med")}
                               for (lv, k), rs in sorted(g.items(), key=lambda t: (-int(t[0][0]), float(t[0][1])))}
    (RUN / "extra" / "data" / "by_success.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
