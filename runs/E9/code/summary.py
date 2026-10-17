"""E9 的汇总（#157）：各方法 × 档位的 Val / Test AUC@5、AUC@10、SR@10（与 E7 并列），以及「按原始 + 难版 9 档平均选模」会选到哪一步。

    python runs/E9/code/summary.py [方法 …]      # 在 worktree 根目录，先 workbench eval E9 与各 sweep/<m>/H

读 runs/E9/metrics.json、runs/E7/metrics.json、runs/E9/sweep/<m>/{S,H}/metrics.json、peak.json；
写 runs/E9/extra/data/summary.csv（每行：实验、方法、档位、split、指标）与 runs/E9/extra/data/select.csv（每行：方法、step、
原始 Val AUC@5、9 档平均、10 档平均），并打印两张表。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

R = Path("runs/E9")
V = ["", "shift12", "shift64", "shift128", "rot3", "rot10", "rot20", "scale3", "scale10", "scale20"]
KEYS = ("auc@5", "auc@10", "sr@10")
E7_METHODS = ["anymatch_loftr", "P10", "Q4", "anymatch_roma", "erase_geo__r2"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))["methods"]


def summary_rows(final):
    rows = []
    for exp, M, ms in (("E7", load("runs/E7/metrics.json"), E7_METHODS), ("E9", load(R / "metrics.json"), final)):
        for m in ms:
            for v in V:
                k = m + ("__" + v if v else "")
                for sp in ("val", "test"):
                    s = M.get(k, {}).get(sp, {}).get("summary")
                    if s:
                        rows.append([exp, m, v or "orig", sp] + [round(s[x], 4) for x in KEYS])
    return rows


def select_rows(m):
    S, H = load(R / "sweep" / m / "S" / "metrics.json"), load(R / "sweep" / m / "H" / "metrics.json")
    out = []
    for k, v in S.items():
        if not k.startswith("step") or int(k[4:]) == 0 or "val" not in v:
            continue
        hard = [H.get(f"{k}__{x}", {}).get("val", {}).get("summary", {}).get("auc@5") for x in V[1:]]
        if None in hard:
            continue
        o = v["val"]["summary"]["auc@5"]
        out.append([m, int(k[4:]), round(o, 4), round(sum(hard) / 9, 4), round((o + sum(hard)) / 10, 4)])
    return sorted(out, key=lambda r: r[1])


def main():
    final = sys.argv[1:] or sorted(p.name for p in (R / "sweep").iterdir() if (p / "H" / "metrics.json").exists())
    d = R / "extra" / "data"
    d.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(final)
    with open(d / "summary.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["exp", "method", "level", "split", *KEYS]] + rows)
    print("Val / Test AUC@5：" + " ".join(v or "orig" for v in V))
    for sp in ("val", "test"):
        for exp, m in dict.fromkeys((r[0], r[1]) for r in rows):
            a = {r[2]: r[4] for r in rows if r[:2] == [exp, m] and r[3] == sp}
            print(sp, exp, m, " ".join(f"{a[v or 'orig']:.3f}" if (v or "orig") in a else "  -  " for v in V))
    sel = []
    for m in final:
        if (R / "sweep" / m / "H" / "metrics.json").exists():
            sel += select_rows(m)
    with open(d / "select.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["method", "step", "orig_val_auc5", "hard9_mean", "all10_mean"]] + sel)
    print("\n选模：method step 原始 9档平均 10档平均")
    for m in dict.fromkeys(r[0] for r in sel):
        rs = [r for r in sel if r[0] == m]
        peak = json.loads((R / "sweep" / m / "peak.json").read_text(encoding="utf-8"))["step"]
        for r in rs:
            print(*r, "← 原始峰值" if r[1] == peak else "", "← 10 档平均峰值" if r == max(rs, key=lambda x: x[4]) else "")


if __name__ == "__main__":
    main()
