"""全指标表（#51 #53）：Val / Test 的 AUC@3/5/10/20、SR@1/2/3/5/10、中位误差，加真值最小二乘仿射的上限行。

    python scripts/finetune/fulltab.py <val|test> <run1,run2,...>

run 名取自 runs/Q、runs/P 的方法，另有 zero-shot（B0/anymatch_loftr）、S1（S1/main）、C2（C/C2）、
「上限：真值最小二乘」（runs/Q/extra/oracle_gt_fit.json 的 gt_lsq）。
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "runs"
L = lambda r: json.loads((R / r / "metrics.json").read_text(encoding="utf-8"))["methods"]
M = {"zero-shot": L("B0")["anymatch_loftr"], "S1": L("S1")["main"], "C2": L("C")["C2"]}
M.update(L("Q"))
M.update(L("P"))
M["上限：真值最小二乘"] = json.loads((R / "Q/extra/oracle_gt_fit.json").read_text(encoding="utf-8"))["methods"]["gt_lsq"]
K = ["auc@3", "auc@5", "auc@10", "auc@20", "sr@1", "sr@2", "sr@3", "sr@5", "sr@10", "median"]

split, rows = sys.argv[1], sys.argv[2].split(",")
print(f"**{split}**\n")
print("| run | " + " | ".join(k.upper() if k != "median" else "中位误差 px" for k in K) + " |")
print("|---|" + "---|" * len(K))
for r in rows:
    s = M[r][split]["summary"]
    print(f"| {r} | " + " | ".join(f"{s[k]:.2f}" if k == "median" else f"{s[k]:.3f}" for k in K) + " |")
