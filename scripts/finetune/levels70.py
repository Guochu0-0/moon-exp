"""分级评估汇总（#70）：完整输出（粗级 + 细级）vs 只用粗级（RANSAC 3 / 8 px），各方法按 Val 选定的 ckpt。

    python scripts/finetune/levels70.py   # 先 python -m workbench eval B0c S1c Cc Qc Pc

粗级记录由 configs/baselines/anymatch_loftr_coarse.json（LoFTR 适配器 level="coarse"）匹配，
baselines.fit --ransac 3 / 8 估仿射，方法名带 _r3 / _r8 后缀。
"""
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"
ROWS = [("B0", "B0", "anymatch_loftr", "B0c", "anymatch_loftr"), ("S1", "S1", "main", "S1c", "main"),
        ("C2", "C", "C2", "Cc", "C2"), ("Q4", "Q", "Q4", "Qc", "Q4"), ("Q4s1", "Q", "Q4s1", "Qc", "Q4s1"),
        ("P8", "P", "P8", "Pc", "P8")]


def methods(run):
    return json.loads((RUNS / run / "metrics.json").read_text(encoding="utf-8"))["methods"]


def stats(m, split):
    e, s = np.array(m[split]["errors"], float), m[split]["summary"]
    return [s["auc@3"], s["auc@5"], s["auc@10"], *(np.mean(e < t) for t in (3, 5, 10)), np.median(e)]


def main():
    for split in ("val", "test"):
        print(f"\n### {split}\n\n| 方法 | 级别 | AUC@3 | AUC@5 | AUC@10 | SR@3 | SR@5 | SR@10 | 中位误差 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for label, pr, pm, cr, cm in ROWS:
            for level, m in (("完整", methods(pr)[pm]), ("粗级 r3", methods(cr)[cm + "_r3"]),
                             ("粗级 r8", methods(cr)[cm + "_r8"])):
                v = stats(m, split)
                print(f"| {label} | {level} | " + " | ".join(f"{x:.3f}" for x in v[:6]) + f" | {v[6]:.2f} |")


if __name__ == "__main__":
    main()
