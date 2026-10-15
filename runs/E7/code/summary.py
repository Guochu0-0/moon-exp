"""E7 汇总（#138）：读 runs/E7/metrics.json，出各方法 × 扰动档位的表和图。

    /opt/envs/wb/bin/python runs/E7/code/summary.py

产物（runs/E7/extra/）：
  data/summary.csv     方法、变体、split、n、剔除 pair / 检查点数、AUC@3/5/10、SR@3/5/10、中位误差、失败率
  auc5_vs_level.png    三个轴各一栏，横轴档位（0 = 原始），纵轴 AUC@5；上排 Val、下排 Test
  sr10_vs_level.png    同上，纵轴 SR@10（粗配准是否成功）
并在标准输出打印 Markdown 表（Val / Test 的 AUC@5 与 SR@10）和每档的方法间极差。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from moonlib.perturb import VARIANTS  # noqa: E402

R = REPO / "runs" / "E7"
METHODS = ["anymatch_loftr", "P10", "Q4", "Q11p", "loftr_lr5e5", "anymatch_roma", "erase_geo__r2", "roma_vgg"]
SHORT = {"anymatch_loftr": "LoFTR zero-shot", "P10": "LoFTR 离线伪标签（P10）", "Q4": "LoFTR 在线（Q4）",
         "Q11p": "LoFTR 在线 + 细级随机对照（Q11p）", "loftr_lr5e5": "LoFTR 标注过拟合",
         "anymatch_roma": "RoMa zero-shot", "erase_geo__r2": "RoMa 离线伪标签（N）", "roma_vgg": "RoMa 标注过拟合"}
AXES = {"shift": ("平移（px）", [0, 12, 64, 128]), "rot": ("旋转（度）", [0, 3, 10, 20]), "scale": ("缩放（%）", [0, 3, 10, 20])}
KEYS = ["auc@3", "auc@5", "auc@10", "sr@3", "sr@5", "sr@10", "median", "fail_rate"]


def name(m, axis, lvl):
    return m if lvl == 0 else f"{m}__{axis}{lvl}"


def main():
    met = json.loads((R / "metrics.json").read_text(encoding="utf-8"))["methods"]
    out = R / "extra"
    (out / "data").mkdir(parents=True, exist_ok=True)
    rows = []
    for m in METHODS + ["gtfit"]:
        for v in [""] + list(VARIANTS):
            k = f"{m}__{v}" if v else m
            for s in ("val", "test"):
                if k not in met or s not in met[k]:
                    continue
                r = met[k][s]
                d = r.get("perturb", {}).get("dropped", {"pairs": 0, "points": 0})
                rows.append({"method": m, "variant": v or "orig", "split": s, "n": r["summary"]["n"],
                             "dropped_pairs": d["pairs"], "dropped_points": d["points"],
                             **{x: r["summary"][x] for x in KEYS}})
    with open(out / "data" / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    get = {(r["method"], r["variant"], r["split"]): r for r in rows}

    def val(m, axis, lvl, s, key):
        r = get.get((m, "orig" if lvl == 0 else f"{axis}{lvl}", s))
        return None if r is None or r[key] is None else r[key]

    cols = [("orig", None, 0)] + [(f"{a}{l}", a, l) for a, (_, ls) in AXES.items() for l in ls[1:]]
    for key in ("auc@5", "sr@10"):
        for s in ("val", "test"):
            print(f"\n{s} {key}\n")
            print("| 方法 | " + " | ".join(c[0] for c in cols) + " |")
            print("|---" * (len(cols) + 1) + "|")
            for m in METHODS:
                cells = [val(m, a or "shift", l, s, key) for _, a, l in cols]
                print(f"| {SHORT[m]} | " + " | ".join("—" if c is None else f"{c:.3f}" for c in cells) + " |")
            spread = []
            for _, a, l in cols:
                xs = [x for m in METHODS if m not in ("loftr_lr5e5", "roma_vgg") and (x := val(m, a or "shift", l, s, key)) is not None]
                spread.append(f"{max(xs) - min(xs):.3f}" if len(xs) > 1 else "—")
            print("| 极差（不含标注过拟合） | " + " | ".join(spread) + " |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    style = {"anymatch_loftr": ("C0", ":"), "P10": ("C0", "-"), "Q4": ("C2", "-"), "Q11p": ("C2", "--"),
             "loftr_lr5e5": ("C0", "-."), "anymatch_roma": ("C3", ":"), "erase_geo__r2": ("C3", "-"), "roma_vgg": ("C3", "-.")}
    for key, fn in (("auc@5", "auc5_vs_level.png"), ("sr@10", "sr10_vs_level.png")):
        fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
        for j, (a, (lab, ls)) in enumerate(AXES.items()):
            for i, s in enumerate(("val", "test")):
                ax = axs[i, j]
                for m in METHODS:
                    ys = [val(m, a, l, s, key) for l in ls]
                    if any(y is not None for y in ys):
                        c, st = style[m]
                        ax.plot(ls, [np.nan if y is None else y for y in ys], st, color=c, marker="o", ms=3, label=SHORT[m])
                ax.set_xticks(ls)
                ax.set_xlabel(lab)
                ax.grid(alpha=.3)
                if j == 0:
                    ax.set_ylabel(f"{'Val' if s == 'val' else 'Test'} {key.upper()}")
        axs[0, 2].legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out / fn, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
