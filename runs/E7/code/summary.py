"""E7 汇总（#138）：各方法 × 扰动档位的表、「是否跟上 T」的统计和图。

    /opt/envs/wb/bin/python runs/E7/code/summary.py          # 服务器上：读 metrics.json 与 preds，写 csv、打印表
    python runs/E7/code/summary.py --plot                    # 任何装了 matplotlib 的环境：只读 csv 画图

产物（runs/E7/extra/）：
  data/summary.csv     方法、变体、split、n、剔除 pair / 检查点数、AUC@3/5/10、SR@3/5/10、中位误差、失败率
  data/follow.csv      每个方法、变体、split：扰动后的仿射是否跟上 T（patch 四角平均位移 < 5 px）。
                       follow = 离 T∘A₀ 近（A₀ 是同一方法在原始影像上的仿射），stay = 离 A₀ 近（停在原位置），other = 都不近
  auc5_vs_level.png    三个轴各一栏，横轴档位（0 = 原始），纵轴 AUC@5；上排 Val、下排 Test
  sr10_vs_level.png    同上，纵轴 SR@10
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

R = REPO / "runs" / "E7"
OUT = R / "extra"
METHODS = ["anymatch_loftr", "P10", "Q4", "Q11p", "loftr_lr5e5", "anymatch_roma", "erase_geo__r2", "roma_vgg"]
SHORT = {"anymatch_loftr": "LoFTR zero-shot", "P10": "LoFTR 离线伪标签（P10）", "Q4": "LoFTR 在线（Q4）",
         "Q11p": "LoFTR 在线 + 细级随机对照（Q11p）", "loftr_lr5e5": "LoFTR 标注过拟合",
         "anymatch_roma": "RoMa zero-shot", "erase_geo__r2": "RoMa 离线伪标签（N）", "roma_vgg": "RoMa 标注过拟合"}
AXES = {"shift": ("平移（px）", [0, 12, 64, 128]), "rot": ("旋转（度）", [0, 3, 10, 20]), "scale": ("缩放（%）", [0, 3, 10, 20])}
KEYS = ["auc@3", "auc@5", "auc@10", "sr@3", "sr@5", "sr@10", "median", "fail_rate"]
VARS = [f"{a}{l}" for a, (_, ls) in AXES.items() for l in ls[1:]]
CORNERS = np.array([[0, 0], [512, 0], [0, 512], [512, 512]], float)


def H(M):
    return np.r_[np.asarray(M, float).reshape(2, 3), [[0, 0, 1]]]


def corner_dist(A, B):
    a, b = (CORNERS @ H(M)[:2, :2].T + H(M)[:2, 2] for M in (A, B))
    return float(np.linalg.norm(a - b, axis=1).mean())


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def tables():
    from moonlib import perturb as P
    from workbench.records import load_runs

    met = json.loads((R / "metrics.json").read_text(encoding="utf-8"))["methods"]
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    rows = []
    for m in METHODS + ["gtfit"]:
        for v in ["orig"] + VARS:
            k = m if v == "orig" else f"{m}__{v}"
            for s in ("val", "test"):
                if k in met and s in met[k]:
                    r = met[k][s]
                    d = r.get("perturb", {}).get("dropped", {"pairs": 0, "points": 0})
                    rows.append({"method": m, "variant": v, "split": s, "n": r["summary"]["n"],
                                 "dropped_pairs": d["pairs"], "dropped_points": d["points"],
                                 **{x: r["summary"][x] for x in KEYS}})
    write_csv(OUT / "data" / "summary.csv", rows)

    run, fol = load_runs(REPO / "runs")["E7"], []
    for m in METHODS:
        for s in ("val", "test"):
            base = {p: x.A for p, x in run.preds(m, s).items()}
            for v in VARS:
                pr = run.preds(f"{m}__{v}", s)
                if not pr or not base:
                    continue
                T, c = P.load(R, v, s), {"follow": 0, "stay": 0, "other": 0, "fail": 0}
                for p, x in pr.items():
                    if x.A is None or base.get(p) is None:
                        c["fail"] += 1
                    elif corner_dist(x.A, (H(T[p]) @ H(base[p]))[:2]) < 5:
                        c["follow"] += 1
                    elif corner_dist(x.A, base[p]) < 5:
                        c["stay"] += 1
                    else:
                        c["other"] += 1
                n = sum(c.values())
                fol.append({"method": m, "variant": v, "split": s, "n": n, **{k: round(x / n, 4) for k, x in c.items()}})
    write_csv(OUT / "data" / "follow.csv", fol)

    get = {(r["method"], r["variant"], r["split"]): r for r in rows}
    for key in ("auc@5", "sr@10"):
        for s in ("val", "test"):
            print(f"\n{s} {key}\n\n| 方法 | 原始 | " + " | ".join(VARS) + " |\n" + "|---" * (len(VARS) + 2) + "|")
            for m in METHODS:
                cells = [get.get((m, v, s), {}).get(key) for v in ["orig"] + VARS]
                print(f"| {SHORT[m]} | " + " | ".join("—" if c is None else f"{c:.3f}" for c in cells) + " |")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    with open(OUT / "data" / "summary.csv", encoding="utf-8") as f:
        get = {(r["method"], r["variant"], r["split"]): r for r in csv.DictReader(f)}
    style = {"anymatch_loftr": ("C0", ":"), "P10": ("C0", "-"), "Q4": ("C2", "-"), "Q11p": ("C2", "--"),
             "loftr_lr5e5": ("C0", "-."), "anymatch_roma": ("C3", ":"), "erase_geo__r2": ("C3", "-"), "roma_vgg": ("C3", "-.")}
    for key, fn in (("auc@5", "auc5_vs_level.png"), ("sr@10", "sr10_vs_level.png")):
        fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharey="row")
        for j, (a, (lab, ls)) in enumerate(AXES.items()):
            for i, s in enumerate(("val", "test")):
                ax = axs[i, j]
                for m in METHODS:
                    ys = [get.get((m, "orig" if l == 0 else f"{a}{l}", s), {}).get(key) for l in ls]
                    ys = [np.nan if y in (None, "") else float(y) for y in ys]
                    if not np.isnan(ys).all():
                        c, st = style[m]
                        ax.plot(ls, ys, st, color=c, marker="o", ms=3, label=SHORT[m])
                ax.set_xticks(ls)
                ax.set_xlabel(lab)
                ax.grid(alpha=.3)
                if j == 0:
                    ax.set_ylabel(f"{'Val' if s == 'val' else 'Test'} {key.upper()}")
        axs[0, 2].legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT / fn, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    plot() if "--plot" in sys.argv else tables()
