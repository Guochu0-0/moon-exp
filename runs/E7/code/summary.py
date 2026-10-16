"""E7 汇总（#138）：各方法 × 扰动档位的表、「是否跟上 T」的统计和图。

    /opt/envs/wb/bin/python runs/E7/code/summary.py          # 服务器上：读 metrics.json 与 preds，写 csv、打印表
    python runs/E7/code/summary.py --plot                    # 任何装了 matplotlib 的环境：只读 csv 画图

产物（runs/E7/extra/）：
  data/summary.csv     方法、变体、split、n、剔除 pair / 检查点数、AUC@3/5/10、SR@3/5/10、中位误差、失败率
  data/follow.csv      每个方法、变体、split：扰动后的仿射是否跟上 T（patch 四角平均位移 < 5 px）。
                       follow = 离 T∘A₀ 近（A₀ 是同一方法在原始影像上的仿射），stay = 离 A₀ 近（停在原位置），other = 都不近
  data/gt_dist.csv     原始数据本身的平移 / 旋转 / 缩放分布（由 gtfit 仿射分解，--plot 时写）
  auc5_vs_level.png    总览：三个轴各一栏，横轴档位（0 = 原始），纵轴 Val AUC@5，只画 5 个方法；灰底为原数据 90% 的范围
  auc5_bars.png        原始与三个轴最重一档的 Val AUC@5 柱状图
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
SHORT = {"anymatch_loftr": "LoFTR zero-shot", "P10": "LoFTR 离线伪标签自训练", "Q4": "LoFTR 在线自监督",
         "Q11p": "LoFTR 在线自监督 + 细级随机 reward", "loftr_lr5e5": "LoFTR 标注过拟合",
         "anymatch_roma": "RoMa zero-shot", "erase_geo__r2": "RoMa 离线伪标签自训练", "roma_vgg": "RoMa 标注过拟合"}
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


# 图里只画方法本身：Q11p（随机 reward 对照）和两个标注过拟合模型只留在表里
PLOT_METHODS = ["anymatch_loftr", "P10", "Q4", "anymatch_roma", "erase_geo__r2"]
# 颜色 = 路线（zero-shot 灰、离线伪标签蓝、在线自监督绿），线型 / 柱子花纹 = 骨干（LoFTR 实线、RoMa 虚线 + 斜纹）
STYLE = {"anymatch_loftr": ("#8c8c8c", "-", "o"), "P10": ("#1f77b4", "-", "o"), "Q4": ("#2ca02c", "-", "o"),
         "anymatch_roma": ("#8c8c8c", "--", "s"), "erase_geo__r2": ("#1f77b4", "--", "s")}
BARS = [("orig", "原始"), ("shift128", "平移 128 px"), ("rot20", "旋转 20°"), ("scale20", "缩放 20%")]


def gt_dist():
    """原始数据本身的几何差异：用标注点最小二乘拟合的仿射（gtfit）分解出 patch 中心位移、旋转角、各向同性缩放。"""
    rows = []
    for s in ("val", "test"):
        t, r, k = [], [], []
        for line in open(R / "preds" / "gtfit" / f"{s}.jsonl", encoding="utf-8"):
            A = np.array(json.loads(line)["A"], float)
            L, c = A[:, :2], np.array([256.0, 256.0])
            t.append(np.linalg.norm(L @ c + A[:, 2] - c))
            U, S, Vt = np.linalg.svd(L)
            Rm = U @ Vt
            r.append(abs(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))))
            k.append(abs(np.sqrt(S.prod()) - 1) * 100)
        for name, v in (("shift_px", t), ("rot_deg", r), ("scale_pct", k)):
            q = np.percentile(v, [50, 90, 99, 100])
            rows.append({"split": s, "quantity": name, **{f"p{p}": round(float(x), 2) for p, x in zip((50, 90, 99, 100), q)}})
    write_csv(OUT / "data" / "gt_dist.csv", rows)
    return {(r["split"], r["quantity"]): r for r in rows}


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    with open(OUT / "data" / "summary.csv", encoding="utf-8") as f:
        get = {(r["method"], r["variant"], r["split"]): float(r["auc@5"]) for r in csv.DictReader(f)}
    dist = gt_dist()
    p90 = {"shift": dist["val", "shift_px"]["p90"], "rot": dist["val", "rot_deg"]["p90"], "scale": dist["val", "scale_pct"]["p90"]}

    # 图 1：总览，三个轴各一栏，Val AUC@5 随档位
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, (a, (lab, ls)) in zip(axs, AXES.items()):
        ax.axvspan(0, p90[a], color="#eeeeee", zorder=0)
        ax.text(ls[-1] * .01, 0.008, "灰底：原数据 90% 的 pair", fontsize=8, color="#777777")
        for m in PLOT_METHODS:
            c, st, mk = STYLE[m]
            ys = [get[m, "orig" if l == 0 else f"{a}{l}", "val"] for l in ls]
            ax.plot(ls, ys, st, color=c, marker=mk, ms=5, lw=2, label=SHORT[m])
        ax.set_xticks(ls)
        ax.set_xlim(-ls[-1] * .03, ls[-1] * 1.03)
        ax.set_xlabel(lab)
        ax.set_ylim(0, 0.36)
        ax.grid(alpha=.3)
    axs[0].set_ylabel("Val AUC@5")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "auc5_vs_level.png", dpi=130)
    plt.close(fig)

    # 图 2：原始与三个轴最重一档，5 个方法并排
    fig, ax = plt.subplots(figsize=(10, 4.3))
    w = 0.16
    for i, m in enumerate(PLOT_METHODS):
        c, st, _ = STYLE[m]
        xs = np.arange(len(BARS)) + (i - 2) * w
        ys = [get[m, v, "val"] for v, _ in BARS]
        ax.bar(xs, ys, w * .92, color=c, hatch="//" if st == "--" else None, edgecolor="white", label=SHORT[m])
        for x, y in zip(xs, ys):
            ax.text(x, y + .004, f"{y:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(BARS)), [b for _, b in BARS])
    ax.set_ylabel("Val AUC@5")
    ax.set_ylim(0, 0.37)
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=5, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "auc5_bars.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    plot() if "--plot" in sys.argv else tables()
