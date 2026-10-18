"""E9 的数据与图（#157、#159）：读 runs/E7/metrics.json 与 runs/E9/metrics.json（难版保留全部检查点的口径），
任何装了 matplotlib 的环境都能跑。

    python runs/E9/code/plot.py      # 在仓库根目录，先 python -m workbench eval E7 E9

写 extra/data/auc5.csv（模型、档位、split、AUC@5）和两张图：
图 1 extra/heaviest.png：三个轴最重一档的 Val AUC@5 减原始档；两个 zero-shot 各一根灰柱，三种训练方法各两根（浅色旧训练、深色重训）。
图 2 extra/auc5_vs_level.png：三个轴各一栏，五个当前模型的 Val AUC@5 随档位（0 为原始）。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "extra"
AXES = [("shift", "平移", "px", [12, 64, 128]), ("rot", "旋转", "°", [3, 10, 20]), ("scale", "缩放", "%", [3, 10, 20])]
ZS = [("LoFTR zero-shot", ("E7", "anymatch_loftr")), ("RoMa zero-shot", ("E7", "anymatch_roma"))]
# (图例名, 旧训练, 重训, 颜色, 标记)
TRAINED = [("LoFTR 离线伪标签", ("E7", "P10"), ("E9", "lns_r4"), "#1f77b4", "o"),
           ("LoFTR 在线自监督", ("E7", "Q4"), ("E9", "q4g"), "#2ca02c", "o"),
           ("RoMa 离线伪标签", ("E7", "erase_geo__r2"), ("E9", "rns_r2"), "#d62728", "s")]
LEVELS = [""] + [f"{k}{v}" for k, _, _, vs in AXES for v in vs]


def font():
    from matplotlib import font_manager
    for f in ("Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load():
    M = {e: json.loads((ROOT / "runs" / e / "metrics.json").read_text(encoding="utf-8"))["methods"] for e in ("E7", "E9")}
    models = [m for _, m in ZS] + [m for t in TRAINED for m in t[1:3]]
    return {(e, m, lv, s): M[e][f"{m}__{lv}" if lv else m][s]["summary"]["auc@5"]
            for e, m in models for lv in LEVELS for s in ("val", "test")}


def main():
    font()
    A = load()
    OUT.joinpath("data").mkdir(parents=True, exist_ok=True)
    with open(OUT / "data" / "auc5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["exp", "method", "level", "split", "auc@5"])
        w.writerows([*k, round(v, 4)] for k, v in A.items())

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (k, name, unit, vs) in zip(axs, AXES):
        lv, x, ticks = f"{k}{vs[-1]}", 0.0, []
        d = lambda e, m: A[(e, m, lv, "val")] - A[(e, m, "", "val")]
        for lab, m in ZS:
            ax.bar(x, d(*m), 0.6, color="0.7")
            ticks.append((x, lab.replace(" ", "\n", 1))); x += 1.2
        for i, (lab, old, new, *_) in enumerate(TRAINED):
            ax.bar(x - 0.2, d(*old), 0.4, color="#9ecae1", label="旧训练（扰动 ±12 px、±3°、±3%）" if i == 0 else None)
            ax.bar(x + 0.2, d(*new), 0.4, color="#08519c", label="重训（扰动放大到最重一档）" if i == 0 else None)
            ticks.append((x, lab.replace(" ", "\n", 1)))
            x += 1.2
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks([t for t, _ in ticks], [s for _, s in ticks], fontsize=8)
        ax.set_title(f"{name} {vs[-1]}{"" if unit in "°%" else " "}{unit}")
        ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("Val AUC@5：最重一档 - 原始")
    axs[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "heaviest.png", dpi=150)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    lines = [(lab, m, "0.55", "o" if "LoFTR" in lab else "s", "--") for lab, m in ZS] + \
            [(f"{lab}（重训）", new, c, mk, "-") for lab, _, new, c, mk in TRAINED]
    for ax, (k, name, unit, vs) in zip(axs, AXES):
        for lab, (e, m), c, mk, ls in lines:
            ax.plot([0] + vs, [A[(e, m, lv, "val")] for lv in [""] + [f"{k}{v}" for v in vs]], ls, color=c, marker=mk,
                    label=lab)
        ax.set_xticks([0] + vs)
        ax.set_xlabel(f"{name}（{unit}）")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Val AUC@5")
    axs[0].set_ylim(0, max(max(A[(e, m, lv, "val")] for _, (e, m), *_ in lines for lv in LEVELS) + 0.03, 0.4))
    axs[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "auc5_vs_level.png", dpi=150)


if __name__ == "__main__":
    main()
