"""E29 图表（#215）：训练日志曲线与汇总。只读 runs/<id>/ckpt/<m>/log.jsonl（在 gpfs 上跑：ckpt 不同步到本机）。
- 图 1（curves_b4.png）：每步 4 对（b4_s0–2）与 E28 fm_s0–4 的最高分位置的概率（pmode）、离自估仿射的距离（d_ref）、
  画面内格子比例（in_frame），横轴为参数更新次数；E28 每步 1 对，横轴同样是更新次数。
- 图 2（curves_rand.png）：随机奖励 rand_s0–2（温度 0.3）与 E13 s0_rand、s1_rand（温度 1）的熵（H，按各自采样温度的分布算）
  与最高分位置的概率。
曲线与汇总均取 50 步滑动中位数（同 E26–E28）。汇总写 extra/data/summary_b4.csv、summary_rand.csv。

    /opt/envs/wb/bin/python runs/E29/code/curves.py [--main <主 checkout 的 runs/>]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"   # 同 E27 figs.py
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
R = Path(__file__).resolve().parents[1]
WIN = 50


def load(path, keys, upto):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")]
    rows = [r for r in rows if "pmode" in r and r.get("step", 0) <= upto]
    step = np.array([r["step"] for r in rows])
    return step, {k: np.array([r.get(k, np.nan) for r in rows], float) for k in keys}


def smooth(x):
    return np.array([np.nanmedian(x[max(0, i - WIN + 1):i + 1]) if np.isfinite(x[max(0, i - WIN + 1):i + 1]).any()
                     else np.nan for i in range(len(x))])


def write(rows, name):
    with open(R / "extra" / "data" / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in rows:
        print(s)


def first(step, mask):
    return int(step[mask][0]) if mask.any() else ""


def fig_b4(M):
    keys = ("pmode", "d_ref", "in_frame")
    runs = [(f"b4_s{s}", R / "ckpt" / f"b4_s{s}" / "log.jsonl", f"C{s}", "-", 1200) for s in range(3)]
    runs += [(f"E28 fm_s{s}", M / f"E28/ckpt/fm_s{s}/log.jsonl", "0.6", ":", 2000) for s in range(5)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    out = []
    for name, p, c, ls, upto in runs:
        if not p.exists():
            print("missing", p)
            continue
        step, v = load(p, keys, upto)
        sm = {k: smooth(x) for k, x in v.items()}
        for i, k in enumerate(keys):
            ax[i].plot(step, sm[k], color=c, ls=ls, lw=1.2, label=name)
        last = step >= step[-1] - 99
        out.append({"run": name, "last_step": int(step[-1]),
                    "in_frame_min_sm": round(float(np.nanmin(sm["in_frame"])), 3),
                    "first_in_frame_lt_0.5": first(step, sm["in_frame"] < 0.5),
                    "in_frame_last100": round(float(np.nanmean(v["in_frame"][last])), 3),
                    "pmode_max_sm": round(float(np.nanmax(sm["pmode"])), 3),
                    "first_pmode_gt_0.99": first(step, sm["pmode"] > 0.99),
                    "pmode_last100": round(float(np.nanmean(v["pmode"][last])), 3),
                    "d_ref_max_sm": round(float(np.nanmax(sm["d_ref"])), 1),
                    "d_ref_last100": round(float(np.nanmean(v["d_ref"][last])), 1)})
    for i, t in enumerate(("最高分位置的概率", "离自估仿射的距离（px）", "画面内格子比例")):
        ax[i].set_title(t)
        ax[i].set_xlabel("参数更新次数")
    ax[1].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves_b4.png", dpi=110)
    write(out, "summary_b4.csv")


def fig_rand(M):
    keys = ("H", "pmode")
    runs = [(f"rand_s{s}（温度 0.3）", R / "ckpt" / f"rand_s{s}" / "log.jsonl", f"C{s}", "-") for s in range(3)]
    runs += [(f"E13 s{s}_rand（温度 1）", M / f"E13/ckpt/s{s}_rand/log.jsonl", "0.5", ":") for s in range(2)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    out = []
    for name, p, c, ls in runs:
        if not p.exists():
            print("missing", p)
            continue
        step, v = load(p, keys, 2000)
        sm = {k: smooth(x) for k, x in v.items()}
        for i, k in enumerate(keys):
            ax[i].plot(step, sm[k], color=c, ls=ls, lw=1.2, label=name)
        last = step >= step[-1] - 99
        out.append({"run": name, "last_step": int(step[-1]),
                    "H_first100": round(float(np.nanmean(v["H"][step <= 100])), 3),
                    "first_H_lt_0.1": first(step, sm["H"] < 0.1),
                    "H_min_sm": round(float(np.nanmin(sm["H"])), 4),
                    "H_last100": round(float(np.nanmean(v["H"][last])), 4),
                    "pmode_first100": round(float(np.nanmean(v["pmode"][step <= 100])), 3),
                    "pmode_last100": round(float(np.nanmean(v["pmode"][last])), 3)})
    ax[0].set_title("熵（按采样温度的分布，nat）")
    ax[0].set_yscale("log")
    ax[0].axhline(0.1, color="k", lw=0.6, ls="--")
    ax[1].set_title("最高分位置的概率")
    for a in ax:
        a.set_xlabel("步")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves_rand.png", dpi=110)
    write(out, "summary_rand.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="/remote-home/xufang/YGC/moon-exp/runs")
    a = ap.parse_args()
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    fig_b4(Path(a.main))
    fig_rand(Path(a.main))


if __name__ == "__main__":
    main()
