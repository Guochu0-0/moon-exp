"""E28 图 1 与表 2（#212）：5 个种子与原设置（E19 gauss8、E27 gauss8 三个种子）的训练日志曲线。
最高分位置的概率（pmode）、离自估仿射的距离（d_ref）、画面内格子比例（in_frame，只有 E28 有），各取 50 步滑动中位数（同 E26、E27）。
只读 runs/<id>/ckpt/<m>/log.jsonl（在 gpfs 上跑：ckpt 不同步到本机）。

    /opt/envs/wb/bin/python runs/E28/code/curves.py [--main <主 checkout 的 runs/>]
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
KEYS = ("pmode", "d_ref", "in_frame")
WIN = 50


def load(path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")]
    rows = [r for r in rows if "pmode" in r and r.get("step", 0) <= 2000]
    step = np.array([r["step"] for r in rows])
    return step, {k: np.array([r.get(k, np.nan) for r in rows], float) for k in KEYS}


def smooth(x):
    return np.array([np.nanmedian(x[max(0, i - WIN + 1):i + 1]) for i in range(len(x))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="/remote-home/xufang/YGC/moon-exp/runs")
    a = ap.parse_args()
    M = Path(a.main)
    runs = [(f"fm_s{s}", R / "ckpt" / f"fm_s{s}" / "log.jsonl", ("C0", "C0", "C0", "C3", "C1")[s], "-") for s in range(5)]
    runs += [("E19 gauss8", M / "E19/ckpt/gauss8/log.jsonl", "k", "--")]
    runs += [(f"E27 gauss8_s{s}", M / f"E27/ckpt/gauss8_s{s}/log.jsonl", "0.5", ":") for s in range(3)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    out = R / "extra"
    (out / "data").mkdir(parents=True, exist_ok=True)
    summary = []
    for name, p, c, ls in runs:
        if not p.exists():
            print("missing", p)
            continue
        step, v = load(p)
        sm = {k: smooth(x) for k, x in v.items()}
        for i, k in enumerate(KEYS):
            if np.isfinite(sm[k]).any():
                ax[i].plot(step, sm[k], color=c, ls=ls, lw=1.2, label=name)
        hit = step[sm["pmode"] > 0.99]
        last = step >= step[-1] - 99
        summary.append({"run": name, "last_step": int(step[-1]), "pmode_max": round(float(np.nanmax(sm["pmode"])), 3),
                        "first_pmode_gt_0.99": int(hit[0]) if len(hit) else "",
                        "pmode_last100": round(float(np.nanmean(v["pmode"][last])), 3),
                        "d_ref_last100": round(float(np.nanmean(v["d_ref"][last])), 1),
                        "in_frame_mean": round(float(np.nanmean(v["in_frame"])), 3) if np.isfinite(v["in_frame"]).any() else ""})
    for i, (k, t) in enumerate(zip(KEYS, ("最高分位置的概率", "离自估仿射的距离（px）", "画面内格子比例"))):
        ax[i].set_title(t)
        ax[i].set_xlabel("步")
    ax[1].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    with open(out / "data" / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
