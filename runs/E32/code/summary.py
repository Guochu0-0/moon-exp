"""E32 表与图（#221）：6 组（种子 0）是否提前停、训练日志的两条判据曲线、各存档的 Val。
- 日志取 ckpt/<组>/log.jsonl。判据量为离参照的距离 d_ref 与画面内格子比例 in_frame 的 50 步滑动中位数（同 E26–E31）；
  每步 4 对的组按前向步记日志，窗口取 200 个记录（= 50 次参数更新），横轴换算成参数更新次数。
- 提前停取 ckpt/<组>/halt.json（停在第几步、哪个量越线）。
- Val 取 sweep/<组>/S/metrics.json（各存档 step<N>，N 为前向步；每步 4 对的组换算成更新次数），AUC@5 与配准误差中位数。
- 判据（#221）：到第 2000 次更新同时满足——d_ref 滑动中位数始终 ≤ 100 px、in_frame 滑动中位数始终 ≥ 0.5、4 个存档 Val AUC@5 都 ≥ 0.29。
在 gpfs 上跑（ckpt 不同步到本机）：

    /opt/envs/wb/bin/python runs/E32/code/summary.py
写 extra/data/summary.csv、extra/curves.png，并打印表格。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
R = Path(__file__).resolve().parents[1]
GROUPS = {"base": "底座", "gate": "检查", "ema": "慢速参照", "ema_gate": "慢速参照 + 检查", "mp": "变换一致性检验",
          "acc4": "每步 4 对"}
ACCUM = {"acc4": 4}
D_MAX, IN_MIN, AUC_MIN, N_CKPT, UPDATES = 100.0, 0.5, 0.29, 4, 2000
KEYS = ("d_ref", "in_frame", "gated", "mp_block", "l_fine", "rew_16", "rew_8", "rew_4", "dmu_16", "dmu_8", "dmu_4",
        "dmu_1", "in_16")


def load(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.startswith("{")]
    return np.array([r["step"] for r in rows]), {k: np.array([r.get(k, np.nan) for r in rows], float) for k in KEYS}


def roll(x, win):
    return np.array([np.nanmedian(x[max(0, i - win + 1):i + 1]) if np.isfinite(x[max(0, i - win + 1):i + 1]).any()
                     else np.nan for i in range(len(x))])


def r3(x):
    return "" if x is None or not np.isfinite(x) else round(float(x), 3)


def main():
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for i, (m, name) in enumerate(GROUPS.items()):
        ck = R / "ckpt" / m
        if not (ck / "log.jsonl").exists():
            continue
        acc = ACCUM.get(m, 1)
        step, x = load(ck / "log.jsonl")
        upd = step / acc
        win = 50 * acc
        d_s, in_s = roll(x["d_ref"], win), roll(x["in_frame"], win)
        halt = json.loads((ck / "halt.json").read_text()) if (ck / "halt.json").exists() else None
        met = R / "sweep" / m / "S" / "metrics.json"
        val = {}
        if met.exists():
            for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items():
                if k.startswith("step") and "val" in v:
                    val[int(k[4:]) // acc] = (v["val"]["summary"]["auc@5"], v["val"]["summary"]["median"])
        aucs = [val[s][0] for s in sorted(val)]
        ok = (halt is None and len(aucs) == N_CKPT and min(aucs) >= AUC_MIN and np.nanmax(d_s) <= D_MAX
              and np.nanmin(in_s) >= IN_MIN and upd[-1] >= UPDATES)
        row = {"group": m, "name": name, "updates": int(upd[-1]),
               "halt_update": "" if halt is None else halt["step"] / acc,
               "halt_key": "" if halt is None else f"{halt['key']} {halt['op']} {halt['thr']}（中位数 {halt['median']}）",
               "d_ref_max_smooth": r3(np.nanmax(d_s)), "in_frame_min_smooth": r3(np.nanmin(in_s)),
               "gated": r3(np.nanmean(x["gated"])), "mp_block": r3(np.nanmean(x["mp_block"]))}
        for s in (500, 1000, 1500, 2000):
            row[f"auc5_{s}"] = r3(val[s][0]) if s in val else ""
            row[f"med_{s}"] = round(val[s][1], 2) if s in val else ""
        row["pass"] = "是" if ok else "否"
        rows.append(row)
        ax[0].plot(upd, d_s, color=f"C{i}", lw=1.1, label=name)
        ax[1].plot(upd, in_s, color=f"C{i}", lw=1.1, label=name)
        if val:
            ss = sorted(val)
            ax[2].plot(ss, [val[s][0] for s in ss], "o-", color=f"C{i}", lw=1.1, label=name)
        if halt is not None:
            for a in ax[:2]:
                a.axvline(halt["step"] / acc, color=f"C{i}", ls=":", lw=0.8)
    ax[0].axhline(D_MAX, color="0.5", ls="--", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("离参照的距离（px，50 次更新滑动中位数）")
    ax[1].axhline(IN_MIN, color="0.5", ls="--", lw=0.8)
    ax[1].set_ylabel("画面内格子比例（50 次更新滑动中位数）")
    ax[2].axhline(AUC_MIN, color="0.5", ls="--", lw=0.8)
    ax[2].set_ylabel("Val AUC@5")
    for a in ax:
        a.set_xlabel("参数更新次数")
        a.legend(fontsize=8)
    fig.tight_layout()
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    fig.savefig(R / "extra" / "curves.png", dpi=110)
    with open(R / "extra" / "data" / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" | ".join(f"{k}={r[k]}" for k in r))


if __name__ == "__main__":
    main()
