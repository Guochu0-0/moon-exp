"""E30 表与图（#216）：24 次运行第 1200 步的 Val 结局，和训练日志里的画面内格子比例、被拦比例、慢速参照离当前模型的距离、
H(p̄)。Val 取 runs/E30/metrics.json（第 1200 步，配准误差中位数 > 20 px 记失效，同 E28）；日志取 ckpt/<m>/log.jsonl，
曲线为 50 步滑动中位数（同 E26–E28）。在 gpfs 上跑（ckpt 不同步到本机）。

    /opt/envs/wb/bin/python runs/E30/code/summary.py
写 extra/summary.csv、extra/curves.png，并打印表格。
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
WIN = 50
FAIL_PX = 20.0
GROUPS = ("a_gate", "b_ema990", "b_ema996", "b_ema999", "c_ema990g", "c_ema996g", "c_ema999g", "d_spread")


def load(m):
    rows = [json.loads(x) for x in (R / "ckpt" / m / "log.jsonl").read_text().splitlines() if x.startswith("{")]
    keys = ("in_frame", "gated", "gate_det", "gate_sv", "gate_in", "ref_gap", "H_bar", "pmode", "d_ref")
    return np.array([r["step"] for r in rows]), {k: np.array([r.get(k, np.nan) for r in rows], float) for k in keys}


def smooth(x):
    return np.array([np.nanmedian(x[max(0, i - WIN + 1):i + 1]) if np.isfinite(x[max(0, i - WIN + 1):i + 1]).any()
                     else np.nan for i in range(len(x))])


def nanmean(x):
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def main():
    met = json.loads((R / "metrics.json").read_text(encoding="utf-8"))["methods"]
    rows = []
    fig, ax = plt.subplots(2, 4, figsize=(18, 7.5), sharex=True, sharey=True)
    for gi, g in enumerate(GROUPS):
        a = ax.flat[gi]
        for s in range(3):
            m = f"{g}_s{s}"
            if not (R / "ckpt" / m / "log.jsonl").exists():
                continue
            v = met.get(m, {}).get("val", {}).get("summary", {})
            med = v.get("median", float("nan"))
            ok = med <= FAIL_PX if np.isfinite(med) else None
            step, x = load(m)
            sm = {k: smooth(y) for k, y in x.items()}
            first = step[np.argmax(sm["in_frame"] < 0.5)] if (sm["in_frame"] < 0.5).any() else ""
            rows.append({"method": m, "group": g, "seed": s, "outcome": "" if ok is None else ("健康" if ok else "失效"),
                         "val_auc5": v.get("auc@5", ""), "val_median_px": med,
                         "in_frame_mean": round(nanmean(x["in_frame"]), 3),
                         "in_frame_min_smooth": round(float(np.nanmin(sm["in_frame"])), 3),
                         "in_frame_below_0.5_from": first,
                         "gated_frac": round(nanmean(x["gated"]), 4),
                         "gated_det": round(nanmean(x["gate_det"]), 4), "gated_sv": round(nanmean(x["gate_sv"]), 4),
                         "gated_in": round(nanmean(x["gate_in"]), 4),
                         "ref_gap_last100": round(nanmean(x["ref_gap"][-100:]), 2),
                         "H_bar_min_smooth": round(float(np.nanmin(sm["H_bar"])), 3) if np.isfinite(sm["H_bar"]).any() else "",
                         "pmode_last100": round(nanmean(x["pmode"][-100:]), 3)})
            a.plot(step, sm["in_frame"], color=f"C{s}" if ok else "C3", lw=1.1, label=f"种子 {s}")
        a.set_title(g)
        a.axhline(0.427, color="0.6", ls=":", lw=0.8)
        a.legend(fontsize=8, loc="lower left")
    for a in ax[:, 0]:
        a.set_ylabel("画面内格子比例（50 步滑动中位数）")
    for a in ax[-1]:
        a.set_xlabel("步")
    fig.tight_layout()
    (R / "extra").mkdir(exist_ok=True)
    fig.savefig(R / "extra" / "curves.png", dpi=110)
    with open(R / "extra" / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" | ".join(str(r[k]) for k in r))


if __name__ == "__main__":
    main()
