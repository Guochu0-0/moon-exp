"""E16 的图与补充统计（读 extra/data/pairs.csv，写 extra/diff_by_temp.png 与 extra/data/extra_stats.json）。只用 CPU。

    /opt/envs/loftr/bin/python runs/E16/code/plot.py      （工作台环境没有 matplotlib）
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path(__file__).resolve().parents[1]
FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 服务器上没有中文字体，放了一份在 gpfs（同 E4）
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OK = 5.0                                                # 不抽签时标注点误差 ≤ 5 px 记为配准成功（同主指标 AUC@5 的阈值）


def load():
    with open(RUN / "extra/data/pairs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "pair":
                r[k] = float(v) if v not in ("True", "False") else v == "True"
    return rows


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def main():
    rows = load()
    temps = sorted({r["temp"] for r in rows}, reverse=True)
    col = lambda rs, k: np.array([r[k] for r in rs], np.float64)
    out = {}
    for tp in temps:
        rs = [r for r in rows if r["temp"] == tp]
        e0 = col(rs, "err0")
        groups = {"all": np.ones(len(rs), bool), "ok": e0 <= OK, "bad": e0 > OK}
        g = {}
        for name, m in groups.items():
            sub = [r for r, x in zip(rs, m) if x]
            g[name] = {"n": len(sub),
                       "diff_med": r4(np.median(col(sub, "diff_med"))),
                       "frac_all16_worse": r4((col(sub, "n_worse") == 16).mean()),
                       "frac_best_better": r4(col(sub, "best_better").mean()),
                       "n_worse_med": r4(np.median(col(sub, "n_worse")))}
        out[str(tp)] = {
            "groups": g,
            "frac_rho_ncc_neg": r4((col(rs, "rho_ncc") < 0).mean()),
            "frac_rho_r1_neg": r4((col(rs, "rho_r1") < 0).mean()),
            "conf_cells": {"d_mode_med": r4(np.nanmedian(col(rs, "d_mode_med_conf"))),
                           "d_samp_med": r4(np.nanmedian(col(rs, "d_samp_med_conf")))},
            "all_cells": {"d_mode_med": r4(np.nanmedian(col(rs, "d_mode_med"))),
                          "d_samp_med": r4(np.nanmedian(col(rs, "d_samp_med")))},
        }
    (RUN / "extra/data/extra_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    data = [col([r for r in rows if r["temp"] == tp], "diff_med") for tp in temps]
    pos = np.arange(len(temps))
    for x, d in zip(pos, data):
        p10, p25, p50, p75, p90 = np.percentile(d, [10, 25, 50, 75, 90])
        ax.plot([x, x], [p10, p90], color="0.4", lw=1)
        ax.add_patch(plt.Rectangle((x - 0.25, p25), 0.5, p75 - p25, fc="C0", alpha=0.35, ec="C0"))
        ax.plot([x - 0.25, x + 0.25], [p50, p50], color="C0", lw=2.2)
        ax.text(x + 0.3, p50, f"{p50:+.2f}", va="center", fontsize=9, color="C0")
    ax.axhline(1.0, color="C3", ls="--", lw=1.2, label="选温度的门槛：中位数 ≤ 1 像素")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(pos, [f"{tp:g}" for tp in temps])
    ax.set_xlabel("抽签温度")
    ax.set_ylabel("16 次抽签误差的中位数 − 不抽签误差（像素）")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_xlim(-0.6, len(temps) - 0.2)
    fig.tight_layout()
    fig.savefig(RUN / "extra/diff_by_temp.png", dpi=150)


if __name__ == "__main__":
    main()
