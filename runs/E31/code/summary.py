"""E31 表与图（#217）：3 个运行第 1200 步的 Val 结局与训练日志（核对距离 mp_gap、被拦 mp_block、画面内格子比例），
以及只读检查 gap_check 的逐存档汇总（E26 的坍缩存档与健康存档上的核对距离能否把两者分开）。
Val 取 runs/E31/metrics.json（第 1200 步，配准误差中位数 > 20 px 记失效，同 E28）；日志取 ckpt/<m>/log.jsonl，
曲线为 50 步滑动中位数 / 滑动平均（同 E26–E28、E30）。对照取 E30 的 d_spread 组（约束从未生效，等于底座，种子 0–2、第 1200 步）。
两组存档之间、逐对核对距离的区分度用 ROC 曲线下面积（AUC，等于 Mann–Whitney U / (n₁n₂)，Hanley & McNeil 1982）。
在 gpfs 上跑（ckpt 不同步到本机）：

    /opt/envs/wb/bin/python runs/E31/code/summary.py
写 extra/data/summary.csv、extra/data/gap_summary.csv、extra/curves.png、extra/gap_check.png，并打印表格。
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
E30 = Path("/remote-home/xufang/YGC/moon-exp/runs/E30")    # 已结票：ckpt 在 gpfs 主 checkout
WIN = 50
FAIL_PX = 20.0
THR = 16.0
RUNS = ("mp_s0", "mp_s1", "mp_s2")
BASE = ("d_spread_s0", "d_spread_s1", "d_spread_s2")     # E30：等于底座的一组
COLLAPSED = ("e15_loo_t02", "e15_none_t02", "e15_none_t05", "e19_clip", "e22_f0002", "e23_c16_s1", "e25_f1_s1")
HEALTHY = ("e19_gauss8", "e23_c16_s2", "e25_fr1_s1")


def load(path, keys):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.startswith("{")]
    return np.array([r["step"] for r in rows]), {k: np.array([r.get(k, np.nan) for r in rows], float) for k in keys}


def roll(x, f):
    return np.array([f(x[max(0, i - WIN + 1):i + 1]) if np.isfinite(x[max(0, i - WIN + 1):i + 1]).any() else np.nan
                     for i in range(len(x))])


def auc(pos, neg):
    """P(pos > neg) + ½P(相等)：pos 的值大于 neg 的概率（ROC 曲线下面积）。"""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def r3(x):
    return round(float(x), 3)


def runs_table():
    met = json.loads((R / "metrics.json").read_text(encoding="utf-8"))["methods"]
    met30 = json.loads((E30 / "metrics.json").read_text(encoding="utf-8"))["methods"]
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for s, m in enumerate(RUNS):
        v = met[m]["val"]["summary"]
        step, x = load(R / "ckpt" / m / "log.jsonl", ("mp_gap", "mp_block", "mp_fail", "in_frame", "pmode"))
        gap_s, blk_s, inf_s = roll(x["mp_gap"], np.nanmedian), roll(x["mp_block"], np.nanmean), roll(x["in_frame"], np.nanmedian)
        rows.append({"method": m, "seed": s, "outcome": "健康" if v["median"] <= FAIL_PX else "失效",
                     "val_auc5": r3(v["auc@5"]), "val_median_px": round(v["median"], 2),
                     "blocked": r3(np.nanmean(x["mp_block"])), "blocked_1_100": r3(np.nanmean(x["mp_block"][:100])),
                     "blocked_last100": r3(np.nanmean(x["mp_block"][-100:])),
                     "fit_fail": r3(np.nanmean(x["mp_fail"])),
                     "gap_median": round(float(np.nanmedian(x["mp_gap"])), 1),
                     "gap_median_last100": round(float(np.nanmedian(x["mp_gap"][-100:])), 1),
                     "in_frame_mean": r3(np.nanmean(x["in_frame"])), "in_frame_min_smooth": r3(np.nanmin(inf_s)),
                     "pmode_last100": r3(np.nanmean(x["pmode"][-100:]))})
        ax[0].plot(step, gap_s, color=f"C{s}", lw=1.1, label=f"种子 {s}")
        ax[1].plot(step, blk_s, color=f"C{s}", lw=1.1, label=f"种子 {s}")
        ax[2].plot(step, inf_s, color=f"C{s}", lw=1.1, label=f"种子 {s}")
    for s, m in enumerate(BASE):
        v = met30[m]["val"]["summary"]
        step, x = load(E30 / "ckpt" / m / "log.jsonl", ("in_frame", "pmode"))
        inf_s = roll(x["in_frame"], np.nanmedian)
        rows.append({"method": f"E30/{m}", "seed": s, "outcome": "健康" if v["median"] <= FAIL_PX else "失效",
                     "val_auc5": r3(v["auc@5"]), "val_median_px": round(v["median"], 2),
                     "blocked": "", "blocked_1_100": "", "blocked_last100": "", "fit_fail": "", "gap_median": "",
                     "gap_median_last100": "", "in_frame_mean": r3(np.nanmean(x["in_frame"])),
                     "in_frame_min_smooth": r3(np.nanmin(inf_s)), "pmode_last100": r3(np.nanmean(x["pmode"][-100:]))})
        ax[2].plot(step, inf_s, color=f"C{s}", lw=0.8, ls=":", alpha=0.8, label=f"底座（E30）种子 {s}")
    ax[0].axhline(THR, color="0.5", ls="--", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("核对距离（px，50 步滑动中位数）")
    ax[1].set_ylabel("被拦比例（50 步滑动平均）")
    ax[2].set_ylabel("画面内格子比例（50 步滑动中位数）")
    for a in ax:
        a.set_xlabel("步")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)
    return rows


def gap_table():
    rows = list(csv.DictReader(open(R / "extra" / "data" / "gap_check.csv", encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["ckpt"], []).append(r)
    out, groups = [], {"坍缩": [], "健康": [], "未训练": []}
    order = [f"{c}_4000" for c in COLLAPSED] + [f"{c}_4000" for c in HEALTHY] + ["zs"]
    for name in order:
        rs = by[name]
        g = np.array([float(r["gap"]) for r in rs])
        e = np.array([float(r["err0"]) for r in rs])
        grp = "未训练" if name == "zs" else ("坍缩" if name[:-5] in COLLAPSED else "健康")
        groups[grp].append(g)
        q1, q2, q3 = np.nanpercentile(g, [25, 50, 75])
        out.append({"ckpt": name, "group": grp, "n": len(g), "gap_q1": round(q1, 1), "gap_median": round(q2, 1),
                    "gap_q3": round(q3, 1), "blocked": r3(np.mean(~(g <= THR))),
                    "fit_fail": r3(np.mean([int(r["fail"]) for r in rs])),
                    "err0_median": round(float(np.nanmedian(e)), 1), "err0_gt20": r3(np.mean(~(e <= FAIL_PX))),
                    "auc_vs_healthy": ""})
    healthy = np.concatenate(groups["健康"])
    healthy = healthy[np.isfinite(healthy)]
    for r in out:
        if r["group"] != "健康":
            g = np.array([float(x["gap"]) for x in by[r["ckpt"]]])
            r["auc_vs_healthy"] = r3(auc(np.nan_to_num(g, nan=np.inf, posinf=1e9), healthy))
    coll = np.nan_to_num(np.concatenate(groups["坍缩"]), nan=np.inf, posinf=1e9)
    pooled = auc(coll, healthy)
    # 逐对：健康存档里配错（err0 > 20）的对与配对的对
    hr = [r for r in rows if r["ckpt"][:-5] in HEALTHY]
    good = np.array([float(r["gap"]) for r in hr if float(r["err0"]) <= FAIL_PX])
    bad = np.array([float(r["gap"]) for r in hr if not float(r["err0"]) <= FAIL_PX])
    extra = {"pooled_auc_collapsed_vs_healthy": r3(pooled),
             "healthy_pairs_ok": len(good), "healthy_pairs_bad": len(bad),
             "healthy_ok_blocked": r3(np.mean(~(good <= THR))),
             "healthy_bad_blocked": r3(np.mean(~(bad <= THR))) if len(bad) else "",
             "auc_bad_vs_ok_in_healthy": r3(auc(np.nan_to_num(bad, nan=1e9), good)) if len(bad) else ""}

    fig, a = plt.subplots(figsize=(11, 4.6))
    col = {"坍缩": "C3", "健康": "C2", "未训练": "C7"}
    for i, r in enumerate(out):
        g = np.array([float(x["gap"]) for x in by[r["ckpt"]]])
        g = np.clip(np.nan_to_num(g, nan=3e3, posinf=3e3), 0.1, 3e3)
        jit = np.random.default_rng(i).uniform(-0.25, 0.25, len(g))
        a.scatter(i + jit, g, s=6, color=col[r["group"]], alpha=0.5)
        a.plot([i - 0.3, i + 0.3], [r["gap_median"]] * 2, color="k", lw=1.5)
    a.axhline(THR, color="0.4", ls="--", lw=0.8)
    a.set_yscale("log")
    a.set_xticks(range(len(out)), [r["ckpt"].replace("_4000", "") for r in out], rotation=30, ha="right", fontsize=8)
    a.set_ylabel("核对距离（px，对数轴）")
    for grp, c in col.items():
        a.scatter([], [], color=c, label=grp)
    a.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(R / "extra" / "gap_check.png", dpi=110)
    return out, extra


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    rows = runs_table()
    write(R / "extra" / "data" / "summary.csv", rows)
    gap, extra = gap_table()
    write(R / "extra" / "data" / "gap_summary.csv", gap)
    (R / "extra" / "data" / "gap_separation.json").write_text(json.dumps(extra, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
    for r in rows + gap:
        print(" | ".join(f"{k}={r[k]}" for k in r))
    print(extra)


if __name__ == "__main__":
    main()
