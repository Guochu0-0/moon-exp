"""E37 表与图（#236）。在 gpfs 上跑（ckpt 不同步到本机）：

    /opt/envs/loftr/bin/python runs/E37/code/summary.py

写 extra/data/*.csv、extra/curves.png、extra/diff.png，并打印表格（写法照 runs/E36/code/summary.py）。
- 两组（4 卡、每次更新 4 对，训 7907 次更新 = 31628 对）：all（细化尺度 16、8、4、2、1 都训）、three（只训 16、8、4）。
- Val：sweep/<组>/S/metrics.json 各存档的逐对误差与 AUC@5；横轴为已训图对数（更新次数 × 4）。
- 比较用配对自助法：同一批 Val 图对有放回重抽 2000 次，取 AUC@5 之差的 2.5% 与 97.5% 分位（同 E33、E36）。
  1. 每个存档上 all 减 three；
  2. 同一个运行里，最后一个存档与最高的存档比。
- 判据量：离参照的距离 d_ref、画面内格子比例 in_frame，50 次更新的滑动中位数（多卡日志每次更新一行，为各卡平均）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parents[1]))
from workbench.protocol import auc  # noqa: E402

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RUNS = {"all": ("五级（16、8、4、2、1）", "C3"), "three": ("三级（16、8、4）", "C0")}
N, D_MAX, IN_MIN, WIN = 4, 100.0, 0.5, 50
OUT = R / "extra" / "data"


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.startswith("{")]


def roll(x, win):
    return np.array([np.nanmedian(x[max(0, i - win + 1):i + 1]) if np.isfinite(x[max(0, i - win + 1):i + 1]).any()
                     else np.nan for i in range(len(x))])


def paired(e1, e0, n=2000, seed=0):
    fix = lambda e: np.array([np.inf if x is None else x for x in e], float)   # noqa: E731
    e1, e0 = fix(e1), fix(e0)
    idx = np.random.default_rng(seed).integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return auc(e1, 5.0) - auc(e0, 5.0), np.percentile(diff, [2.5, 97.5])


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"== {name}")
    for r in rows:
        print(" | ".join(f"{k}={r[k]}" for k in r))


def load_val(m):
    """{已训图对数: Val 记录}"""
    met = R / "sweep" / m / "S" / "metrics.json"
    if not met.exists():
        return {}
    return {int(k[4:]) * N: v["val"] for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items()
            if k.startswith("step") and "val" in v}


def cmp_row(a, b, va, vb, **key):
    d, ci = paired(va["errors"], vb["errors"])
    return {**key, "a": a, "b": b, "auc5_a": round(va["summary"]["auc@5"], 3), "auc5_b": round(vb["summary"]["auc@5"], 3),
            "diff": round(d, 3), "ci_lo": round(ci[0], 3), "ci_hi": round(ci[1], 3)}


def main():
    (R / "extra").mkdir(exist_ok=True)
    val = {m: load_val(m) for m in RUNS}
    write("val.csv", [{"run": m, "pairs": p, "updates": p // N, "auc5": round(val[m][p]["summary"]["auc@5"], 3),
                       "median_err": round(val[m][p]["summary"]["median"], 2)} for m in RUNS for p in sorted(val[m])])

    ps = sorted(set(val["all"]) & set(val["three"]))
    diff = [cmp_row("all", "three", val["all"][p], val["three"][p], pairs=p) for p in ps]
    if diff:
        write("all_vs_three.csv", diff)

    within = []
    for m in RUNS:
        v = val[m]
        if v:
            last, best = max(v), max(v, key=lambda q: v[q]["summary"]["auc@5"])
            within.append({"best_pairs": best, **cmp_row(m, m, v[last], v[best], last_pairs=last)})
    if within:
        write("last_vs_best.csv", within)

    # 图 1：判据量与 Val AUC@5
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for m, (name, c) in RUNS.items():
        f = R / "ckpt" / m / "log.jsonl"
        if f.exists():
            log = jl(f)
            pairs = np.array([r["step"] * N for r in log])
            d_s = roll(np.array([r.get("d_ref", np.nan) for r in log], float), WIN)
            in_s = roll(np.array([r.get("in_frame", np.nan) for r in log], float), WIN)
            ax[0].plot(pairs, d_s, color=c, lw=1.1, label=name)
            ax[1].plot(pairs, in_s, color=c, lw=1.1, label=name)
            h = R / "ckpt" / m / "halt.json"
            halt = json.loads(h.read_text()) if h.exists() else None
            print(f"{m}: 训到 {int(pairs[-1])} 对；提前停 {halt}；d_ref 滑动中位数最大 {np.nanmax(d_s):.1f}，"
                  f"in_frame 滑动中位数最小 {np.nanmin(in_s):.3f}；每次更新 {np.median([r['sec'] for r in log if 'sec' in r]):.2f} s")
        if val[m]:
            p_ = sorted(val[m])
            ax[2].plot(p_, [val[m][p]["summary"]["auc@5"] for p in p_], "o-", color=c, lw=1.1, ms=3, label=name)
    ax[0].axhline(D_MAX, color="0.5", ls=":", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("离参照的距离（px，50 次更新滑动中位数）")
    ax[1].axhline(IN_MIN, color="0.5", ls=":", lw=0.8)
    ax[1].set_ylabel("画面内格子比例（50 次更新滑动中位数）")
    ax[2].set_ylabel("Val AUC@5")
    for a in ax:
        a.set_xlabel("已训图对数")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)

    # 图 2：每个存档上五级减三级之差与 95% 区间
    if diff:
        fig, a = plt.subplots(figsize=(7, 4.2))
        x = [r["pairs"] for r in diff]
        y = np.array([r["diff"] for r in diff])
        a.errorbar(x, y, yerr=[y - [r["ci_lo"] for r in diff], [r["ci_hi"] for r in diff] - y], fmt="o", color="k",
                   ms=3, capsize=2, lw=0.8)
        a.axhline(0, color="0.5", ls=":", lw=0.8)
        a.set_xlabel("已训图对数")
        a.set_ylabel("Val AUC@5 之差（五级 − 三级）")
        fig.tight_layout()
        fig.savefig(R / "extra" / "diff.png", dpi=110)


if __name__ == "__main__":
    main()
