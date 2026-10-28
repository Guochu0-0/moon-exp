"""E36 表与图（#233）。在 gpfs 上跑（ckpt 不同步到本机）：

    /opt/envs/loftr/bin/python runs/E36/code/summary.py

写 extra/data/*.csv、extra/curves.png、extra/e33_by_updates.png，并打印表格。
- 本实验三组（4 卡、每次更新 4 对，训 2000 次更新）：ema_only、sqrt、linear。
- E33（gpfs 主 checkout 的 runs/E33）：gpu4（4 卡，未缩放）、gpu1（单卡，每次更新 1 对）。
- Val：sweep/<组>/S/metrics.json 各存档的逐对误差与 AUC@5；横轴统一为已训图对数（更新次数 × 每次更新的对数）。
- 比较一律用配对自助法：同一批 Val 图对有放回重抽 2000 次，取 AUC@5 之差的 2.5% 与 97.5% 分位（同 E33 summary.py）。
  1. 各组（含 E33 gpu4）与 E33 gpu1 在同一图对数的存档比（1000–8000 对），数区间不含 0 的存档；
  2. 本实验各组之间、ema_only 与 E33 gpu4 之间，同一图对数比（拆开 τ 与学习率各自的作用）；
  3. E33 按更新次数对齐：gpu4 第 u 次更新与 gpu1 第 u 次更新（u = 1000、2000、3000；gpu4 只训到 3954 次，
     u = 4000 用 gpu4 的最后一个存档 3954 次）；
  4. 同一个运行里，最后一个存档（8000 对）与最高的存档比。
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
E33 = Path("/remote-home/xufang/YGC/moon-exp/runs/E33")
# 名字、实验目录、每次更新的对数、颜色、线型
RUNS = {
    "gpu1": ("E33 单卡（每次更新 1 对）", E33, 1, "k", "-"),
    "gpu4": ("E33 4 卡，未缩放", E33, 4, "0.55", "--"),
    "ema_only": ("4 卡，只改 τ", R, 4, "C0", "-"),
    "sqrt": ("4 卡，τ + 学习率 ×2", R, 4, "C1", "-"),
    "linear": ("4 卡，τ + 学习率 ×4", R, 4, "C2", "-"),
}
D_MAX, IN_MIN, WIN, MAX_PAIRS = 100.0, 0.5, 50, 8000
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


def load_val(root, m):
    """{更新次数: Val 记录}"""
    met = root / "sweep" / m / "S" / "metrics.json"
    if not met.exists():
        return {}
    return {int(k[4:]): v["val"] for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items()
            if k.startswith("step") and "val" in v}


def cmp_row(a, b, va, vb, **key):
    d, ci = paired(va["errors"], vb["errors"])
    return {**key, "a": a, "b": b, "auc5_a": round(va["summary"]["auc@5"], 3), "auc5_b": round(vb["summary"]["auc@5"], 3),
            "diff": round(d, 3), "ci_lo": round(ci[0], 3), "ci_hi": round(ci[1], 3),
            "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)}


def main():
    (R / "extra").mkdir(exist_ok=True)
    upd = {m: load_val(root, m) for m, (_, root, n, *_) in RUNS.items()}
    val = {m: {u * RUNS[m][2]: v for u, v in upd[m].items()} for m in RUNS}     # {图对数: Val 记录}

    rows = [{"run": m, "pairs": p, "updates": p // RUNS[m][2], "auc5": round(val[m][p]["summary"]["auc@5"], 3),
             "median_err": round(val[m][p]["summary"]["median"], 2)}
            for m in RUNS for p in sorted(val[m]) if p <= MAX_PAIRS or m in ("ema_only", "sqrt", "linear")]
    write("val.csv", rows)

    # 1. 各组与单卡按图对数比；2. 组与组比
    vs1, count = [], []
    for m in ("gpu4", "ema_only", "sqrt", "linear"):
        ps = [p for p in sorted(set(val[m]) & set(val["gpu1"])) if p <= MAX_PAIRS]
        rs = [cmp_row(m, "gpu1", val[m][p], val["gpu1"][p], pairs=p) for p in ps]
        vs1 += rs
        if rs:
            count.append({"run": m, "n_ckpt": len(rs), "n_ci_excludes_0": sum(r["ci_excludes_0"] for r in rs),
                          "n_below": sum(r["ci_hi"] < 0 for r in rs), "n_above": sum(r["ci_lo"] > 0 for r in rs),
                          "mean_diff": round(float(np.mean([r["diff"] for r in rs])), 4),
                          "mean_abs_diff": round(float(np.mean([abs(r["diff"]) for r in rs])), 4)})
    if vs1:
        write("vs_gpu1.csv", vs1)
        write("vs_gpu1_count.csv", count)
    pair_cmp = []
    for a, b in (("ema_only", "gpu4"), ("sqrt", "ema_only"), ("linear", "ema_only"), ("linear", "sqrt")):
        for p in sorted(set(val[a]) & set(val[b])):
            if p <= MAX_PAIRS:
                pair_cmp.append(cmp_row(a, b, val[a][p], val[b][p], pairs=p))
    if pair_cmp:
        write("between.csv", pair_cmp)

    # 3. E33 按更新次数对齐
    by_u = []
    for u in (1000, 2000, 3000, 4000):
        u4 = u if u in upd["gpu4"] else max(upd["gpu4"])
        if u in upd["gpu1"]:
            by_u.append(cmp_row("gpu4", "gpu1", upd["gpu4"][u4], upd["gpu1"][u], updates_gpu4=u4, updates_gpu1=u))
    write("e33_by_updates.csv", by_u)

    # 4. 同一个运行里，最后一个存档与最高的存档
    within = []
    for m in RUNS:
        v = {p: x for p, x in val[m].items() if p <= MAX_PAIRS}
        if not v:
            continue
        last, best = max(v), max(v, key=lambda q: v[q]["summary"]["auc@5"])
        within.append({"best_pairs": best, **cmp_row(m, m, v[last], v[best], last_pairs=last)})
    write("last_vs_best.csv", within)

    # 图 1：按图对数对齐的判据量与 Val
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for m, (name, root, n, c, ls) in RUNS.items():
        f = root / "ckpt" / m / "log.jsonl"
        if f.exists():
            log = jl(f)
            pairs = np.array([r["step"] * n for r in log])
            keep = pairs <= MAX_PAIRS
            d_s = roll(np.array([r.get("d_ref", np.nan) for r in log], float), WIN)
            in_s = roll(np.array([r.get("in_frame", np.nan) for r in log], float), WIN)
            ax[0].plot(pairs[keep], d_s[keep], color=c, ls=ls, lw=1.1, label=name)
            ax[1].plot(pairs[keep], in_s[keep], color=c, ls=ls, lw=1.1, label=name)
            h = root / "ckpt" / m / "halt.json"
            halt = json.loads(h.read_text()) if h.exists() else None
            print(f"{m}: 训到 {int(pairs[-1])} 对；提前停 {halt}；d_ref 滑动中位数最大 {np.nanmax(d_s[keep]):.1f}，"
                  f"in_frame 滑动中位数最小 {np.nanmin(in_s[keep]):.3f}")
        ps = [p for p in sorted(val[m]) if p <= MAX_PAIRS]
        if ps:
            ax[2].plot(ps, [val[m][p]["summary"]["auc@5"] for p in ps], "o", color=c, ls=ls, lw=1.1, ms=3, label=name)
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

    # 图 2：E33 按更新次数对齐
    fig, a = plt.subplots(figsize=(6, 4.2))
    for m in ("gpu1", "gpu4"):
        name, _, n, c, ls = RUNS[m]
        us = [u for u in sorted(upd[m]) if u <= 4000]
        a.plot(us, [upd[m][u]["summary"]["auc@5"] for u in us], "o", color=c, ls=ls, ms=3, label=name)
    a.set_xlabel("参数更新次数")
    a.set_ylabel("Val AUC@5")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "e33_by_updates.png", dpi=110)


if __name__ == "__main__":
    main()
