"""E38 表与图（#238）。在 gpfs 上跑（ckpt 不同步到本机）：

    /opt/envs/loftr/bin/python runs/E38/code/summary.py

写 extra/data/*.csv、extra/val_curve.png、extra/diff.png、extra/train_curves.png，并打印表格（写法照 runs/E37/code/summary.py）。
- E38 main 从 E37 all 最后一个存档（31628 对）续训 4 遍；横轴为累计已训图对数 = E37 的 31628 对 + 续训已训图对数。
- Val：E37 读 runs/E37/sweep/all/S/metrics.json，E38 读 sweep/main/S/metrics.json（step0 为续训起点）。
- 比较用配对自助法：同一批 Val 图对有放回重抽 2000 次，取 AUC@5 之差的 2.5% 与 97.5% 分位（同 E33、E36、E37）。
  1. 续训各存档减续训起点（step0）；起点另与 E37 all 最后一个存档比，核对装入；
  2. 按遍数分段（每遍 7907 对）的平均 AUC@5：E37 第 1–4 遍、续训第 1–4 遍（记为第 5–8 遍，不含起点）。
- 训练曲线：离参照的距离 d_ref、画面内格子比例 in_frame，50 次更新的滑动中位数；E37 all 的日志在 gpfs 主 checkout。
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
N, D_MAX, IN_MIN, WIN, EPOCH = 4, 100.0, 0.5, 50, 7907
START = 31628                                  # E37 all 训过的图对数 = 续训起点的横坐标
E37 = R.parent / "E37"
E37_LOG = Path("/remote-home/xufang/YGC/moon-exp/runs/E37/ckpt/all/log.jsonl")
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


def load_val(met, off):
    """{累计已训图对数: Val 记录}"""
    if not met.exists():
        return {}
    return {off + int(k[4:]) * N: v["val"] for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items()
            if k.startswith("step") and "val" in v}


def a5(v):
    return v["summary"]["auc@5"]


def main():
    (R / "extra").mkdir(exist_ok=True)
    old = load_val(E37 / "sweep" / "all" / "S" / "metrics.json", 0)
    new = load_val(R / "sweep" / "main" / "S" / "metrics.json", START)
    write("val.csv", [{"run": run, "pairs": p, "auc5": round(a5(v[p]), 3), "median_err": round(v[p]["summary"]["median"], 2)}
                      for run, v in (("E37", old), ("E38", new)) for p in sorted(v)])
    if START not in new:
        print("续训起点（step0）还没评")
        return
    z = new[START]
    d, ci = paired(z["errors"], old[START]["errors"])
    print(f"起点核对：续训 step0 {a5(z):.4f}，E37 all 最后一个存档 {a5(old[START]):.4f}，差 {d:+.4f}（{ci[0]:+.4f}，{ci[1]:+.4f}）；"
          f"逐对误差相同的比例 {np.mean([a == b for a, b in zip(z['errors'], old[START]['errors'])]):.3f}")
    diff = []
    for p in sorted(new):
        if p == START:
            continue
        d, ci = paired(new[p]["errors"], z["errors"])
        diff.append({"pairs": p, "cont_pairs": p - START, "auc5": round(a5(new[p]), 3), "diff": round(d, 3),
                     "ci_lo": round(ci[0], 3), "ci_hi": round(ci[1], 3)})
    if diff:
        write("vs_start.csv", diff)
    by_pass = []
    for k in range(1, 9):
        v = old if k <= 4 else new
        ps = [p for p in v if (k - 1) * EPOCH < p <= k * EPOCH]
        if ps:
            by_pass.append({"pass": k, "run": "E37" if k <= 4 else "E38", "n_ckpt": len(ps),
                            "auc5_mean": round(float(np.mean([a5(v[p]) for p in ps])), 4),
                            "auc5_max": round(max(a5(v[p]) for p in ps), 3), "auc5_min": round(min(a5(v[p]) for p in ps), 3)})
    write("by_pass.csv", by_pass)

    # 图 1：Val AUC@5，E37 all 与续训接在一起
    fig, a = plt.subplots(figsize=(9, 4.2))
    for v, c, name in ((old, "C3", "E37 五级（种子 0）"), (new, "C2", "续训（种子 1）")):
        p_ = sorted(v)
        a.plot(p_, [a5(v[p]) for p in p_], "o-", color=c, lw=1.1, ms=3, label=name)
    a.axvline(START, color="k", ls="--", lw=0.8)
    a.text(START, a.get_ylim()[0], " 续训起点", va="bottom", fontsize=8)
    for k in range(1, 8):
        if k != 4:
            a.axvline(k * EPOCH, color="0.85", lw=0.8)
    a.set_xlabel("累计已训图对数")
    a.set_ylabel("Val AUC@5")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "val_curve.png", dpi=110)

    # 图 2：续训各存档相对起点之差与 95% 区间
    if diff:
        fig, a = plt.subplots(figsize=(8, 4.2))
        x = [r["cont_pairs"] for r in diff]
        y = np.array([r["diff"] for r in diff])
        a.errorbar(x, y, yerr=[y - [r["ci_lo"] for r in diff], [r["ci_hi"] for r in diff] - y], fmt="o", color="k",
                   ms=3, capsize=2, lw=0.8)
        a.axhline(0, color="0.5", ls=":", lw=0.8)
        for k in range(1, 4):
            a.axvline(k * EPOCH, color="0.85", lw=0.8)
        a.set_xlabel("续训已训图对数")
        a.set_ylabel("Val AUC@5 之差（存档减续训起点）")
        fig.tight_layout()
        fig.savefig(R / "extra" / "diff.png", dpi=110)

    # 图 3：训练中的判据量
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for f, off, c, name in ((E37_LOG, 0, "C3", "E37 五级"), (R / "ckpt" / "main" / "log.jsonl", START, "C2", "续训")):
        if not f.exists():
            continue
        log = jl(f)
        pairs = off + np.array([r["step"] * N for r in log])
        d_s = roll(np.array([r.get("d_ref", np.nan) for r in log], float), WIN)
        in_s = roll(np.array([r.get("in_frame", np.nan) for r in log], float), WIN)
        ax[0].plot(pairs, d_s, color=c, lw=1.0, label=name)
        ax[1].plot(pairs, in_s, color=c, lw=1.0, label=name)
        h = f.parent / "halt.json"
        print(f"{name}: 训到 {int(pairs[-1])} 对；提前停 {json.loads(h.read_text()) if h.exists() else None}；"
              f"d_ref 滑动中位数 最大 {np.nanmax(d_s):.2f}、最后 {d_s[-1]:.2f}，in_frame 滑动中位数 最小 {np.nanmin(in_s):.3f}、"
              f"最后 {in_s[-1]:.3f}；每次更新 {np.median([r['sec'] for r in log if 'sec' in r]):.2f} s")
    ax[0].axhline(D_MAX, color="0.5", ls=":", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("离参照的距离（px，50 次更新滑动中位数）")
    ax[1].axhline(IN_MIN, color="0.5", ls=":", lw=0.8)
    ax[1].set_ylabel("画面内格子比例（50 次更新滑动中位数）")
    for a in ax:
        a.axvline(START, color="k", ls="--", lw=0.8)
        a.set_xlabel("累计已训图对数")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "train_curves.png", dpi=110)


if __name__ == "__main__":
    main()
