"""E33 表与图（#226）。在 gpfs 上跑（ckpt 不同步到本机）：

    /opt/envs/loftr/bin/python runs/E33/code/summary.py

写 extra/data/*.csv、extra/curves.png，并打印表格。
- 测时（单卡，160）：ckpt/profile/<配置>/plain/profile_plain.json（profile_step.py，60 步、前 10 步不计）；
  一步各部分耗时取 ckpt/profile/base/detail/profile_detail.json（改 _autocast 之前的代码，见 Notes）。
- 梯度核对：ckpt/gradcheck/<配置>/grad_check.json（grad_check.py，8 对图，与 e32ema 存下的梯度比）。
- 多卡短跑：ckpt/ddp/n<卡数>/log.jsonl、ddp_sync.jsonl（ddp_test.sh，100 次更新）；单卡对照为测时的 e32ema。
- 长训练：gpu4（4 卡，每次更新 4 对）、gpu1（单卡，每次更新 1 对）。横轴统一为已训的图对数。判据量为离参照的距离
  d_ref 与画面内格子比例 in_frame 的滑动中位数，窗口 50 次更新（gpu4 为 200 个记录）。
- Val：sweep/<组>/S/metrics.json 各存档 AUC@5；E32 `ema`（gpfs 主 checkout）的 4 个存档一并画上。
  gpu4 与 gpu1 在相同图对数的存档上做配对自助法（同一批 Val 图对有放回重抽 2000 次，同 E32 summary.py）。
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
E32 = Path("/remote-home/xufang/YGC/moon-exp/runs/E32/sweep/ema/S/metrics.json")
RUNS = {"gpu4": ("4 卡（每次更新 4 对）", 4), "gpu1": ("单卡（每次更新 1 对）", 1)}
D_MAX, IN_MIN = 100.0, 0.5
CK = R / "ckpt"
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


def speed():
    rows = []
    for tag, name in (("base", "E32 `ema` 原代码"), ("e32ema", "改 _autocast 后（长训练用的代码）"), ("sdpa", "再加 SDPA"),
                      ("cudnn", "再加 cuDNN 自动调优"), ("foreach", "再加 EMA 参照一次更新全部参数")):
        f = CK / "profile" / tag / "plain" / "profile_plain.json"
        if f.exists():
            p = json.loads(f.read_text())
            rows.append({"config": name, "gpus": 1, "pairs_per_update": 1, "sec_per_update": round(p["step_sec_median"], 3),
                         "sec_per_pair": round(p["step_sec_median"], 3), "gpu_util": p["gpu_util_mean"],
                         "mem_gb": p["mem_gb"], "params_maxdiff": "", "ema_maxdiff": ""})
    one = rows[1]["sec_per_pair"]
    for n in (2, 4):
        d = CK / "ddp" / f"n{n}"
        log, sync = jl(d / "log.jsonl"), jl(d / "ddp_sync.jsonl")[-1]
        sec = float(np.median([r["sec"] for r in log[10:]]))
        rows.append({"config": f"{n} 卡数据并行", "gpus": n, "pairs_per_update": n, "sec_per_update": round(sec, 3),
                     "sec_per_pair": round(sec / n, 3), "gpu_util": "", "mem_gb": log[0].get("mem_gb"),
                     "params_maxdiff": sync["params"], "ema_maxdiff": sync["rloo"]})
    for r in rows:
        r["speedup_vs_e32ema"] = round(one / r["sec_per_pair"], 2)
    write("speed.csv", rows)
    det = json.loads((CK / "profile" / "base" / "detail" / "profile_detail.json").read_text())
    write("step_parts.csv", [{"part": k, "sec": v} for k, v in det["parts"].items()]
          + [{"part": "合计（带逐段同步）", "sec": det["step_sec_median"]}])


def gradcheck():
    rows = []
    for tag, name in (("e32ema_again", "同一份代码再算一遍"), ("sdpa", "SDPA"), ("cudnn", "cuDNN 自动调优")):
        f = CK / "gradcheck" / tag / "grad_check.json"
        if f.exists():
            g = json.loads(f.read_text())
            rows.append({"config": name, "cos_min": min(g["cos"]), "cos_median": float(np.median(g["cos"])),
                         "n_pass_0.99": sum(c >= 0.99 for c in g["cos"]), "n": len(g["cos"]), "cos_sum": g["cos_sum"],
                         "cos_update": g["cos_update"]})
    write("gradcheck.csv", rows)


def short():
    """同种子 200 步短跑（单卡）与 2、4 卡短跑：判据量按已训图对数分段平均。"""
    rows = []
    runs = {"base_a": ("单卡，原代码", 1), "base_b": ("单卡，原代码再跑一遍", 1), "all_a": ("单卡，四个提速开关全开", 1),
            "n2": ("2 卡", 2), "n4": ("4 卡", 4)}
    for k, (name, n) in runs.items():
        d = CK / ("ddp" if k.startswith("n") else "short") / k
        log = jl(d / "log.jsonl")
        pairs = np.array([r["step"] * n for r in log])
        row = {"run": name}
        for key in ("d_ref", "in_frame", "rew"):
            v = np.array([r.get(key, np.nan) for r in log], float)
            for lo, hi in ((0, 100), (100, 200)):
                m = (pairs > lo) & (pairs <= hi)
                row[f"{key}_{lo}-{hi}"] = round(float(np.nanmean(v[m])), 3) if m.any() else ""
        rows.append(row)
    write("short.csv", rows)


def long_runs():
    e32 = {}
    if E32.exists():
        for k, v in json.loads(E32.read_text(encoding="utf-8"))["methods"].items():
            if k.startswith("step") and "val" in v:
                e32[int(k[4:])] = v["val"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    vals, rows = {}, []
    for i, (m, (name, n)) in enumerate(RUNS.items()):
        ck = CK / m
        if not (ck / "log.jsonl").exists():
            continue
        log = jl(ck / "log.jsonl")
        pairs = np.array([r["step"] * n for r in log])
        win = 50       # 日志每次更新一行，多卡为各卡平均；判据窗口 50 次更新
        d_s = roll(np.array([r.get("d_ref", np.nan) for r in log], float), win)
        in_s = roll(np.array([r.get("in_frame", np.nan) for r in log], float), win)
        halt = json.loads((ck / "halt.json").read_text()) if (ck / "halt.json").exists() else None
        met = R / "sweep" / m / "S" / "metrics.json"
        val = {}
        if met.exists():
            for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items():
                if k.startswith("step") and "val" in v:
                    val[int(k[4:]) * n] = v["val"]
        vals[m] = val
        for p in sorted(val):
            rows.append({"run": m, "pairs": p, "updates": p // n, "auc5": round(val[p]["summary"]["auc@5"], 3),
                         "median_err": round(val[p]["summary"]["median"], 2)})
        ax[0].plot(pairs, d_s, color=f"C{i}", lw=1.1, label=name)
        ax[1].plot(pairs, in_s, color=f"C{i}", lw=1.1, label=name)
        if val:
            ps = sorted(val)
            ax[2].plot(ps, [val[p]["summary"]["auc@5"] for p in ps], "o-", color=f"C{i}", lw=1.1, ms=3, label=name)
        if halt is not None:
            for a in ax[:2]:
                a.axvline(halt["step"] * n, color=f"C{i}", ls=":", lw=0.8)
        print(f"{m}: 训到 {int(pairs[-1])} 对；提前停 {halt}；d_ref 滑动中位数最大 {np.nanmax(d_s):.1f}，"
              f"in_frame 滑动中位数最小 {np.nanmin(in_s):.3f}")
    if e32:
        ps = sorted(e32)
        ax[2].plot(ps, [e32[p]["summary"]["auc@5"] for p in ps], "s--", color="0.4", lw=1, ms=3,
                   label="E32 EMA 参照（单卡，余弦调度）")
    if rows:
        write("val.csv", rows)
    if "gpu4" in vals and "gpu1" in vals:
        cmp = []
        for p in sorted(set(vals["gpu4"]) & set(vals["gpu1"])):
            d, ci = paired(vals["gpu4"][p]["errors"], vals["gpu1"][p]["errors"])
            cmp.append({"pairs": p, "auc5_gpu4": round(vals["gpu4"][p]["summary"]["auc@5"], 3),
                        "auc5_gpu1": round(vals["gpu1"][p]["summary"]["auc@5"], 3), "diff": round(d, 3),
                        "ci_lo": round(ci[0], 3), "ci_hi": round(ci[1], 3)})
        if cmp:
            write("gpu4_vs_gpu1.csv", cmp)
    ax[0].axhline(D_MAX, color="0.5", ls="--", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("离参照的距离（px，50 次更新滑动中位数）")
    ax[1].axhline(IN_MIN, color="0.5", ls="--", lw=0.8)
    ax[1].set_ylabel("画面内格子比例（50 次更新滑动中位数）")
    ax[2].set_ylabel("Val AUC@5")
    for a in ax:
        a.set_xlabel("已训图对数")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)


def main():
    (R / "extra").mkdir(exist_ok=True)
    speed()
    gradcheck()
    short()
    long_runs()


if __name__ == "__main__":
    main()
