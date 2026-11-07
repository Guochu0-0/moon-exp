"""E47 表与图（#255，照 E42 code/summary.py）。在 gpfs worktree 里跑（ckpt、日志不同步到本机）：

    /opt/envs/loftr/bin/python runs/E47/code/summary.py

写 extra/data/*.csv、extra/*.png，并打印表格。对照为 E41 main__screen0.8（门槛 0.8，gpfs 主 checkout，不重跑）。横轴为更新次数（每次 4 对）。
1. Val（val.csv、vs_e41.csv、by_segment.csv）：两个运行各存档的 AUC@5、AUC@3、SR@3、SR@5、误差中位数与评测所用 GPU；
   同一存档位置 E47 减 E41 的 AUC@5 之差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 E41）；按训练集遍数分段的平均。
2. 训练曲线（curves.png）：现有项与两项互逆奖励的均值（粗级、细化尺度 16 与 1）、新项与现有项梯度之比（cyc_ratio 实际施加、
   cyc_coef 让两者 1:1 的系数）、训练中正向参照与反向参照复合后的残差 cyc_res，以及离参照的距离、画面内格子比例、
   内点比例与 E41 叠画；均为 200 次更新滑动中位数。cyc_by_segment.csv 为这些量按更新段的中位数。
3. Val 上的正反向（swap.csv、swap.png）：反向推理（code/swap_queue.py，extra/data/swap/<名>.jsonl）取逆后的配准误差
   （反向本身的准确度：AUC@5、误差中位数）；正反向互逆残差：每对图正向预测的仿射 A（光学 → SAR，同一存档的 Val 预测）
   与反向推理拟合的仿射 B（SAR → 光学）复合 B∘A，在光学图 32 × 32 个格心（16 px 一格）上离原位的平均位移（px），
   报各存档的中位数与均值（任一方向拟合失败的对另计数）。E41 只剩第 7907 次的存档，只有这一点。
4. 开销（cost.csv）：每次更新耗时中位数与总和、新项耗时 t_cyc 中位数、第 1 次更新的峰值显存、训练墙钟时间、训练所用 GPU。
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
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
from workbench.dataset import Dataset  # noqa: E402
from workbench.evaluate import pair_error  # noqa: E402
from workbench.protocol import auc, summarize  # noqa: E402

Y = Path("/remote-home/xufang/YGC")
E41 = Y / "moon-exp" / "runs" / "E41"
FONT = str(Y / "fonts" / "simhei.ttf")
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
GPU = {"bcb626e99cea": "RTX 3090", "7b3282290ade": "V100"}     # 150、160 的容器 hostname
RUNS = {"E47": (R, "main", "加正反向互逆约束奖励（E47）", "C3"),
        "E41": (E41, "main__screen0.8", "不加（E41 门槛 0.8）", "0.35")}
PASS, UPD = 1977, 7907
OUT = R / "extra" / "data"
GRID = np.stack(np.meshgrid(np.arange(32) * 16 + 8.0, np.arange(32) * 16 + 8.0), -1).reshape(-1, 2)


def jl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.startswith("{")]


def roll(x, win, f=np.nanmedian):
    x = np.asarray(x, float)
    return np.array([f(x[max(0, i - win + 1):i + 1]) if np.isfinite(x[max(0, i - win + 1):i + 1]).any() else np.nan
                     for i in range(len(x))])


def fix(e):
    return np.array([np.inf if x is None else x for x in e], float)


def paired(e1, e0, n=2000, seed=0):
    e1, e0 = fix(e1), fix(e0)
    idx = np.random.default_rng(seed).integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return auc(e1, 5.0) - auc(e0, 5.0), np.percentile(diff, [2.5, 97.5])


def write(name, rows):
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"== {name}")
    for r in rows:
        print(" | ".join(f"{k}={r[k]}" for k in r))


def seg_of(s):
    return min(4, (s - 1) // PASS + 1)


def load_val(d, m):
    met = d / "sweep" / m / "S" / "metrics.json"
    if not met.exists():
        return {}
    out = {}
    for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items():
        if k.startswith("step") and "val" in v:
            meta = d / "sweep" / m / "S" / "preds" / k / "val.meta.json"
            host = json.loads(meta.read_text(encoding="utf-8")).get("host") if meta.exists() else None
            out[int(k[4:])] = {**v["val"], "gpu": GPU.get(host, host)}
    return out


def train_hours(d, m):
    ck = d / "ckpt" / m
    last = max(ck.glob("ckpt_*.pt"), key=lambda q: int(q.stem[5:]), default=None)
    if last is None or not (ck / "args.json").exists():
        return None
    return round((last.stat().st_mtime - (ck / "args.json").stat().st_mtime) / 3600, 2)


def train_gpu(d, m):
    f = d / "launch" / f"{m}.json"
    if not f.exists():
        return None
    host = json.loads(f.read_text(encoding="utf-8"))[-1].get("host")
    return GPU.get(host, host)


def h3(A):
    return np.vstack([np.asarray(A, float), [0, 0, 1]])


def swap_rows(ds):
    """每个有反向推理结果的存档一行：反向准确度与正反向互逆残差。"""
    rows = []
    for f in sorted((OUT / "swap").glob("*.jsonl"), key=lambda p: (p.stem.split("_")[0], int(p.stem.split("step")[1]))):
        run, st = ("E41", "main__screen0.8") if f.stem.startswith("E41") else ("E47", "main")
        step = int(f.stem.split("step")[1])
        fwd_f = RUNS[run][0] / "sweep" / st / "S" / "preds" / f"step{step}" / "val.jsonl"
        if not fwd_f.exists():
            continue
        fwd = {r["pair"]: r.get("A") for r in jl(fwd_f)}
        rev = {r["pair"]: r.get("A") for r in jl(f)}
        pairs = ds.labelled("val")
        err = fix([pair_error(ds, "val", p, rev.get(p)) for p in pairs])
        res, fail = [], 0
        for p in pairs:
            A, Ainv = fwd.get(p), rev.get(p)      # rev 存的是反向仿射 B 求逆后的「光学 → SAR」
            if A is None or Ainv is None:
                fail += 1
                continue
            B = np.linalg.inv(h3(Ainv))
            x = np.c_[GRID, np.ones(len(GRID))] @ (B @ h3(A)).T
            res.append(float(np.linalg.norm(x[:, :2] - GRID, axis=1).mean()))
        s = summarize(err)
        rows.append({"run": run, "update": step, "rev_auc@5": round(s["auc@5"], 4), "rev_median_err": round(s["median"], 2),
                     "fwd_auc@5": round(summarize(fix([pair_error(ds, "val", p, fwd.get(p)) for p in pairs]))["auc@5"], 4),
                     "cyc_res_median": round(float(np.median(res)), 3), "cyc_res_mean": round(float(np.mean(res)), 3),
                     "n_fail": fail})
    return rows


def main():
    (R / "extra").mkdir(exist_ok=True)
    val = {k: load_val(d, m) for k, (d, m, _, _) in RUNS.items()}
    logs = {k: jl(d / "ckpt" / m / "log.jsonl") for k, (d, m, _, _) in RUNS.items()}

    # 1. Val
    write("val.csv", [{"run": k, "update": s, **{x: round(val[k][s]["summary"][x], 3)
                                                  for x in ("auc@5", "auc@3", "sr@3", "sr@5")},
                       "median_err": round(val[k][s]["summary"]["median"], 2), "gpu": val[k][s]["gpu"]}
                      for k in RUNS for s in sorted(val[k])])
    diff = []
    for s in sorted(set(val["E47"]) & set(val["E41"])):
        d_, ci = paired(val["E47"][s]["errors"], val["E41"][s]["errors"])
        diff.append({"update": s, "auc5": round(val["E47"][s]["summary"]["auc@5"], 4),
                     "auc5_e41": round(val["E41"][s]["summary"]["auc@5"], 4),
                     "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                     "gpu": val["E47"][s]["gpu"], "gpu_e41": val["E41"][s]["gpu"]})
    write("vs_e41.csv", diff)
    bp = []
    for g in range(1, 5):
        rs = [r for r in diff if seg_of(r["update"]) == g]
        if rs:
            bp.append({"segment": g, "n_ckpt": len(rs), "auc5": round(float(np.mean([r["auc5"] for r in rs])), 4),
                       "auc5_e41": round(float(np.mean([r["auc5_e41"] for r in rs])), 4),
                       "mean_diff": round(float(np.mean([r["diff"] for r in rs])), 4),
                       "n_lower": sum(r["diff"] < 0 for r in rs),
                       "n_above_ci": sum(r["ci_lo"] > 0 for r in rs), "n_below_ci": sum(r["ci_hi"] < 0 for r in rs)})
    write("by_segment.csv", bp)

    # 2. 训练曲线
    lg = logs["E47"]
    st = np.array([r["step"] for r in lg])
    get = lambda log, key: np.array([r.get(key, np.nan) for r in log], float)
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    a = ax[0, 0]
    for key, lab, c in (("rew", "现有项", "C0"), ("cyc_rew_f", "正向互逆项", "C3"), ("cyc_rew_r", "反向互逆项", "C1")):
        a.plot(st, roll(get(lg, key), 200), color=c, lw=1.1, label=lab)
    a.set(ylabel="粗级奖励均值", xlabel="更新次数")
    a = ax[0, 1]
    for s_, ls in ((16, "-"), (1, "--")):
        a.plot(st, roll(get(lg, f"rew_{s_}"), 200), color="C0", ls=ls, lw=1.1, label=f"现有项 尺度 {s_}")
        a.plot(st, roll(get(lg, f"cyc_rew_f_{s_}"), 200), color="C3", ls=ls, lw=1.1, label=f"正向互逆项 尺度 {s_}")
        a.plot(st, roll(get(lg, f"cyc_rew_r_{s_}"), 200), color="C1", ls=ls, lw=1.1, label=f"反向互逆项 尺度 {s_}")
    a.set(ylabel="细化抽签奖励均值", xlabel="更新次数")
    a = ax[0, 2]
    a.plot(st, roll(get(lg, "cyc_ratio"), 200), color="C3", lw=1.1, label="新项 ÷ 现有项（实际施加，系数 0.61）")
    a.plot(st, roll(get(lg, "cyc_coef"), 200), color="C2", lw=1.1, label="让两者 1:1 的系数")
    a.axhline(1.0, color="0.6", ls=":", lw=0.8)
    a.set(ylabel="共用参数上的梯度范数之比", xlabel="更新次数")
    a = ax[1, 0]
    a.plot(st, roll(get(lg, "cyc_res"), 200), color="C3", lw=1.1, label="正向参照与反向参照复合后的残差")
    a.plot(st, roll(get(lg, "d_ref"), 200), color="C0", lw=1.0, label="离参照的距离（E47）")
    a.plot([r["step"] for r in logs["E41"]], roll(get(logs["E41"], "d_ref"), 200), color="0.35", lw=1.0,
           label="离参照的距离（E41 门槛 0.8）")
    a.set_yscale("log")
    a.set(ylabel="px", xlabel="更新次数")
    for a, key, lab in ((ax[1, 1], "in_frame", "画面内格子比例"), (ax[1, 2], "n_inl", "训练图对上模型自身拟合的内点比例")):
        for k, (_, _, name, c) in RUNS.items():
            log = logs[k]
            v = get(log, key) / (5000.0 if key == "n_inl" else 1.0)
            a.plot([r["step"] for r in log], roll(v, 200), color=c, lw=1.0, label=name)
        a.set(ylabel=lab, xlabel="更新次数")
    for a in ax.flat:
        a.legend(fontsize=8)
        for g in range(1, 4):
            a.axvline(g * PASS, color="0.88", lw=0.8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)
    rows = []
    edges = [0, 250, 500, 1000, 1977, 3954, 5931, UPD]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (st > lo) & (st <= hi)
        if not m.any():
            continue
        row = {"updates": f"{lo + 1}-{hi}", "n": int(m.sum())}
        for key in ("rew", "cyc_rew_f", "cyc_rew_r", "cyc_res", "cyc_ratio", "cyc_coef", "cyc_ok", "d_ref"):
            row[key] = round(float(np.nanmedian(get(lg, key)[m])), 3)
        rows.append(row)
    write("cyc_by_segment.csv", rows)

    # 3. Val 上的正反向
    sw = swap_rows(Dataset(Y / "dataset" / "Moon"))
    write("swap.csv", sw)
    if sw:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for k, (_, _, name, c) in RUNS.items():
            rs = [r for r in sw if r["run"] == k]
            x = [r["update"] for r in rs]
            ax[0].plot(x, [r["cyc_res_median"] for r in rs], "o-", color=c, ms=3, lw=1, label=name)
            ax[1].plot(x, [r["rev_auc@5"] for r in rs], "o-", color=c, ms=3, lw=1, label=f"{name} 反向")
            ax[1].plot(x, [r["fwd_auc@5"] for r in rs], "s--", color=c, ms=3, lw=0.8, alpha=0.6, label=f"{name} 正向")
        ax[0].set(xlabel="更新次数", ylabel="Val 正反向互逆残差中位数（px，格心平均位移）")
        ax[1].set(xlabel="更新次数", ylabel="Val AUC@5")
        for a in ax:
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "swap.png", dpi=110)

    # 4. 开销
    cost = []
    for k, (d, m, _, _) in RUNS.items():
        log = logs[k]
        sec = get(log, "sec")
        cost.append({"run": k, "updates": int(log[-1]["step"]), "train_gpu": train_gpu(d, m),
                     "median_sec": round(float(np.median(sec)), 3), "sec_sum_h": round(float(np.nansum(sec)) / 3600, 2),
                     "median_t_cyc": round(float(np.nanmedian(get(log, "t_cyc"))), 3) if k == "E47" else None,
                     "mem_gb_step1": log[0].get("mem_gb"), "wall_h": train_hours(d, m)})
    write("cost.csv", cost)

    if diff:
        fig, a_ = plt.subplots(figsize=(9, 4.6))
        x = np.array([r["update"] for r in diff])
        y = np.array([r["diff"] for r in diff])
        a_.errorbar(x, y, yerr=[y - [r["ci_lo"] for r in diff], [r["ci_hi"] for r in diff] - y], fmt="o",
                    color="C3", ms=3, capsize=1.5, lw=0.7)
        a_.axhline(0, color="0.5", ls=":", lw=0.8)
        for g in range(1, 4):
            a_.axvline(g * PASS, color="0.85", lw=0.8)
        a_.set(xlabel="更新次数", ylabel="Val AUC@5 之差（E47 减 E41 门槛 0.8 同位置存档）")
        fig.tight_layout()
        fig.savefig(R / "extra" / "diff.png", dpi=110)


if __name__ == "__main__":
    main()
