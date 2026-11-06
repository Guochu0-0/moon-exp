"""E43 表与图（#250）。在 gpfs worktree 里跑（ckpt、日志不同步到本机）：

    /opt/envs/wb/bin/python runs/E43/code/summary.py

对照为 E41 `main`（同配置不加新项，不重跑），产物从 gpfs 主 checkout 读；E43 从本 worktree 读（没有就读主 checkout）。
写 extra/data/*.csv、extra/*.png，并打印表格。横轴为更新次数（每次 4 对），训练集一遍 = 1977 次更新。
1. Val（val.csv、vs_e41.csv、by_segment.csv）：sweep/<方法>/S/metrics.json 各存档的逐对误差；AUC@5、AUC@3、SR@3、SR@5、
   误差中位数、评测机器的 GPU；同一存档位置 E43 减 E41 的 AUC@5 之差与配对自助法 95% 区间（同一批 Val 图对有放回重抽
   2000 次）；按训练集遍数分段平均。
2. 训练曲线（curves.png）：现有奖励（粗级 rew、细化各级 rew_<s>）与新项（occ_rew_c、occ_rew_<s>）的均值，新项与现有项
   梯度范数之比（occ_g_ratio，每 10 次更新记一次），离参照的距离 d_ref、画面内格子比例 in_frame、自身内点比例
   n_inl ÷ 5000，与 E41 叠画（200 次更新滑动中位数）。
3. 逐像素参照与参照仿射的分歧（divergence.csv、divergence.png）：各级逐像素参照点离参照仿射映射点的距离（每步的均值、
   中位数、90% 分位，按更新次数分段取中位数）。
4. 开销（cost.csv）：每次更新耗时（日志 sec）的中位数与总和、训练墙钟时间、第 1 步的峰值显存（mem_gb）、新项耗时 t_occ。
5. 定系数（probe.csv）：extra/data/probe.json 的逐步梯度范数与比值汇总。
6. 第一次训练（crash.csv、rerun_vs_crash.csv）：gpfs 故障中断的第一次训练（同配置）各存档的 Val 指标与 E41 的差，
   以及重训减第一次在同一存档位置的 AUC@5 差与配对自助法区间，作同配置重复的起伏参考。
7. 遮挡比值（occ.png）：画 code/occ_ratio.py 写的 occ_ratio.csv。
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
from workbench.protocol import auc  # noqa: E402

Y = Path("/remote-home/xufang/YGC")
MAIN = Y / "moon-exp" / "runs"
FONT = str(Y / "fonts" / "simhei.ttf")
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
GPU = {"bcb626e99cea": "RTX 3090", "7b3282290ade": "V100"}     # 150、160 的容器 hostname
PASS, UPD = 1977, 7907
LEVELS = (16, 8, 4, 2, 1)
OUT = R / "extra" / "data"
RUNS = {"E41": ("E41", "main", "E41（不加新项）", "0.4"), "E43": ("E43", "main", "E43（加遮挡约束）", "C3")}


def where(exp, m):
    here = REPO / "runs" / exp
    return here if (here / "ckpt" / m).exists() else MAIN / exp


def jl(path):
    out = []
    for x in Path(path).read_text(encoding="utf-8").splitlines():
        if x.startswith("{"):
            try:
                out.append(json.loads(x))
            except ValueError:
                pass
    return out


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
    f = d / "launch" / f"{m}.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text(encoding="utf-8"))[-1]
    log = d / "ckpt" / m / "log.jsonl"
    t0 = datetime.strptime(rec["start"], "%Y-%m-%d %H:%M:%S %z").timestamp()
    return (log.stat().st_mtime - t0) / 3600 if log.exists() else None


def col(log, k):
    return np.array([r.get(k, np.nan) for r in log], float)


def main():
    (R / "extra").mkdir(exist_ok=True)
    dirs = {k: where(e, m) for k, (e, m, *_) in RUNS.items()}
    val = {k: load_val(dirs[k], RUNS[k][1]) for k in RUNS}
    logs = {k: jl(dirs[k] / "ckpt" / RUNS[k][1] / "log.jsonl") for k in RUNS}

    # 1. Val
    write("val.csv", [{"run": k, "update": s, **{x: round(val[k][s]["summary"][x], 4)
                                                 for x in ("auc@5", "auc@3", "sr@3", "sr@5")},
                       "median_err": round(val[k][s]["summary"]["median"], 2), "gpu": val[k][s]["gpu"]}
                      for k in RUNS for s in sorted(val[k])])
    diff = []
    for s in sorted(set(val["E43"]) & set(val["E41"])):
        d_, ci = paired(val["E43"][s]["errors"], val["E41"][s]["errors"])
        diff.append({"update": s, "auc5": round(val["E43"][s]["summary"]["auc@5"], 4),
                     "auc5_e41": round(val["E41"][s]["summary"]["auc@5"], 4),
                     "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                     "gpu": val["E43"][s]["gpu"], "gpu_e41": val["E41"][s]["gpu"]})
    write("vs_e41.csv", diff)
    bp = []
    for g in range(1, 5):
        rs = [r for r in diff if seg_of(r["update"]) == g]
        if rs:
            bp.append({"segment": g, "n_ckpt": len(rs), "auc5": round(float(np.mean([r["auc5"] for r in rs])), 4),
                       "auc5_e41": round(float(np.mean([r["auc5_e41"] for r in rs])), 4),
                       "mean_diff": round(float(np.mean([r["diff"] for r in rs])), 4),
                       "n_above": sum(r["ci_lo"] > 0 for r in rs), "n_below": sum(r["ci_hi"] < 0 for r in rs)})
    write("by_segment.csv", bp)
    if diff:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for k, (_, _, name, c) in RUNS.items():
            ss = sorted(val[k])
            ax[0].plot(ss, [val[k][s]["summary"]["auc@5"] for s in ss], "o-", ms=3, color=c, label=name)
        u = [r["update"] for r in diff]
        ax[1].errorbar(u, [r["diff"] for r in diff], yerr=[[r["diff"] - r["ci_lo"] for r in diff],
                                                           [r["ci_hi"] - r["diff"] for r in diff]],
                       fmt="o", ms=3, color="C3", capsize=2)
        ax[1].axhline(0, color="0.5", lw=0.8)
        for a in ax:
            for g in range(1, 4):
                a.axvline(g * PASS, color="0.85", lw=0.8)
            a.set_xlabel("更新次数")
        ax[0].set_ylabel("Val AUC@5")
        ax[0].legend(fontsize=8)
        ax[1].set_ylabel("E43 − E41 的 Val AUC@5（同一存档位置，竖线为配对自助法 95% 区间）")
        fig.tight_layout()
        fig.savefig(R / "extra" / "val.png", dpi=110)

    # 2. 训练曲线
    l3, l1 = logs["E43"], logs["E41"]
    if l3:
        fig, ax = plt.subplots(2, 3, figsize=(18, 8.5))
        st3, st1 = col(l3, "step"), col(l1, "step")
        a = ax[0, 0]
        for k, c in [("rew", "k")] + [(f"rew_{s}", f"C{i}") for i, s in enumerate(LEVELS)]:
            a.plot(st3, roll(col(l3, k), 200), color=c, lw=1.1, label=f"E43 {k}")
            a.plot(st1, roll(col(l1, k), 200), color=c, lw=0.8, ls=":")
        a.set(title="现有奖励（实线 E43，点线 E41）", ylabel="均值")
        a = ax[0, 1]
        for k, c in [("occ_rew_c", "k")] + [(f"occ_rew_{s}", f"C{i}") for i, s in enumerate(LEVELS)]:
            a.plot(st3, roll(col(l3, k), 200), color=c, lw=1.1, label=k)
        a.set(title="新项（遮挡版本对逐像素参照）的奖励", ylabel="均值")
        a = ax[0, 2]
        g = col(l3, "occ_g_ratio")
        m = np.isfinite(g)
        a.plot(st3[m], g[m], ".", ms=2, color="0.6")
        a.plot(st3[m], roll(g[m], 20), color="C3", lw=1.2)
        a.axhline(1.7, color="k", ls=":", lw=0.8)
        a.set(title="现有项 ÷ 新项（系数 1 折算）的梯度范数，共用参数", ylabel="比值", yscale="log")
        for a, k, t in ((ax[1, 0], "d_ref", "粗匹配离参照仿射的距离（px）"), (ax[1, 1], "in_frame", "画面内格子比例")):
            a.plot(st3, roll(col(l3, k), 200), color="C3", lw=1.1, label="E43")
            a.plot(st1, roll(col(l1, k), 200), color="0.4", lw=1.1, label="E41")
            a.set(title=t)
        a = ax[1, 2]
        a.plot(st3, roll(col(l3, "n_inl") / 5000, 200), color="C3", lw=1.1, label="E43")
        a.plot(st1, roll(col(l1, "n_inl") / 5000, 200), color="0.4", lw=1.1, label="E41")
        a.set(title="训练图对上当前模型自身的内点比例")
        for a in ax.flat:
            a.set_xlabel("更新次数")
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(R / "extra" / "curves.png", dpi=100)

        # 3. 分歧
        bins = ((0, 250), (250, 1000), (1000, 2000), (2000, 4000), (4000, UPD))
        dv = []
        for name in ["c"] + [str(s) for s in LEVELS]:
            for lo, hi in bins:
                e = (st3 > lo) & (st3 <= hi)
                if not e.any():
                    continue
                dv.append({"level": "粗级" if name == "c" else f"尺度 {name}", "updates": f"{lo + 1}-{hi}",
                           **{q: round(float(np.nanmedian(col(l3, f"occ_{k}_{name}")[e])), 2)
                              for q, k in (("mean_px", "dv"), ("median_px", "dvm"), ("p90_px", "dv90"))}})
        write("divergence.csv", dv)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        for i, name in enumerate(["c"] + [str(s) for s in LEVELS]):
            lab = "粗级" if name == "c" else f"尺度 {name}"
            ax[0].plot(st3, roll(col(l3, f"occ_dvm_{name}"), 200), color=f"C{i}", lw=1.1, label=lab)
            ax[1].plot(st3, roll(col(l3, f"occ_dv90_{name}"), 200), color=f"C{i}", lw=1.1, label=lab)
        ax[0].set(ylabel="逐像素参照离参照仿射的距离，每步中位数（px）", yscale="log")
        ax[1].set(ylabel="同上，每步 90% 分位（px）", yscale="log")
        for a in ax:
            a.set_xlabel("更新次数")
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "divergence.png", dpi=110)

    # 4. 开销
    cost = []
    for k, log in logs.items():
        if not log:
            continue
        sec = col(log, "sec")
        cost.append({"run": k, "updates": int(log[-1]["step"]), "median_sec": round(float(np.nanmedian(sec)), 3),
                     "sec_sum_h": round(float(np.nansum(sec)) / 3600, 2),
                     "wall_h": None if train_hours(dirs[k], RUNS[k][1]) is None else round(train_hours(dirs[k], RUNS[k][1]), 2),
                     "mem_gb_step1": log[0].get("mem_gb"),
                     "median_t_occ": round(float(np.nanmedian(col(log, "t_occ"))), 3) if "t_occ" in log[0] else None})
    write("cost.csv", cost)

    # 5. 定系数
    pf = OUT / "probe.json"
    if pf.exists():
        pr = json.loads(pf.read_text(encoding="utf-8"))
        rows = pr["rows"]
        write("probe.csv", [{"quantity": k, **{q: round(float(np.nanpercentile([r[k] for r in rows if r[k] is not None], p)), 4)
                                               for q, p in (("p25", 25), ("median", 50), ("p75", 75))}}
                            for k in ("occ_g_old", "occ_g_new", "occ_g_ratio", "occ_g_old_all", "occ_g_new_all")])

    # 6. 第一次训练（gpfs 故障中断于约第 2600 次更新，同配置）与重训在同一存档位置的 Val，作同配置重复的起伏参考
    crash()

    # 7. 遮挡版本与原图离逐像素参照的距离（code/occ_ratio.py 写的 occ_ratio.csv）→ occ.png
    f = OUT / "occ_ratio.csv"
    if f.exists():
        rs = list(csv.DictReader(open(f, encoding="utf-8")))
        x = [int(r["ckpt"]) for r in rs]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for a, (k, t) in zip(ax, (("sample", "粗级抽签"), ("mode", "粗级最高分位置"), ("s1", "正常推理尺度 1 的输出"))):
            a.plot(x, [float(r[f"{k}_occ"]) for r in rs], "o-", color="C3", label="遮挡版本")
            a.plot(x, [float(r[f"{k}_orig"]) for r in rs], "o-", color="0.4", label="原图")
            b = a.twinx()
            b.plot(x, [float(r[f"{k}_ratio"]) for r in rs], "s--", color="C0", ms=3, label="比值（右轴）")
            b.set_ylabel("遮挡版本 ÷ 原图")
            a.set(title=t, xlabel="存档（更新次数）", ylabel="离逐像素参照的距离（px，各对中位数）")
            a.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(R / "extra" / "occ.png", dpi=110)


def val_errors(ds, f):
    """一个 Val 预测文件 → 按 ds.labelled('val') 顺序的逐对误差（失败记 ∞，同 workbench eval）。"""
    from workbench.evaluate import pair_error
    preds = {}
    for line in Path(f).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["pair"]] = r.get("A")
    return fix([pair_error(ds, "val", q, preds.get(q)) for q in ds.labelled("val")])


def crash():
    """ckpt/main_gpfs_crash/sweep/S/preds/step<N>/val.jsonl（第一次训练）对 sweep/main/S/preds（重训）与 E41：
    crash.csv（第一次训练各存档的 Val 指标、与 E41 同位置的 AUC@5 差和区间）、rerun_vs_crash.csv（同位置重训减第一次）。"""
    from workbench.dataset import Dataset
    src = R / "ckpt" / "main_gpfs_crash" / "sweep" / "S" / "preds"
    if not src.exists():
        return
    ds = Dataset(Y / "dataset" / "Moon")
    e41 = MAIN / "E41" / "sweep" / "main" / "S" / "preds"
    re_ = R / "sweep" / "main" / "S" / "preds"
    steps = sorted(int(p.name[4:]) for p in src.iterdir() if (p / "val.jsonl").exists())
    rows, rr = [], []
    for s in steps:
        e = val_errors(ds, src / f"step{s}" / "val.jsonl")
        fin = e[np.isfinite(e)]
        row = {"update": s, "auc@5": round(auc(e, 5.0), 4), "auc@3": round(auc(e, 3.0), 4),
               "sr@3": round(float((e <= 3).mean()), 4), "sr@5": round(float((e <= 5).mean()), 4),
               "median_err": round(float(np.median(e)), 2) if len(fin) else None}
        if (e41 / f"step{s}" / "val.jsonl").exists():
            d_, ci = paired(e, val_errors(ds, e41 / f"step{s}" / "val.jsonl"))
            row.update(diff_e41=round(d_, 4), ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4))
        rows.append(row)
        if (re_ / f"step{s}" / "val.jsonl").exists():
            er = val_errors(ds, re_ / f"step{s}" / "val.jsonl")
            d_, ci = paired(er, e)
            rr.append({"update": s, "auc5_rerun": round(auc(er, 5.0), 4), "auc5_first": row["auc@5"],
                       "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4)})
    write("crash.csv", rows)
    write("rerun_vs_crash.csv", rr)
    if rr:
        d = np.array([r["diff"] for r in rr])
        print(f"重训减第一次：{len(d)} 个存档，平均 {d.mean():.4f}，绝对值平均 {np.abs(d).mean():.4f}，"
              f"范围 {d.min():.4f}–{d.max():.4f}")


if __name__ == "__main__":
    main()
