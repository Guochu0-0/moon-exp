"""E48 表与图（#256）。在 gpfs worktree 里跑（ckpt、日志、raw/ 不同步到本机）：

    /opt/envs/loftr/bin/python runs/E48/code/summary.py

写 extra/data/*.csv、extra/*.png，并打印表格。对照为 E41 main__screen0.8（门槛 0.8，gpfs 主 checkout，不重跑）。
横轴为更新次数（每次 4 对）。
1. Val（val.csv、vs_e41.csv、by_segment.csv）：两个运行各存档的 AUC@5、AUC@3、SR@3、SR@5、误差中位数与评测所用 GPU；
   同一存档位置 E48 减 E41 的 AUC@5 之差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 E41）；
   按训练集遍数（每 1977 次更新）分段的平均。
2. 训练曲线（curves.png、cert_by_segment.csv）：粗级与细化各级可信度损失在被擦处 / 未擦处各自的值（l_cert_e、l_cert_u，
   细化级加 _s）、平均可信度（cert_e、cert_u）、被擦处占画面内格子的比例（era_frac）；离参照的距离、画面内格子比例、
   训练图对上模型自身拟合的内点比例、粗级奖励均值与 E41 叠画；均为 200 次更新滑动中位数。
3. 开销（cost.csv）：每次更新耗时（日志 sec）中位数与总和、第 1 次更新的峰值显存 mem_gb、训练墙钟时间、训练所用 GPU。
4. 擦过的 Val（erased.csv、erased.png）：erased_val.py 的原始结果（raw/erased/*.jsonl）按存档汇总：AUC@5、
   被擦处 / 未擦处平均可信度（逐对平均再对图对平均）、被擦处占画面内的比例；E48 各存档与 E41 最后存档的 AUC@5 差与配对区间。
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
from workbench.protocol import auc  # noqa: E402

Y = Path("/remote-home/xufang/YGC")
E41 = Y / "moon-exp" / "runs" / "E41"
FONT = str(Y / "fonts" / "simhei.ttf")
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
GPU = {"bcb626e99cea": "RTX 3090", "7b3282290ade": "V100", "653cd62be801": "V100"}     # 150、160、126 的容器 hostname
RUNS = {"E48": (R, "main", "被擦处目标为 0（E48）", "C3"), "E41": (E41, "main__screen0.8", "不加（E41 门槛 0.8）", "0.35")}
PASS, UPD = 1977, 7907
LV = (16, 8, 4, 2, 1)
OUT = R / "extra" / "data"


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
    """训练墙钟时间：args.json（开训时写）到最后一个存档的修改时间。"""
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
    for s in sorted(set(val["E48"]) & set(val["E41"])):
        d_, ci = paired(val["E48"][s]["errors"], val["E41"][s]["errors"])
        diff.append({"update": s, "auc5": round(val["E48"][s]["summary"]["auc@5"], 4),
                     "auc5_e41": round(val["E41"][s]["summary"]["auc@5"], 4),
                     "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                     "gpu": val["E48"][s]["gpu"], "gpu_e41": val["E41"][s]["gpu"]})
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
    lg = logs["E48"]
    st = np.array([r["step"] for r in lg])
    get = lambda log, key: np.array([r.get(key, np.nan) for r in log], float)
    fig, ax = plt.subplots(2, 4, figsize=(22, 9))
    for a, sfx, lab in ((ax[0, 0], "", "粗级"), (ax[0, 1], "_1", "细化尺度 1")):
        a.plot(st, roll(get(lg, f"l_cert_e{sfx}"), 200), color="C3", lw=1.1, label="被擦处（目标 0）")
        a.plot(st, roll(get(lg, f"l_cert_u{sfx}"), 200), color="C0", lw=1.1, label="未擦处（目标 1）")
        a.set(ylabel=f"{lab}可信度损失（BCE）", xlabel="更新次数")
    a = ax[0, 2]
    for s_, c in zip(("",) + tuple(f"_{x}" for x in LV), ("k", "C0", "C1", "C2", "C4", "C3")):
        a.plot(st, roll(get(lg, f"cert_e{s_}"), 200), color=c, lw=1.0, label=f"被擦处 {'粗级' if not s_ else '尺度 ' + s_[1:]}")
        a.plot(st, roll(get(lg, f"cert_u{s_}"), 200), color=c, lw=1.0, ls="--")
    a.set(ylabel="平均可信度（实线被擦处，虚线未擦处）", xlabel="更新次数")
    a = ax[0, 3]
    a.plot(st, roll(get(lg, "era_frac"), 200), color="k", lw=1.0, label="粗级")
    a.plot(st, roll(get(lg, "era_frac_1"), 200), color="C3", lw=1.0, label="尺度 1")
    a.set(ylabel="被擦处占画面内格子的比例", xlabel="更新次数")
    for a, key, lab in ((ax[1, 0], "d_ref", "离参照的距离（px）"), (ax[1, 1], "in_frame", "画面内格子比例"),
                        (ax[1, 2], "n_inl", "训练图对上模型自身拟合的内点比例"), (ax[1, 3], "rew", "粗级奖励均值")):
        for k, (_, _, name, c) in RUNS.items():
            log = logs[k]
            v = get(log, key) / (5000.0 if key == "n_inl" else 1.0)
            a.plot([r["step"] for r in log], roll(v, 200), color=c, lw=1.0, label=name)
        a.set(ylabel=lab, xlabel="更新次数")
    for a in ax.flat:
        a.legend(fontsize=7)
        for g in range(1, 4):
            a.axvline(g * PASS, color="0.88", lw=0.8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)

    rows = []
    edges = [0, 250, 500, 1000, 1977, 3954, 5931, UPD]
    keys = ["era_frac", "l_cert_e", "l_cert_u", "cert_e", "cert_u"] + \
        [f"{n}_{s_}" for s_ in LV for n in ("era_frac", "l_cert_e", "l_cert_u", "cert_e", "cert_u")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (st > lo) & (st <= hi)
        if not m.any():
            continue
        row = {"updates": f"{lo + 1}-{hi}", "n": int(m.sum())}
        for key in keys:
            v = get(lg, key)[m]
            row[key] = round(float(np.nanmedian(v)), 4) if np.isfinite(v).any() else None
        rows.append(row)
    write("cert_by_segment.csv", rows)
    # 同一段 E41 的可信度损失（目标是画面内 1、外 0，不分擦没擦），作参考
    l41 = logs["E41"]
    s41 = np.array([r["step"] for r in l41])
    ref = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s41 > lo) & (s41 <= hi)
        if m.any():
            ref.append({"updates": f"{lo + 1}-{hi}", "l_cert": round(float(np.nanmedian(get(l41, "l_cert")[m])), 4),
                        "l_cert_1": round(float(np.nanmedian(get(l41, "l_cert_1")[m])), 4),
                        "l_cert_e48": round(float(np.nanmedian(get(lg, "l_cert")[(st > lo) & (st <= hi)])), 4)
                        if ((st > lo) & (st <= hi)).any() else None,
                        "l_cert_1_e48": round(float(np.nanmedian(get(lg, "l_cert_1")[(st > lo) & (st <= hi)])), 4)
                        if ((st > lo) & (st <= hi)).any() else None})
    write("cert_loss_vs_e41.csv", ref)

    # 3. 开销
    cost = []
    for k, (d, m, _, _) in RUNS.items():
        log = logs[k]
        sec = get(log, "sec")
        cost.append({"run": k, "updates": int(log[-1]["step"]), "train_gpu": train_gpu(d, m),
                     "median_sec": round(float(np.median(sec)), 3), "sec_sum_h": round(float(np.nansum(sec)) / 3600, 2),
                     "mem_gb_step1": log[0].get("mem_gb"), "mem_gb_max": round(float(np.nanmax(get(log, "mem_gb"))), 2),
                     "wall_h": train_hours(d, m)})
    write("cost.csv", cost)

    # Val 差的图
    if diff:
        fig, a_ = plt.subplots(figsize=(9, 4.6))
        x = np.array([r["update"] for r in diff])
        y = np.array([r["diff"] for r in diff])
        a_.errorbar(x, y, yerr=[y - [r["ci_lo"] for r in diff], [r["ci_hi"] for r in diff] - y], fmt="o",
                    color="C3", ms=3, capsize=1.5, lw=0.7)
        a_.axhline(0, color="0.5", ls=":", lw=0.8)
        for g in range(1, 4):
            a_.axvline(g * PASS, color="0.85", lw=0.8)
        a_.set(xlabel="更新次数", ylabel="Val AUC@5 之差（E48 减 E41 门槛 0.8 同位置存档）")
        fig.tight_layout()
        fig.savefig(R / "extra" / "diff.png", dpi=110)

    # 4. 擦过的 Val
    raw = R / "raw" / "erased"
    e41f = raw / "e41_screen0.8_7907.jsonl"
    if e41f.exists():
        base = {r["pair"]: r for r in jl(e41f)}
        out = []
        files = [e41f] + sorted(raw.glob("step*.jsonl"), key=lambda q: int(q.stem[4:]))
        for f in files:
            rs = {r["pair"]: r for r in jl(f)}
            if len(rs) != len(base):
                print(f"skip {f.name}: {len(rs)} / {len(base)} 对")
                continue
            pairs = sorted(base)
            e1 = [rs[p]["err"] for p in pairs]
            row = {"run": "E41" if f == e41f else "E48", "update": 7907 if f == e41f else int(f.stem[4:]),
                   "auc5": round(auc(fix(e1), 5.0), 4),
                   "cert_e": round(float(np.nanmean([np.nan if rs[p]["cert_e"] is None else rs[p]["cert_e"] for p in pairs])), 4),
                   "cert_u": round(float(np.nanmean([np.nan if rs[p]["cert_u"] is None else rs[p]["cert_u"] for p in pairs])), 4),
                   "frac_e": round(float(np.mean([rs[p]["frac_e"] for p in pairs])), 4)}
            if f != e41f:
                d_, ci = paired(e1, [base[p]["err"] for p in pairs])
                row.update(diff_vs_e41_last=round(d_, 4), ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4))
            out.append(row)
        write("erased.csv", out)
        e48 = [r for r in out if r["run"] == "E48"]
        if e48:
            b = out[0]
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
            x = [r["update"] for r in e48]
            ax[0].plot(x, [r["cert_e"] for r in e48], "o-", color="C3", ms=3, label="E48 被擦处")
            ax[0].plot(x, [r["cert_u"] for r in e48], "o-", color="C0", ms=3, label="E48 未擦处")
            ax[0].axhline(b["cert_e"], color="C3", ls=":", lw=1, label="E41 门槛 0.8 最后存档 被擦处")
            ax[0].axhline(b["cert_u"], color="C0", ls=":", lw=1, label="E41 门槛 0.8 最后存档 未擦处")
            ax[0].set(xlabel="更新次数", ylabel="擦过的 Val 上光学一侧的平均可信度")
            ax[1].plot(x, [r["auc5"] for r in e48], "o-", color="C3", ms=3, label="E48")
            ax[1].axhline(b["auc5"], color="0.35", ls=":", lw=1, label="E41 门槛 0.8 最后存档")
            ax[1].set(xlabel="更新次数", ylabel="擦过的 Val 上的 AUC@5")
            for a in ax:
                a.legend(fontsize=8)
                for g in range(1, 4):
                    a.axvline(g * PASS, color="0.88", lw=0.8)
            fig.tight_layout()
            fig.savefig(R / "extra" / "erased.png", dpi=110)


if __name__ == "__main__":
    main()
