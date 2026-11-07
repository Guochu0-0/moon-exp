"""E46 表与图（#254）。在 gpfs worktree 里跑（ckpt、日志不同步到本机）：

    /opt/envs/loftr/bin/python runs/E46/code/summary.py

写 extra/data/*.csv、extra/*.png，并打印表格。对照为 E41 main__screen0.8（门槛 0.8，gpfs 主 checkout，不重跑）。横轴为更新次数（每次 4 对）。
1. Val（val.csv、vs_e41.csv、by_segment.csv、val_fine.png）：两个运行各存档的 AUC@5、AUC@3、SR@3、SR@5、误差中位数与评测所用 GPU；
   同一存档位置 E46 减 E41 的 AUC@5 之差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 E41）；
   按训练集遍数（每 1977 次更新）分段的平均；误差中位数与 AUC@3 逐存档的图。
2. 训练曲线（curves.png）：现有项与新项各自的奖励均值（粗级 rew / sp_rew，细化尺度 16 与 1），新项与现有项梯度之比
   （sp_ratio，实际施加；sp_coef，让两者 1:1 的系数），离参照的距离、画面内格子比例、内点比例与 E41 叠画；200 次更新滑动中位数。
3. 自配对上的奖励（selfpair_train.csv）：按更新段，自配对粗级与细化各级奖励均值、组内标准差、离真值的距离；
   与原图对（现有项）同一量对比。
4. 开销（cost.csv）：每次更新耗时（日志 sec）中位数与总和、新项耗时 t_sp 中位数、第 1 次更新的峰值显存 mem_gb、
   训练墙钟时间、训练所用 GPU。
5. 同模态自配对的准确度（selfpair_eval.csv、selfpair_eval.png）：code/selfpair_eval.py 的结果，每个模型分 SAR 对 SAR、
   光学对光学的配准误差中位数、AUC@5、SR@3。
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
RUNS = {"E46": (R, "加自配对约束奖励（E46）", "C3"), "E41": (E41, "不加（E41 门槛 0.8）", "0.35")}
E41M = {"E46": "main", "E41": "main__screen0.8"}
PASS, UPD = 1977, 7907
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


def selfpair_eval():
    d = R / "extra" / "data" / "selfpair"
    if not d.exists():
        return []
    order = lambda n: (0, 0) if n == "zeroshot" else (1, 0) if n.startswith("e41") else (2, int(n.split("_")[1]))
    rows = []
    for fp in sorted(d.glob("*.jsonl"), key=lambda q: order(q.stem)):
        rs = jl(fp)
        for mod in ("sar", "opt"):
            e = np.array([r["err"] for r in rs if r["mod"] == mod], float)
            if len(e):
                rows.append({"model": fp.stem, "mod": mod, "n": len(e), "median_err": round(float(np.median(e)), 3),
                             "auc@5": round(auc(e, 5.0), 4), "sr@3": round(float((e <= 3).mean()), 4),
                             "fail": int((~np.isfinite(e)).sum())})
    return rows


def main():
    (R / "extra").mkdir(exist_ok=True)
    val = {k: load_val(d, E41M[k]) for k, (d, _, _) in RUNS.items()}
    logs = {k: jl(d / "ckpt" / E41M[k] / "log.jsonl") for k, (d, _, _) in RUNS.items()}

    # 1. Val
    write("val.csv", [{"run": k, "update": s, **{x: round(val[k][s]["summary"][x], 3)
                                                  for x in ("auc@5", "auc@3", "sr@3", "sr@5")},
                       "median_err": round(val[k][s]["summary"]["median"], 2), "gpu": val[k][s]["gpu"]}
                      for k in RUNS for s in sorted(val[k])])
    diff = []
    for s in sorted(set(val["E46"]) & set(val["E41"])):
        d_, ci = paired(val["E46"][s]["errors"], val["E41"][s]["errors"])
        diff.append({"update": s, "auc5": round(val["E46"][s]["summary"]["auc@5"], 4),
                     "auc5_e41": round(val["E41"][s]["summary"]["auc@5"], 4),
                     "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                     "gpu": val["E46"][s]["gpu"], "gpu_e41": val["E41"][s]["gpu"]})
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
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, key, lab in ((ax[0], "median", "Val 配准误差中位数（px）"), (ax[1], "auc@3", "Val AUC@3")):
        for k, (_, name, c) in RUNS.items():
            ss = sorted(val[k])
            a.plot(ss, [val[k][s]["summary"][key] for s in ss], "o-", color=c, ms=3, lw=1.0, label=name)
        a.set(xlabel="更新次数", ylabel=lab)
        a.legend(fontsize=8)
        for g in range(1, 4):
            a.axvline(g * PASS, color="0.88", lw=0.8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "val_fine.png", dpi=110)

    # 2. 训练曲线
    lg = logs["E46"]
    st = np.array([r["step"] for r in lg])
    get = lambda log, key: np.array([r.get(key, np.nan) for r in log], float)
    cal = OUT / "calib.json"
    w_sp = json.loads(cal.read_text(encoding="utf-8"))["selfpair_w"] if cal.exists() else None
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    a = ax[0, 0]
    for key, lab, c in (("rew", "现有项（原图对，对参照）", "C0"), ("sp_rew", "新项（自配对，对真值）", "C3")):
        a.plot(st, roll(get(lg, key), 200), color=c, lw=1.1, label=lab)
    a.set(ylabel="粗级奖励均值", xlabel="更新次数")
    a = ax[0, 1]
    for s_, ls in ((16, "-"), (1, "--")):
        a.plot(st, roll(get(lg, f"rew_{s_}"), 200), color="C0", ls=ls, lw=1.1, label=f"现有项 尺度 {s_}")
        a.plot(st, roll(get(lg, f"sp_rew_{s_}"), 200), color="C3", ls=ls, lw=1.1, label=f"新项 尺度 {s_}")
    a.set(ylabel="细化抽签奖励均值", xlabel="更新次数")
    a = ax[0, 2]
    a.plot(st, roll(get(lg, "sp_ratio"), 200), color="C3", lw=1.1, label=f"新项 ÷ 现有项（实际施加，系数 {w_sp}）")
    a.plot(st, roll(get(lg, "sp_coef"), 200), color="C2", lw=1.1, label="让两者 1:1 的系数")
    a.axhline(1.0, color="0.6", ls=":", lw=0.8)
    a.set_yscale("log")
    a.set(ylabel="共用参数上的梯度范数之比", xlabel="更新次数")
    for a, key, lab in ((ax[1, 0], "d_ref", "离参照的距离（px）"), (ax[1, 1], "in_frame", "画面内格子比例"),
                        (ax[1, 2], "n_inl", "训练图对上模型自身拟合的内点比例")):
        for k, (_, name, c) in RUNS.items():
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

    # 3. 自配对上的奖励（与原图对的同一量对比）
    rows = []
    edges = [0, 50, 250, 1000, 1977, 3954, 5931, UPD]
    keys = ["sp_rew", "sp_rew_sd", "rew", "rew_sd", "sp_d", "sp_dmode", "d_ref"]
    keys += [f"{p}_{s_}" for s_ in (16, 8, 4, 2, 1) for p in ("sp_rew", "sp_rew_sd", "rew", "rew_sd", "sp_dmu", "dmu")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (st > lo) & (st <= hi)
        if not m.any():
            continue
        row = {"updates": f"{lo + 1}-{hi}", "n": int(m.sum())}
        for key in keys:
            row[key] = round(float(np.nanmedian(get(lg, key)[m])), 4)
        rows.append(row)
    write("selfpair_train.csv", rows)

    # 4. 开销
    cost = []
    for k, (d, _, _) in RUNS.items():
        log = logs[k]
        sec = get(log, "sec")
        cost.append({"run": k, "updates": int(log[-1]["step"]), "train_gpu": train_gpu(d, E41M[k]),
                     "median_sec": round(float(np.median(sec)), 3), "sec_sum_h": round(float(np.nansum(sec)) / 3600, 2),
                     "median_t_sp": round(float(np.nanmedian(get(log, "t_sp"))), 3) if k == "E46" else None,
                     "mem_gb_step1": log[0].get("mem_gb"), "wall_h": train_hours(d, E41M[k])})
    write("cost.csv", cost)

    # 5. 同模态自配对的准确度
    se = selfpair_eval()
    write("selfpair_eval.csv", se)
    if se:
        fig, a_ = plt.subplots(figsize=(9, 4.6))
        for mod, c, lab in (("sar", "C1", "SAR 对 SAR"), ("opt", "C2", "光学对光学")):
            tr = [r for r in se if r["mod"] == mod and r["model"].startswith("e46_")]
            a_.plot([int(r["model"].split("_")[1]) for r in tr], [r["median_err"] for r in tr], "o-", color=c, ms=3,
                    lw=1.0, label=f"本次 {lab}")
            for r in se:
                if r["mod"] == mod and not r["model"].startswith("e46_"):
                    zs = r["model"] == "zeroshot"
                    a_.axhline(r["median_err"], color=c, ls=":" if zs else "--", lw=0.9,
                               label=("zero-shot " if zs else "E41 门槛 0.8 最后存档 ") + lab)
        a_.set(xlabel="更新次数", ylabel="自配对配准误差中位数（px）")
        a_.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "selfpair_eval.png", dpi=110)

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
        a_.set(xlabel="更新次数", ylabel="Val AUC@5 之差（E46 减 E41 门槛 0.8 同位置存档）")
        fig.tight_layout()
        fig.savefig(R / "extra" / "diff.png", dpi=110)


if __name__ == "__main__":
    main()
