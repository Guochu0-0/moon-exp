"""E44 表与图（#252）。在 gpfs worktree 里跑（ckpt、日志不同步到本机）：

    /opt/envs/loftr/bin/python runs/E44/code/summary.py

写 extra/data/*.csv、extra/*.png，并打印表格。对照为 E41 门槛 0.8（main__screen0.8，gpfs 主 checkout，不重跑）。
横轴为更新次数（每次 4 对）。本次训练在第 327 次更新被坍缩判据停下，只有第 250 次更新一个存档。
1. Val（val.csv、vs_e41.csv）：E44 的存档与 E41 门槛 0.8 第 250、500 次更新存档的 AUC@5、AUC@3、SR@3、SR@5、误差中位数；
   同一存档位置 E44 减 E41 的 AUC@5 之差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 E41）。
2. 训练曲线（curves.png）：粗级奖励均值、离参照的距离、画面内格子比例、训练图对上模型自身拟合的内点比例与 E41 叠画；
   新项的组内 NCC 均值与记 −1 的比例、两项梯度之比（sim_ratio 实际施加；sim_coef 让两者 1:1 的系数）；20 次更新滑动中位数。
3. 新项诊断（diag_by_segment.csv）：每 50 次更新一段，组内 NCC 均值、记 −1 的比例、新项优势的组内标准差、
   NCC 最高与离参照最近不是同一份的比例（sim_diff）、Val 图对上组内 NCC 与真实误差的 Spearman 秩相关（sim_rho，
   每 50 次更新每卡一对、4 卡平均后一个数）、两项梯度之比；前 50 次更新另取定系数那次（只量不学）的同一组量。
4. 开销（cost.csv）：第 101–327 次更新（避开开训时样本筛选大量补抽的阶段）每次更新耗时（日志 sec）中位数、
   新项耗时 t_sim 中位数、第 1 次更新的峰值显存 mem_gb。
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
GPU = {"bcb626e99cea": "RTX 3090", "7b3282290ade": "V100"}     # 150、160 的容器 hostname
RUNS = {"E44": (R, "main", "加图像相似度约束（E44）", "C3"),
        "E41": (E41, "main__screen0.8", "不加（E41 门槛 0.8）", "0.35")}
OUT = R / "extra" / "data"
WIN = 20


def jl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def roll(x, win):
    return np.array([np.nanmedian(x[max(0, i - win + 1):i + 1]) for i in range(len(x))])


def fix(e):
    return np.array([np.inf if x is None else x for x in e], float)


def paired(e1, e0, n=2000, seed=0):
    e1, e0 = fix(e1), fix(e0)
    idx = np.random.default_rng(seed).integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return float(auc(e1, 5.0) - auc(e0, 5.0)), np.percentile(diff, [2.5, 97.5])


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"== {name}")
    for r in rows:
        print(r)


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


def main():
    (R / "extra").mkdir(exist_ok=True)
    val = {k: load_val(d, m) for k, (d, m, _, _) in RUNS.items()}
    logs = {k: jl(d / "ckpt" / m / "log.jsonl") for k, (d, m, _, _) in RUNS.items()}
    calib = jl(R / "ckpt" / "_calib" / "log.jsonl")
    get = lambda log, key: np.array([r.get(key, np.nan) for r in log], float)

    # 1. Val
    write("val.csv", [{"run": k, "update": s, **{x: round(val[k][s]["summary"][x], 4)
                                                  for x in ("auc@5", "auc@3", "sr@3", "sr@5")},
                       "median_err": round(val[k][s]["summary"]["median"], 2), "gpu": val[k][s]["gpu"]}
                      for k in RUNS for s in sorted(val[k]) if s <= 500])
    diff = []
    for s in sorted(set(val["E44"]) & set(val["E41"])):
        d_, ci = paired(val["E44"][s]["errors"], val["E41"][s]["errors"])
        diff.append({"update": s, "auc5": round(val["E44"][s]["summary"]["auc@5"], 4),
                     "auc5_e41": round(val["E41"][s]["summary"]["auc@5"], 4),
                     "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                     "gpu": val["E44"][s]["gpu"], "gpu_e41": val["E41"][s]["gpu"]})
    write("vs_e41.csv", diff)

    # 2. 训练曲线
    lg = logs["E44"]
    last = int(lg[-1]["step"])
    st = np.array([r["step"] for r in lg])
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for a, key, lab in ((ax[0, 0], "rew", "粗级奖励均值（现有项）"), (ax[0, 1], "d_ref", "离参照的距离（px）"),
                        (ax[1, 0], "in_frame", "画面内格子比例"), (ax[1, 1], "n_inl", "训练图对上模型自身拟合的内点比例")):
        for k, (_, _, name, c) in RUNS.items():
            log = [r for r in logs[k] if r["step"] <= last + 50]
            v = get(log, key) / (5000.0 if key == "n_inl" else 1.0)
            a.plot([r["step"] for r in log], roll(v, WIN), color=c, lw=1.0, label=name)
        a.set(ylabel=lab, xlabel="更新次数")
    ax[0, 1].set_yscale("log")
    a = ax[0, 2]
    a.plot(st, roll(get(lg, "sim_ncc"), WIN), color="C3", lw=1.1, label="组内 NCC 均值")
    a.plot(st, roll(get(lg, "sim_fail"), WIN), color="C1", lw=1.1, label="记 −1 的样本比例")
    a.set(ylabel="新项（训练图对）", xlabel="更新次数")
    a = ax[1, 2]
    a.plot(st, roll(get(lg, "sim_ratio"), WIN), color="C3", lw=1.1, label="新项 × 190 ÷ 现有项（实际施加）")
    a.plot(st, roll(get(lg, "sim_coef"), WIN) / 190.0, color="C2", lw=1.1, label="让两者 1:1 的系数 ÷ 190")
    a.axhline(1.0, color="0.6", ls=":", lw=0.8)
    a.set_yscale("log")
    a.set(ylabel="共用参数上的梯度范数之比", xlabel="更新次数")
    for a in ax.flat:
        a.legend(fontsize=8)
        a.axvline(250, color="0.85", lw=0.8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)

    # 3. 新项诊断
    keys = ("sim_ncc", "sim_fail", "sim_adv_sd", "sim_diff", "sim_rho", "sim_ratio", "sim_coef", "d_ref", "rew")
    rows = []
    for name, log, lo, hi in [("定系数（只量不学）", calib, 0, 50)] + \
            [("正式训练", lg, a_, min(a_ + 50, last)) for a_ in range(0, last, 50)]:
        sel = [r for r in log if lo < r["step"] <= hi]
        row = {"run": name, "updates": f"{lo + 1}-{hi}"}
        for k in keys:
            v = get(sel, k)
            row[k] = round(float(np.nanmean(v)), 3) if np.isfinite(v).any() else None
        row["n_rho"] = int(np.isfinite(get(sel, "sim_rho")).sum())
        rows.append(row)
    write("diag_by_segment.csv", rows)

    # 4. 开销
    cost = []
    for k in RUNS:
        sel = [r for r in logs[k] if 100 < r["step"] <= last]
        cost.append({"run": k, "updates": f"101-{last}", "median_sec": round(float(np.median(get(sel, "sec"))), 3),
                     "median_t_sim": round(float(np.nanmedian(get(sel, "t_sim"))), 3) if k == "E44" else None,
                     "mem_gb_step1": logs[k][0].get("mem_gb")})
    write("cost.csv", cost)


if __name__ == "__main__":
    main()
