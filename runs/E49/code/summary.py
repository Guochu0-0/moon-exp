"""E49 表与图（#261）。在 gpfs worktree 里跑（ckpt、日志不同步到本机）：

    /opt/envs/loftr/bin/python runs/E49/code/summary.py

写 extra/data/*.csv、extra/*.png，并打印表格。对照为 E41 门槛 0.8（main__screen0.8，gpfs 主 checkout，不重跑）。
横轴为更新次数（每次 4 对）。两次训练：强档 main（系数 130）、弱档 main__weak（系数 39）；还没训完的按已有的算。
1. Val（val.csv、vs_e41.csv、by_segment.csv）：各存档的 AUC@5、AUC@3、SR@3、SR@5、误差中位数；同一存档位置减 E41 门槛 0.8
   的 AUC@5 之差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 E41）；按训练集遍数（每 1977 次更新）
   分四段的平均差，以及区间两端都为正 / 都为负的存档数。图 auc.png：逐存档 AUC@5 与 E41 叠画。
2. 训练曲线（curves.png）：现有项粗级奖励均值、离参照的距离、画面内格子比例与 E41 叠画；实际施加的梯度比
   （sim_ratio_avg：系数 × 新项梯度 4 卡平均后的范数 ÷ 现有项梯度 4 卡平均后的范数，共用参数上）；
   NCC 最高与离参照最近不是同一份的比例（sim_diff）；20 次更新滑动中位数（比例为滑动平均），横轴只画到第 900 次更新。
3. 分段诊断（diag_by_segment.csv）：每 100 次更新一段，实际施加的梯度比的中位数与四分位，sim_diff 平均，
   现有项粗级奖励均值与 E41 同段之差，组内 NCC 均值、记 −1 的比例、新项优势的组内标准差、
   Val 图对上组内 NCC 与真实误差的 Spearman 秩相关（sim_rho，有记录的步平均）、离参照的距离中位数。
4. 开销（cost.csv）：第 101–822 次更新（避开开训时样本筛选大量补抽的阶段；822 为弱档停下处，各训练取到自己的最后一次为止）每次更新耗时（日志 sec）中位数、新项耗时 t_sim 中位数、峰值显存、
   开训到最后一个存档的小时数（ckpt 文件修改时间减 log 第一行之前的 args.json 修改时间）。
5. 坍缩（halt.csv）：ckpt/<方法>/halt.json（停在哪一步、哪个量越线）；stop.json（停止规则）。
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
RUNS = {"strong": (R, "main", "强档（系数 130）", "C3"),
        "weak": (R, "main__weak", "弱档（系数 39）", "C0"),
        "E41": (E41, "main__screen0.8", "不加（E41 门槛 0.8）", "0.35")}
OUT = R / "extra" / "data"
WIN = 20
XMAX = 900
SEG = 1977


def jl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def roll(x, win, f=np.nanmedian):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - win + 1):i + 1]
        if np.isfinite(w).any():
            out[i] = f(w)
    return out


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


def seg_of(s):
    return min((s - 1) // SEG + 1, 4)


def main():
    (R / "extra").mkdir(exist_ok=True)
    val = {k: load_val(d, m) for k, (d, m, _, _) in RUNS.items()}
    logs = {k: jl(d / "ckpt" / m / "log.jsonl") for k, (d, m, _, _) in RUNS.items()}
    mine = [k for k in ("strong", "weak") if logs[k]]
    get = lambda log, key: np.array([r.get(key, np.nan) for r in log], float)

    # 1. Val
    write("val.csv", [{"run": k, "update": s, **{x: round(val[k][s]["summary"][x], 4)
                                                  for x in ("auc@5", "auc@3", "sr@3", "sr@5")},
                       "median_err": round(val[k][s]["summary"]["median"], 2), "gpu": val[k][s]["gpu"]}
                      for k in RUNS for s in sorted(val[k])])
    diff = []
    for k in mine:
        for s in sorted(set(val[k]) & set(val["E41"])):
            d_, ci = paired(val[k][s]["errors"], val["E41"][s]["errors"])
            diff.append({"run": k, "update": s, "auc5": round(val[k][s]["summary"]["auc@5"], 4),
                         "auc5_e41": round(val["E41"][s]["summary"]["auc@5"], 4),
                         "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4)})
    write("vs_e41.csv", diff)
    bs = []
    for k in mine:
        for g in range(1, 5):
            rs = [r for r in diff if r["run"] == k and seg_of(r["update"]) == g]
            if rs:
                bs.append({"run": k, "segment": g, "n_ckpt": len(rs),
                           "mean_auc5": round(float(np.mean([r["auc5"] for r in rs])), 4),
                           "mean_auc5_e41": round(float(np.mean([r["auc5_e41"] for r in rs])), 4),
                           "mean_diff": round(float(np.mean([r["diff"] for r in rs])), 4),
                           "n_higher": sum(r["ci_lo"] > 0 for r in rs), "n_lower": sum(r["ci_hi"] < 0 for r in rs)})
    write("by_segment.csv", bs)
    fig, a = plt.subplots(figsize=(9, 4.5))
    for k, (_, _, name, c) in RUNS.items():
        ss = sorted(val[k])
        if ss:
            a.plot(ss, [val[k][s]["summary"]["auc@5"] for s in ss], "o-", ms=3, lw=1.0, color=c, label=name)
    for g in range(1, 4):
        a.axvline(g * SEG, color="0.85", lw=0.8)
    a.set(xlabel="更新次数", ylabel="Val AUC@5", xlim=(0, 2100))
    a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "extra" / "auc.png", dpi=110)

    # 2. 训练曲线
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for a, key, lab in ((ax[0, 0], "rew", "现有项粗级奖励均值"), (ax[0, 1], "d_ref", "离参照的距离（px）"),
                        (ax[1, 0], "in_frame", "画面内格子比例")):
        for k, (_, _, name, c) in RUNS.items():
            if logs[k]:
                lk = [r for r in logs[k] if r["step"] <= XMAX]
                a.plot([r["step"] for r in lk], roll(get(lk, key), WIN), color=c, lw=1.0, label=name)
        a.set(ylabel=lab, xlabel="更新次数")
    ax[0, 1].set_yscale("log")
    for a, key, lab, f in ((ax[0, 2], "sim_ratio_avg", "实际施加的梯度比（新项 ÷ 现有项）", np.nanmedian),
                           (ax[1, 1], "sim_diff", "NCC 最高与离参照最近不是同一份的比例", np.nanmean)):
        for k in mine:
            a.plot([r["step"] for r in logs[k]], roll(get(logs[k], key), WIN, f), color=RUNS[k][3], lw=1.0,
                   label=RUNS[k][2])
        a.set(ylabel=lab, xlabel="更新次数")
    for y in (1.0, 0.3):
        ax[0, 2].axhline(y, color="0.6", ls=":", lw=0.8)
    ax[0, 2].set_yscale("log")
    a = ax[1, 2]
    for k in mine:
        a.plot([r["step"] for r in logs[k]], roll(get(logs[k], "sim_ncc"), WIN), color=RUNS[k][3], lw=1.0,
               label=RUNS[k][2])
    a.set(ylabel="组内 NCC 均值（训练图对）", xlabel="更新次数")
    for a in ax.flat:
        a.legend(fontsize=8)
        a.set_xlim(0, XMAX)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)

    # 3. 分段诊断
    rows = []
    e41 = logs["E41"]
    for k in mine:
        lg = logs[k]
        last = int(lg[-1]["step"])
        for lo in range(0, last, 100):
            hi = min(lo + 100, last)
            sel = [r for r in lg if lo < r["step"] <= hi]
            ref = [r for r in e41 if lo < r["step"] <= hi]
            ra = get(sel, "sim_ratio_avg")
            ra = ra[np.isfinite(ra)]
            m = lambda key, s=sel: (round(float(np.nanmean(get(s, key))), 4)
                                    if np.isfinite(get(s, key)).any() else None)
            rows.append({"run": k, "updates": f"{lo + 1}-{hi}",
                         "ratio_med": round(float(np.median(ra)), 3) if len(ra) else None,
                         "ratio_q25": round(float(np.quantile(ra, 0.25)), 3) if len(ra) else None,
                         "ratio_q75": round(float(np.quantile(ra, 0.75)), 3) if len(ra) else None,
                         "sim_diff": m("sim_diff"), "rew": m("rew"), "rew_e41": m("rew", ref),
                         "rew_minus_e41": round(m("rew") - m("rew", ref), 4) if ref else None,
                         "sim_ncc": m("sim_ncc"), "sim_fail": m("sim_fail"), "sim_adv_sd": m("sim_adv_sd"),
                         "sim_rho": m("sim_rho"), "n_rho": int(np.isfinite(get(sel, "sim_rho")).sum()),
                         "d_ref_med": round(float(np.nanmedian(get(sel, "d_ref"))), 2),
                         "d_ref_med_e41": round(float(np.nanmedian(get(ref, "d_ref"))), 2) if ref else None})
    write("diag_by_segment.csv", rows)

    # 4. 开销、5. 坍缩与停训
    cost, halt = [], []
    for k in RUNS:
        d, mth = RUNS[k][0], RUNS[k][1]
        lg = logs[k]
        if not lg:
            continue
        sel = [r for r in lg if 100 < r["step"] <= 822]
        cks = sorted((d / "ckpt" / mth).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        t0 = (d / "ckpt" / mth / "args.json").stat().st_mtime if (d / "ckpt" / mth / "args.json").exists() else None
        cost.append({"run": k, "last_update": int(lg[-1]["step"]),
                     "median_sec_101_822": round(float(np.median(get(sel, "sec"))), 3) if sel else None,
                     "median_t_sim_101_822": round(float(np.nanmedian(get(sel, "t_sim"))), 3)
                     if sel and np.isfinite(get(sel, "t_sim")).any() else None,
                     "max_mem_gb": round(float(np.nanmax(get(lg, "mem_gb"))), 2),
                     "hours_to_last_ckpt": round((cks[-1].stat().st_mtime - t0) / 3600, 2) if cks and t0 else None})
        for f in ("halt.json", "stop.json"):
            p = d / "ckpt" / mth / f
            if p.exists() and k != "E41":
                halt.append({"run": k, "file": f, **json.loads(p.read_text(encoding="utf-8"))})
    write("cost.csv", cost)
    write("halt.csv", halt)


if __name__ == "__main__":
    main()
