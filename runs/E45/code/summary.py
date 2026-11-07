"""E45 表与图（#253）。在 gpfs worktree 里跑（ckpt、日志不同步到本机）：

    /opt/envs/loftr/bin/python runs/E45/code/summary.py

写 extra/data/*.csv、extra/*.png，并打印表格。对照为 E41 main__screen0.8（门槛 0.8，gpfs 主 checkout，不重跑）。横轴为更新次数（每次 4 对）。
1. Val（val.csv、vs_e41.csv、by_segment.csv）：两个运行各存档的 AUC@5、AUC@3、SR@3、SR@5、误差中位数与评测所用 GPU；
   同一存档位置 E45 减 E41 的 AUC@5 之差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 E41）；
   按训练集遍数（每 1977 次更新）分段的平均。
2. 训练曲线（curves.png）：两项奖励各自的均值（粗级 rew / eqv_rew，细化尺度 16 与 1 的 rew_s / eqv_rew_s），
   新项与现有项梯度之比（eqv_ratio，实际施加；eqv_coef，让两者 1:1 的系数），离参照的距离、画面内格子比例、训练图对上
   模型自身拟合的内点比例与 E41 叠画；均为 200 次更新滑动中位数。
3. 变换后一版与原图（eqv.png、eqv_by_segment.csv）：参照映射到画面内的格子 / 像素上，T2 变换后一版离 T2∘参照、原图离
   参照的距离（粗级抽签 eqv_d / eqv_d0；正常推理尺度 1、16 的输出 eqv_dmu_s / eqv_dmu0_s）及其比值，随训练的变化。
4. 开销（cost.csv）：每次更新耗时（日志 sec）中位数与总和、新项耗时 t_eqv 中位数、第 1 次更新的峰值显存 mem_gb、
   训练墙钟时间、训练所用 GPU。
5. 误差随真实几何差的变化（by_geom.csv、by_geom.png）：两边最后一个存档的 Val 预测（preds/<方法>/val.jsonl），每对用标注
   检查点最小二乘拟合出真实仿射，取它的旋转角绝对值、缩放偏离 |√det − 1|、平移幅度（光学检查点重心被映过去的位移）；
   各量按三分位分三档，每档给配准误差中位数，以及估出的仿射同一量与真值之比的中位数（1 = 估得一样大，< 1 = 估小）。
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
from finetune.config import tomllib  # noqa: E402  （3.11 以前的环境没有 tomllib，借 config 的兜底）
from workbench.dataset import Dataset  # noqa: E402
from workbench.protocol import auc, pair_error  # noqa: E402

Y = Path("/remote-home/xufang/YGC")
E41 = Y / "moon-exp" / "runs" / "E41"
FONT = str(Y / "fonts" / "simhei.ttf")
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
GPU = {"bcb626e99cea": "RTX 3090", "7b3282290ade": "V100"}     # 150、160 的容器 hostname
RUNS = {"E45": (R, "加等变约束奖励（E45）", "C3"), "E41": (E41, "不加（E41 门槛 0.8）", "0.35")}
METH = {"E45": "main", "E41": "main__screen0.8"}
PASS, UPD = 1977, 7907
OUT = R / "extra" / "data"
W = tomllib.loads((R / "configs" / "main.toml").read_text(encoding="utf-8"))["rloo"]["eqv_w"]
DATA = Path(__import__("os").environ.get("MOON_DATA", str(Y / "dataset" / "Moon")))


def geom(A, c):
    """仿射 A（2×3）的旋转角绝对值（度）、缩放偏离 |√det − 1|、点 c 处的位移幅度（px）。"""
    A = np.asarray(A, float)
    M = A[:, :2]
    rot = abs(np.degrees(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])))
    return rot, abs(np.sqrt(abs(np.linalg.det(M))) - 1), float(np.linalg.norm(M @ c + A[:, 2] - c))


def by_geom():
    ds = Dataset(DATA)
    pred = {k: {r["pair"]: r for r in jl(d / "preds" / METH[k] / "val.jsonl")} for k, (d, _, _) in RUNS.items()}
    rows = []
    for pair in sorted(set(pred["E45"]) & set(pred["E41"])):
        cp = ds.checkpoints("val", pair)
        if cp is None:
            continue
        po, ps = (np.asarray(x, float) for x in cp)
        A_t = np.linalg.lstsq(np.c_[po, np.ones(len(po))], ps, rcond=None)[0].T
        c = po.mean(0)
        g_t = geom(A_t, c)
        row = {"pair": pair, "rot": g_t[0], "scl": g_t[1], "sh": g_t[2]}
        for k in RUNS:
            A = pred[k][pair].get("A")
            e = pair_error(A, po, ps) if A is not None else None
            row[f"err_{k}"] = np.inf if e is None else e
            g = geom(A, c) if A is not None else (np.nan,) * 3
            for name, gt, ge in zip(("rot", "scl", "sh"), g_t, g):
                row[f"ratio_{name}_{k}"] = ge / gt if gt > 0 else np.nan
        rows.append(row)
    out = []
    for name, lab in (("rot", "旋转角（度）"), ("scl", "缩放偏离"), ("sh", "平移幅度（px）")):
        v = np.array([r[name] for r in rows])
        q = np.quantile(v, [0, 1 / 3, 2 / 3, 1])
        for b in range(3):
            sel = [r for r in rows if (q[b] <= r[name] <= q[b + 1]) and (b == 0 or r[name] > q[b])]
            o = {"quantity": name, "bin": b + 1, "lo": round(float(q[b]), 3), "hi": round(float(q[b + 1]), 3),
                 "n": len(sel)}
            for k in RUNS:
                o[f"median_err_{k}"] = round(float(np.median([r[f"err_{k}"] for r in sel])), 2)
            for k in RUNS:
                o[f"ratio_{k}"] = round(float(np.nanmedian([r[f"ratio_{name}_{k}"] for r in sel])), 3)
            out.append(o)
    write("by_geom.csv", out)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, (name, lab) in enumerate((("rot", "真实旋转角（度）"), ("scl", "真实缩放偏离"), ("sh", "真实平移幅度（px）"))):
        rs = [o for o in out if o["quantity"] == name]
        x = np.arange(len(rs))
        for i, (k, (_, nm, c)) in enumerate(RUNS.items()):
            ax[0, j].bar(x + (i - 0.5) * 0.38, [o[f"median_err_{k}"] for o in rs], 0.38, color=c, label=nm)
            ax[1, j].bar(x + (i - 0.5) * 0.38, [o[f"ratio_{k}"] for o in rs], 0.38, color=c, label=nm)
        for a in ax[:, j]:
            a.set_xticks(x, [f"{o['lo']:.3g}–{o['hi']:.3g}" for o in rs])
            a.set_xlabel(lab + "（三分位档）")
            a.legend(fontsize=8)
        ax[0, j].set_ylabel("配准误差中位数（px）")
        ax[1, j].set_ylabel("估出值 ÷ 真值（中位数）")
        ax[1, j].axhline(1.0, color="0.5", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "by_geom.png", dpi=110)


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
    val = {k: load_val(d, METH[k]) for k, (d, _, _) in RUNS.items()}
    logs = {k: jl(d / "ckpt" / METH[k] / "log.jsonl") for k, (d, _, _) in RUNS.items()}

    # 1. Val
    write("val.csv", [{"run": k, "update": s, **{x: round(val[k][s]["summary"][x], 3)
                                                  for x in ("auc@5", "auc@3", "sr@3", "sr@5")},
                       "median_err": round(val[k][s]["summary"]["median"], 2), "gpu": val[k][s]["gpu"]}
                      for k in RUNS for s in sorted(val[k])])
    diff = []
    for s in sorted(set(val["E45"]) & set(val["E41"])):
        d_, ci = paired(val["E45"][s]["errors"], val["E41"][s]["errors"])
        diff.append({"update": s, "auc5": round(val["E45"][s]["summary"]["auc@5"], 4),
                     "auc5_e41": round(val["E41"][s]["summary"]["auc@5"], 4),
                     "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                     "gpu": val["E45"][s]["gpu"], "gpu_e41": val["E41"][s]["gpu"]})
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
    lg = logs["E45"]
    st = np.array([r["step"] for r in lg])
    get = lambda log, key: np.array([r.get(key, np.nan) for r in log], float)
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    a = ax[0, 0]
    for key, lab, c in (("rew", "现有项（原图）", "C0"), ("eqv_rew", "新项（T2 变换后的 SAR，对 T2∘参照）", "C3")):
        a.plot(st, roll(get(lg, key), 200), color=c, lw=1.1, label=lab)
    a.set(ylabel="粗级奖励均值", xlabel="更新次数")
    a = ax[0, 1]
    for s_, ls in ((16, "-"), (1, "--")):
        a.plot(st, roll(get(lg, f"rew_{s_}"), 200), color="C0", ls=ls, lw=1.1, label=f"现有项 尺度 {s_}")
        a.plot(st, roll(get(lg, f"eqv_rew_{s_}"), 200), color="C3", ls=ls, lw=1.1, label=f"新项 尺度 {s_}")
    a.set(ylabel="细化抽签奖励均值", xlabel="更新次数")
    a = ax[0, 2]
    a.plot(st, roll(get(lg, "eqv_ratio"), 200), color="C3", lw=1.1, label=f"新项 ÷ 现有项（实际施加，系数 {W}）")
    a.plot(st, roll(get(lg, "eqv_coef"), 200), color="C2", lw=1.1, label="让两者 1:1 的系数")
    a.axhline(1.0, color="0.6", ls=":", lw=0.8)
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

    # 3. 变换后一版与原图离各自参照的距离
    rows = []
    edges = [0, 250, 500, 1000, 1977, 3954, 5931, UPD]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (st > lo) & (st <= hi)
        if not m.any():
            continue
        row = {"updates": f"{lo + 1}-{hi}", "n": int(m.sum())}
        for key in ("eqv_d", "eqv_d0", "eqv_dmu_1", "eqv_dmu0_1", "eqv_dmu_16", "eqv_dmu0_16", "eqv_rew", "rew"):
            row[key] = round(float(np.nanmedian(get(lg, key)[m])), 3)
        row["ratio_d"] = round(row["eqv_d"] / row["eqv_d0"], 3)
        row["ratio_dmu_1"] = round(row["eqv_dmu_1"] / row["eqv_dmu0_1"], 3)
        rows.append(row)
    write("eqv_by_segment.csv", rows)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, (k1, k0, lab) in zip(ax, (("eqv_d", "eqv_d0", "粗级抽到的位置"), ("eqv_dmu_1", "eqv_dmu0_1", "正常推理尺度 1 的输出"))):
        a.plot(st, roll(get(lg, k0), 200), color="C0", lw=1.1, label="原图")
        a.plot(st, roll(get(lg, k1), 200), color="C3", lw=1.1, label="T2 变换后（对 T2∘参照）")
        a.set_yscale("log")
        a.set(xlabel="更新次数", ylabel=f"{lab}离参照的距离（px，画面内）")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "eqv.png", dpi=110)

    # 4. 开销
    cost = []
    for k, (d, _, _) in RUNS.items():
        log = logs[k]
        sec = get(log, "sec")
        cost.append({"run": k, "updates": int(log[-1]["step"]), "train_gpu": train_gpu(d, METH[k]),
                     "median_sec": round(float(np.median(sec)), 3), "sec_sum_h": round(float(np.nansum(sec)) / 3600, 2),
                     "median_t_eqv": round(float(np.nanmedian(get(log, "t_eqv"))), 3) if k == "E45" else None,
                     "mem_gb_step1": log[0].get("mem_gb"), "wall_h": train_hours(d, METH[k])})
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
        a_.set(xlabel="更新次数", ylabel="Val AUC@5 之差（E45 减 E41 同位置存档）")
        fig.tight_layout()
        fig.savefig(R / "extra" / "diff.png", dpi=110)

    if all((d / "preds" / METH[k] / "val.jsonl").exists() for k, (d, _, _) in RUNS.items()):
        by_geom()


if __name__ == "__main__":
    main()
