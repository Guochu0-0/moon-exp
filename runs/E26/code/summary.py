"""E26 汇总（#206）：probe.py 的 raw/ 产物、各存档的正式评测（preds/）、各运行的训练日志 → extra/ 的表与图。
用法（在仓库根目录、wb 环境）：MOON_DATA=… PYTHONPATH=. python runs/E26/code/summary.py

- 表 morph.csv（方法 1）：每个方法在验证集 652 对上各量的中位数。正式评测的仿射（preds，560 → 864、symmetric、抽 10000
  点、RANSAC 3 px）给出奇异值、行列式、图块中心 (256, 256) 映射到的位置及其在各对之间的离散程度（两轴标准差的均方根，px），
  与标注仿射同样算出的离散程度对照；其余列来自 probe.py morph（训练时的正常推理）。
- same.csv：七个运行最后一个存档两两之间，同一对图上正式评测的仿射把图块中心映射到的位置相距多远（各对的中位数，px）。
- learn.csv（方法 2）：probe.py learn 在验证集 100 对上各量的中位数。
- events.json、log.png（方法 3）：训练日志的 50 步滑动中位数（图只画前 2000 步，健康对照为灰线）；各运行最高分概率首次超过 0.99、同组奖励标准差首次低于
  1e-3、离自估仿射的距离首次超过 100 px 的步，以及验证集 AUC@5 第一次低于 0.1 的存档。
- trend.png：每 500 步存档的三个运行，方法 1 各量按存档连成走势。
- centers.png：各运行最后一个存档，正式评测的仿射把图块中心映射到的位置（每对一个点）。
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workbench.dataset import Dataset
from workbench.protocol import auc, pair_error

HERE = Path(__file__).resolve().parent
E = HERE.parent
sys.path.insert(0, str(HERE))
from methods import COLLAPSED, EVERY500, HEALTHY, METHODS   # noqa: E402

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
MAIN = Path("/remote-home/xufang/YGC/moon-exp/runs")
OUT = E / "extra"
DATA = OUT / "data"
CENTER = np.array([256.0, 256.0, 1.0])
WIN = 50
COLORS = plt.get_cmap("tab10").colors


def runkey(e, r):
    return f"{e} {r}"


def preds(m):
    out = {}
    for line in (E / "preds" / m / "val.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["pair"]] = None if d.get("A") is None else np.asarray(d["A"], np.float64)
    return out


def spread(xy):
    xy = xy[np.isfinite(xy).all(1)]
    return float(np.sqrt(xy.var(0).sum() / 2)) if len(xy) else math.nan


def med(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else math.nan


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def morph_table(ds):
    rows, eval_ctr = [], {}
    pairs = json.loads((E / "raw" / "morph_zs.json").read_text(encoding="utf-8"))["pairs"]
    cps = {p: ds.checkpoints("val", p) for p in pairs}
    G = np.array([np.linalg.lstsq(np.c_[cps[p][0], np.ones(len(cps[p][0]))], cps[p][1], rcond=None)[0].T @ CENTER
                  for p in pairs])
    for m, (e, r, s) in METHODS.items():
        P = preds(m)
        err = np.array([pair_error(P.get(p), *cps[p]) for p in pairs])
        A = [P.get(p) for p in pairs]
        ctr = np.array([a @ CENTER if a is not None else [np.nan, np.nan] for a in A])
        eval_ctr[m] = ctr
        sv = np.array([np.linalg.svd(a[:, :2], compute_uv=False) if a is not None else [np.nan] * 2 for a in A])
        det = np.array([np.linalg.det(a[:, :2]) if a is not None else np.nan for a in A])
        z = np.load(E / "raw" / f"morph_{m}.npz")
        row = {"method": m, "run": runkey(e, r), "step": s, "auc5": auc(err, 5.0), "err_med": med(err),
               "fail": float(np.mean([a is None for a in A])), "sv1": med(sv[:, 0]), "sv2": med(sv[:, 1]),
               "det": med(det), "det_abs_lt_0.1": float(np.nanmean(np.abs(det) < 0.1)),
               "ctr_x": med(ctr[:, 0]), "ctr_y": med(ctr[:, 1]), "ctr_spread": spread(ctr), "gt_ctr_spread": spread(G),
               "ctr_err_med": med(np.linalg.norm(ctr - G, axis=1))}
        for k in ("H_med", "pmode_med", "n_modes", "top_mode_frac", "c_spread", "f_spread", "f_cover", "cert_mean",
                  "cert_hi", "d_self", "err_self", "sv1", "sv2", "det", "n_inl", "ok"):
            row[f"m_{k}"] = med(z[k]) if k != "ok" else float(np.mean(z[k]))
        row["m_c_ctr_spread"] = spread(np.c_[z["c_cx"], z["c_cy"]])
        row["m_self_ctr_spread"] = spread(np.c_[z["ctr_x"], z["ctr_y"]])
        rows.append(row)
    write_csv(DATA / "morph.csv", rows)
    return {r["method"]: r for r in rows}, eval_ctr, G, pairs


def same_table(eval_ctr):
    last = [f"{e.lower()}_{r}_4000" for e, r in COLLAPSED]
    rows = []
    for a in last:
        row = {"method": a}
        for b in last:
            row[b] = med(np.linalg.norm(eval_ctr[a] - eval_ctr[b], axis=1))
        rows.append(row)
    write_csv(DATA / "same.csv", rows)
    return rows


def learn_table():
    rows = []
    for m, (e, r, s) in METHODS.items():
        z = np.load(E / "raw" / f"learn_{m}.npz")
        row = {"method": m, "run": runkey(e, r), "step": s}
        for k in z.files:
            row[k] = med(z[k])
        rows.append(row)
    write_csv(DATA / "learn.csv", rows)
    return {r["method"]: r for r in rows}


def rolling_median(x, w=WIN):
    x = np.asarray(x, np.float64)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w + 1): i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


LOG_KEYS = ("pmode", "H", "rew", "rew_sd", "d_ref", "nonmode", "gnorm", "upd_coarse", "upd_lora", "upd_fine",
            "d_cur", "ref_gap", "n_inl")


def load_log(e, r):
    rows = [json.loads(x) for x in (MAIN / e / "ckpt" / r / "log.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    rows = list({d["step"]: d for d in rows}.values())       # 续训重叠的步取最后一次
    rows.sort(key=lambda d: d["step"])
    step = np.array([d["step"] for d in rows])
    return step, {k: np.array([d.get(k, np.nan) for d in rows], np.float64) for k in LOG_KEYS}


def sweep_auc(e, r):
    m = json.loads((MAIN / e / "sweep" / r / "S" / "metrics.json").read_text(encoding="utf-8"))["methods"]
    return {int(k[4:]): v["val"]["summary"]["auc@5"] for k, v in m.items()}


def first(step, x, cond):
    idx = np.nonzero(cond(x))[0]
    return int(step[idx[0]]) if len(idx) else None


def logs():
    runs = list(COLLAPSED) + list(HEALTHY)
    ev, curves = {}, {}
    for e, r in runs:
        step, L = load_log(e, r)
        R = {k: rolling_median(v) for k, v in L.items()}
        curves[(e, r)] = (step, R)
        sw = sweep_auc(e, r)
        bad = [s for s in sorted(sw) if sw[s] < 0.1]
        ev[runkey(e, r)] = {
            "pmode_gt_0.99": first(step, R["pmode"], lambda x: x > 0.99),
            "rew_sd_lt_1e-3": first(step, R["rew_sd"], lambda x: x < 1e-3),
            "d_ref_gt_100": first(step, R["d_ref"], lambda x: x > 100),
            "first_ckpt_auc_lt_0.1": bad[0] if bad else None,
            "last_ckpt_auc_ge_0.1": max([s for s in sw if sw[s] >= 0.1], default=None),
            "frac_steps_rew_sd_0": float(np.mean(L["rew_sd"] == 0)),
            "first_step_rew_sd_0": first(step, L["rew_sd"], lambda x: x == 0),
            "max_d_ref_roll_after_100": float(np.nanmax(R["d_ref"][step >= 100])),
            "windows": {str(a): {k: med(v[(step >= a) & (step < a + 100)]) for k, v in L.items()}
                        for a in (1, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 3900)},
        }
    (DATA / "events.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    panels = [("pmode", "最高分位置的概率（训练温度）", False), ("H", "粗级熵（训练温度，nat）", False),
              ("d_ref", "离自估仿射的距离（px）", True), ("rew_sd", "同组奖励的标准差", True),
              ("gnorm", "梯度范数（裁剪前）", True), ("upd_coarse", "粗级 decoder 每步相对改变量", True),
              ("upd_lora", "LoRA 每步相对改变量", True), ("upd_fine", "细化部分每步相对改变量", True)]
    fig, axs = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, (k, title, logy) in zip(axs.flat, panels):
        for i, (e, r) in enumerate(list(HEALTHY) + list(COLLAPSED)):
            step, R = curves[(e, r)]
            healthy = (e, r) in HEALTHY
            if np.isfinite(R[k]).any():
                ax.plot(step, R[k], lw=0.8 if healthy else 1, color="0.65" if healthy else COLORS[i - len(HEALTHY)],
                        label=("健康对照" if i == 0 else None) if healthy else runkey(e, r))
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 2000)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(PLAIN)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("步")
    axs.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "log.png", dpi=110)
    plt.close(fig)
    return ev


def trend(T):
    panels = [("auc5", "验证集 AUC@5", False), ("err_med", "配准误差中位数（px）", True),
              ("det", "正式评测仿射的行列式", False), ("ctr_spread", "图块中心落点在各对间的离散程度（px）", True),
              ("m_H_med", "粗级熵中位数（温度 1，nat）", False), ("m_top_mode_frac", "最常见的最高分位置占的格子比例", False),
              ("m_f_cover", "尺度 1 落点覆盖 SAR 画面的比例", False), ("m_cert_mean", "可信度均值", False),
              ("m_d_self", "粗匹配离自估仿射的距离（px）", True)]
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (k, title, logy) in zip(axs.flat, panels):
        for i, (e, r) in enumerate(EVERY500):
            ms = [m for m, (e2, r2, _) in METHODS.items() if (e2, r2) == (e, r)]
            ax.plot([T[m]["step"] for m in ms], [T[m][k] for m in ms], "o-", color=COLORS[i], label=runkey(e, r))
        for j, (e, r) in enumerate(HEALTHY):
            ax.axhline(T[f"{e.lower()}_{r}_4000"][k], color="gray", ls=":", lw=1)
        ax.axhline(T["zs"][k], color="k", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(PLAIN)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("存档（步）")
    axs.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trend.png", dpi=110)
    plt.close(fig)


def centers(eval_ctr, G):
    show = [f"{e.lower()}_{r}_4000" for e, r in COLLAPSED + HEALTHY] + ["zs"]
    fig, axs = plt.subplots(2, 6, figsize=(20, 7.2))
    for ax, m in zip(axs.flat, show + ["__gt"]):
        xy = G if m == "__gt" else eval_ctr[m]
        ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.5)
        ax.add_patch(plt.Rectangle((0, 0), 512, 512, fill=False, ec="k", lw=0.8))
        ax.set_xlim(-300, 812)
        ax.set_ylim(812, -300)
        ax.set_aspect("equal")
        e, r, s = METHODS.get(m, ("", "", 0))
        ax.set_title("标注仿射" if m == "__gt" else "未训练" if m == "zs" else f"{e} {r} 第 {s} 步", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "centers.png", dpi=110)
    plt.close(fig)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    ds = Dataset(os.environ["MOON_DATA"])
    T, eval_ctr, G, pairs = morph_table(ds)
    same_table(eval_ctr)
    learn_table()
    logs()
    trend(T)
    centers(eval_ctr, G)
    print("ok", len(T), "methods,", len(pairs), "pairs")


if __name__ == "__main__":
    main()
