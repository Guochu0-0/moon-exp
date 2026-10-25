"""E27 的图。先跑 summary.py（写 extra/data/summary.json 与各方法的 *_obs.csv、*_train.npz）。

    /opt/envs/wb/bin/python runs/E27/code/figs.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from load import R, anchor_xy, cell_centers, log, obs, obs_steps, rolling_median, series, train_trace  # noqa: E402

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = R / "extra"
D = OUT / "data"
SUM = json.loads((D / "summary.json").read_text(encoding="utf-8"))
NAMES = {"clip": "截平（E19 clip）", "loo_t02": "温度 0.2（E15 loo_t02）", "f1": "训细化（E25 f1）", "gauss8": "高斯（E19 gauss8）"}


def label(m):
    c, s = m.rsplit("_s", 1)
    return f"{NAMES.get(c, c)} 种子 {s}"


def collapsed():
    return [m for m, v in SUM.items() if v["collapsed"] and "_r" not in m.split("_s")[-1]]


def healthy():
    return [m for m, v in SUM.items() if not v["collapsed"] and "_r" not in m.split("_s")[-1]]


def obs_csv(m):
    with open(D / f"{m}_obs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def break_step(m):
    """观察集上原图误差中位数第一次超过 20 px 的步（配准失效的起点）。"""
    o = obs_csv(m)
    i = np.flatnonzero(o["err_orig"] > 20)
    return int(o["step"][i[0]]) if len(i) else None


def fig_timeline(ms):
    """每个坍缩运行一行：观察集上相对标注的误差与自洽距离、错格与不自洽格子的比例、落在 q0 的比例。"""
    fig, axs = plt.subplots(len(ms), 3, figsize=(15, 2.6 * len(ms)), squeeze=False, sharex=True)
    for row, m in zip(axs, ms):
        o = obs_csv(m)
        st, d = series(log(m), "d_ref")
        b = break_step(m)
        ax = row[0]
        ax.plot(o["step"], o["err_orig"], label="观察集原图：自估仿射离标注（中位）")
        ax.plot(o["step"], o["err_pert"], label="观察集扰动图：自估仿射离标注（中位）")
        ax.plot(o["step"], o["d_self"], label="观察集：粗匹配离自估仿射（中位）")
        ax.plot(st, rolling_median(d), color="gray", lw=0.8, label="训练批次：离自估仿射（50 步滑动中位）")
        ax.set_yscale("log")
        ax.set_ylabel(label(m), fontsize=9)
        ax = row[1]
        ax.plot(o["step"], o["wrong_gt"], label="粗匹配离标注 > 16 px 的格子")
        ax.plot(o["step"], o["off_self"], label="粗匹配离自估仿射 > 16 px 的格子")
        ax.plot(o["step"], o["cert1"], label="可信度均值")
        ax.set_ylim(0, 1)
        ax = row[2]
        ax.plot(o["step"], o["at_q0"], label="观察集：最高分位置落在 q0 的格子")
        t = np.load(D / f"{m}_train.npz")
        ax.plot(t["step"], rolling_median(t["at_q0"], 10), color="gray", lw=0.8, label="训练批次（10 步滑动中位）")
        ax.plot(o["step"], o["pmode1"], label="观察集：最高分位置的概率（温度 1）")
        ax.set_ylim(0, 1)
        for ax in row:
            if b is not None:
                ax.axvline(b, color="k", ls=":", lw=0.8)
    for ax, t in zip(axs[0], ("误差与距离（px）", "格子比例", "聚到 q0")):
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axs[-1]:
        ax.set_xlabel("训练步")
    fig.tight_layout()
    fig.savefig(OUT / "timeline.png", dpi=110)
    plt.close(fig)


def mapped(A, x, y):
    return A[0, 0] * x + A[0, 1] * y + A[0, 2], A[1, 0] * x + A[1, 1] * y + A[1, 2]


def offsets(m):
    """观察集各步：16 对原图上粗匹配相对标注的偏移（各对取中位向量后再取中位）与对内离散（中位）。"""
    cx, cy = cell_centers()
    rows = []
    for st in obs_steps(m):
        o = obs(m, st)
        v = []
        for j in np.flatnonzero(o["version"] == 0):
            gx, gy = mapped(o["A_gt"][j], cx, cy)
            dx, dy = o["pos"][j, 0].astype(float) - gx, o["pos"][j, 1].astype(float) - gy
            mx, my = np.median(dx), np.median(dy)
            v.append((mx, my, np.median(np.hypot(dx - mx, dy - my))))
        v = np.array(v)
        rows.append((st, *np.median(v, 0)))
    return np.array(rows)


def fig_drift(ms):
    """粗匹配相对标注怎样偏：各对一致的偏移（中位向量）与对内的离散；以及一对图上全部格子的落点随步数的变化。"""
    fig, axs = plt.subplots(len(ms), 2, figsize=(11, 3.2 * len(ms)), squeeze=False,
                            gridspec_kw={"width_ratios": [2, 1]})
    for (a, b), m in zip(axs, ms):
        off = offsets(m)
        a.plot(off[:, 0], off[:, 1], label="偏移 x（中位）")
        a.plot(off[:, 0], off[:, 2], label="偏移 y（中位）")
        a.plot(off[:, 0], off[:, 3], label="对内离散（中位）")
        a.axhline(0, color="k", lw=0.5)
        a.set_ylabel(label(m), fontsize=9)
        a.set_xlabel("训练步")
        bs = break_step(m)
        steps = [s for s in (0, bs - 20 if bs else None, bs, bs + 20 if bs else None, 1200) if s is not None]
        o0 = obs(m, 0)
        j = int(np.flatnonzero(o0["version"] == 0)[0])
        for s, c in zip(steps, plt.get_cmap("viridis")(np.linspace(0, 1, len(steps)))):
            o = obs(m, s)
            b.scatter(o["pos"][j, 0].reshape(-1), o["pos"][j, 1].reshape(-1), s=1, color=c, label=f"第 {s} 步")
        x0, y0 = SUM[m]["q0_xy"]
        b.plot([x0], [y0], "r*", ms=10, label="q0")
        b.add_patch(plt.Rectangle((0, 0), 512, 512, fill=False, lw=0.8))
        b.set_xlim(-60, 572)
        b.set_ylim(572, -60)
        b.set_aspect("equal")
        b.set_title(str(o0["pair"][j]), fontsize=8)
    axs[0, 0].legend(fontsize=7)
    axs[0, 1].legend(fontsize=6, markerscale=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "drift.png", dpi=110)
    plt.close(fig)


def main():
    ms = collapsed()
    fig_timeline(ms)
    fig_drift(ms)


if __name__ == "__main__":
    main()
