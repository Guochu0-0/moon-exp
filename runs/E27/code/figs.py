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
    return [m for m, v in SUM.items() if v["failed"] and "_r" not in m.split("_s")[-1]]


def healthy():
    return [m for m, v in SUM.items() if not v["failed"] and "_r" not in m.split("_s")[-1]]


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


def windows(m):
    """三段：配准失效之前 100 步、失效到概率超过 0.99（或第 1200 步）的前半与后半。"""
    b = break_step(m) or 0
    e = SUM[m]["events"]["p99"] or 1200
    mid = (b + e) // 2
    return [(max(1, b - 100), b), (b, mid), (mid, e)]


def signal_maps(m):
    """每段里有信号的格子的最高分位置在 SAR 画面上的分布（按梯度大小加权，64×64 锚点格）、有信号格子比例、
    信号集中度（最强的一成格子占全部信号的比例）、更新后移动的格子比例与移动在指向 q0 方向上的分量。"""
    tr = train_trace(m, ["step", "mode", "gnorm", "pos", "pos_post", "pad"])
    x0, y0 = SUM[m]["q0_xy"]
    st = tr["step"]
    g = np.nan_to_num(tr["gnorm"].astype(float).reshape(len(st), -1))
    pos, post = tr["pos"].astype(float).reshape(len(st), 2, -1), tr["pos_post"].astype(float).reshape(len(st), 2, -1)
    mv = post - pos
    to = np.stack([x0 - pos[:, 0], y0 - pos[:, 1]], 1)
    to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-6
    toward = (mv * to).sum(1)
    moved = np.hypot(mv[:, 0], mv[:, 1]) > 1
    per = {"step": st, "signal": (g > 0).mean(1),
           "top": np.sort(g, 1)[:, -160:].sum(1) / (g.sum(1) + 1e-12),
           "moved": moved.mean(1),
           "toward": np.array([np.median(t[mk]) if mk.any() else np.nan for t, mk in zip(toward, moved)])}
    maps = []
    for a, b in windows(m):
        s = (st > a) & (st <= b)
        h = np.zeros(64 * 64)
        np.add.at(h, tr["mode"][s].reshape(-1).astype(int), g[s].reshape(-1))
        maps.append(((a, b), h.reshape(64, 64) / max(h.sum(), 1e-12)))
    pad = tr["pad"].astype(float).mean(0) / 255
    return per, maps, pad


def fig_converge(ms):
    fig, axs = plt.subplots(len(ms), 6, figsize=(20, 3.2 * len(ms)), squeeze=False,
                            gridspec_kw={"width_ratios": [2, 2, 1, 1, 1, 1]})
    for row, m in zip(axs, ms):
        per, maps, pad = signal_maps(m)
        x0, y0 = SUM[m]["q0_xy"]
        sm = lambda v: rolling_median(v, 20)
        row[0].plot(per["step"], sm(per["signal"]), label="有信号的格子比例")
        row[0].plot(per["step"], sm(per["top"]), label="最强一成格子占的信号")
        row[0].set_ylim(0, 1.02)
        row[0].set_ylabel(label(m), fontsize=9)
        row[1].plot(per["step"], sm(per["moved"]), label="更新后粗匹配移动 > 1 px 的格子比例")
        ax2 = row[1].twinx()
        ax2.plot(per["step"], sm(per["toward"]), color="C1", lw=0.8, label="移动的格子朝 q0 的分量（px，中位）")
        ax2.axhline(0, color="C1", lw=0.4)
        row[1].set_ylim(0, 1.02)
        for ax in row[:2]:
            for a, b in windows(m):
                ax.axvspan(a, b, color="gray", alpha=0.08)
        for ax, ((a, b), h) in zip(row[2:5], maps):
            ax.imshow(h, extent=(0, 512, 512, 0), cmap="magma")
            ax.plot([x0], [y0], "c*", ms=8)
            ax.set_title(f"第 {a}–{b} 步", fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
        im = row[5].imshow(pad, extent=(0, 512, 512, 0), cmap="gray_r", vmin=0, vmax=1)
        row[5].plot([x0], [y0], "c*", ms=8)
        row[5].set_title(f"补零比例（q0 处 {pad[int(y0 // 8), int(x0 // 8)]:.2f}）", fontsize=8)
        row[5].set_xticks([]), row[5].set_yticks([])
    axs[0, 0].legend(fontsize=7)
    axs[0, 1].legend(fontsize=7, loc="upper left")
    axs[0, 2].set_title("有信号格子的最高分位置（按梯度加权）\n" + axs[0, 2].get_title(), fontsize=8)
    for ax in axs[-1, :2]:
        ax.set_xlabel("训练步")
    fig.tight_layout()
    fig.savefig(OUT / "converge.png", dpi=110)
    plt.close(fig)


def step_jumps(m, upto):
    """每步更新后粗匹配移动 > 1 px 的格子比例与梯度范数（训练批次），只取第 upto 步之前。"""
    tr = train_trace(m, ["step", "pos", "pos_post"])
    mv = np.hypot(*(tr["pos_post"].astype(float) - tr["pos"].astype(float)).transpose(1, 0, 2, 3))
    st = tr["step"]
    gn = dict(zip(*series(log(m), "gnorm")))
    keep = st < upto
    return st[keep], (mv > 1).reshape(len(st), -1).mean(1)[keep], np.array([gn.get(s, np.nan) for s in st[keep]])


def fig_jumps(ms, ref):
    """坍缩前每步的更新幅度：坍缩运行与健康运行同一段步数里的分布。"""
    fig, axs = plt.subplots(len(ms), 1, figsize=(12, 2.4 * len(ms)), squeeze=False)
    rows = {}
    for ax, m in zip(axs[:, 0], ms):
        b = break_step(m)
        st, mv, gn = step_jumps(m, b + 1)
        hs = [h for h in ref if h.rsplit("_s", 1)[0] == m.rsplit("_s", 1)[0]]
        base = np.concatenate([step_jumps(h, b + 1)[1] for h in hs]) if hs else np.array([np.nan])
        thr = np.nanpercentile(base, 99) if hs else np.nan
        ax.plot(st, mv, lw=0.6, label="坍缩运行：更新后移动的格子比例")
        if hs:
            ax.axhline(thr, color="gray", ls="--", lw=0.8, label=f"同配置健康种子同段步数的 99% 分位（{thr:.2f}）")
        top = np.argsort(mv)[-5:][::-1]
        ax.plot(st[top], mv[top], "rv", ms=5)
        ax.axvline(b, color="k", ls=":", lw=0.8)
        ax.set_ylabel(label(m), fontsize=9)
        ax.set_ylim(0, 1)
        rows[m] = [{"step": int(st[i]), "moved": float(mv[i]), "gnorm": float(gn[i])} for i in top]
    axs[0, 0].legend(fontsize=7)
    axs[-1, 0].set_xlabel("训练步")
    fig.tight_layout()
    fig.savefig(OUT / "jumps.png", dpi=110)
    plt.close(fig)
    (D / "jumps.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


LEVELS = (16, 8, 4, 2, 1)


def level_series(m):
    """观察集原图各步：粗匹配与细化各级正常推理输出离自估仿射、离标注的距离（各对中位后再取中位）。"""
    out = {"step": []}
    for st in obs_steps(m):
        o = obs(m, st)
        sel = o["version"] == 0
        out["step"].append(st)
        for k in ["d_self", "d_gt"] + [f"dmu_{s}" for s in LEVELS] + [f"dmu_gt_{s}" for s in LEVELS]:
            v = np.nanmedian(o[k][sel].astype(float).reshape(sel.sum(), -1), 1)
            out.setdefault(k, []).append(float(np.median(v)))
    return {k: np.array(v) for k, v in out.items()}


def fig_fine(ms):
    """失控时各级离自估仿射多远：粗匹配与细化尺度 16…1 的输出（观察集原图）。"""
    fig, axs = plt.subplots(len(ms), 2, figsize=(13, 2.8 * len(ms)), squeeze=False, sharex=True)
    for (a, b), m in zip(axs, ms):
        L = level_series(m)
        a.plot(L["step"], L["d_self"], "k", label="粗匹配")
        b.plot(L["step"], L["d_gt"], "k", label="粗匹配")
        for s, c in zip(LEVELS, plt.get_cmap("plasma")(np.linspace(0, 0.85, len(LEVELS)))):
            a.plot(L["step"], L[f"dmu_{s}"], color=c, lw=0.9, label=f"细化尺度 {s}")
            b.plot(L["step"], L[f"dmu_gt_{s}"], color=c, lw=0.9, label=f"细化尺度 {s}")
        for ax in (a, b):
            ax.set_yscale("log")
            bs = break_step(m)
            if bs is not None:
                ax.axvline(bs, color="k", ls=":", lw=0.8)
        a.set_ylabel(label(m), fontsize=9)
    axs[0, 0].set_title("离自估仿射（px，中位）", fontsize=10)
    axs[0, 1].set_title("离标注（px，中位）", fontsize=10)
    axs[0, 0].legend(fontsize=7, ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel("训练步")
    fig.tight_layout()
    fig.savefig(OUT / "fine.png", dpi=110)
    plt.close(fig)


def main():
    ms = collapsed()
    fig_fine(ms)
    fig_timeline(ms)
    fig_drift(ms)
    fig_converge(ms)
    fig_jumps(ms, healthy())


if __name__ == "__main__":
    main()
