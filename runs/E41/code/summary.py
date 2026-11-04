"""E41 表与图（#245）。在 gpfs worktree 里跑（ckpt、点对不同步到本机；E37 的读 gpfs 主 checkout）：

    /opt/envs/loftr/bin/python runs/E41/code/summary.py

写 extra/data/*.csv、extra/*.png，并打印表格。
- 两个运行：E41 main（加样本筛选）与 E37 all（同配置不筛选，对照，不重跑）。横轴为更新次数（每次 4 对）；
  E41 每次更新取到的对数多于 4，另给「已取图对数」。
1. Val：sweep/<方法>/S/metrics.json 各存档的逐对误差；AUC@5、AUC@3、SR@3、SR@5、误差中位数；同一存档位置 E41 减 E37 的
   AUC@5 之差与配对自助法 95% 区间（同一批 Val 图对有放回重抽 2000 次，同 E37）；按训练集遍数（每 1977 次更新）分段的平均。
2. 训练曲线：每次更新的保留比例 = 4 ÷ 四张卡共取到的对数（日志 scr_drawn 为各卡平均，故 = 1 ÷ scr_drawn），
   200 次更新滑动平均；被丢掉的对的内点比例分布（ckpt/main/screen_r*.jsonl，按训练进程分四段）；离参照的距离、
   画面内格子比例、训练图对上当前模型自身拟合的内点比例（日志 n_inl ÷ 5000）与 E37 all 叠画（50 次更新滑动中位数）。
3. 开销：两边日志 sec（每次更新的耗时，E41 含被丢掉的对的参照推理）之和与训练的墙钟时间；平均每次更新取了几对。
4. 每个存档在 Val 上，模型自身的内点比例与真实误差的 Spearman 相关：点对（val_matches.npz，按可信度取前 2000 个）
   到这对图估计仿射的距离 ≤ 3 px 的比例（同工作台逐对视图的内点判定），估计失败记 0；真实误差同评测（失败记 ∞）。
5. 训练中被丢掉次数最多的对：次数、被取到的次数、丢掉时内点比例的中位数（csv），前 6 对的光学与 SAR 原图（图）。
"""
from __future__ import annotations

import collections
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
from workbench.protocol import auc  # noqa: E402

Y = Path("/remote-home/xufang/YGC")
E37 = Y / "moon-exp" / "runs" / "E37"
DATA = Y / "dataset" / "Moon"
FONT = str(Y / "fonts" / "simhei.ttf")
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RUNS = {"E41": (R, "main", "加样本筛选（E41）", "C3"), "E37": (E37, "all", "不筛选（E37 五级）", "C0")}
N, PASS, UPD, THR_PX = 4, 1977, 7907, 3.0
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
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"== {name}")
    for r in rows:
        print(" | ".join(f"{k}={r[k]}" for k in r))


def load_val(d, m):
    """{更新次数: Val 记录}"""
    met = d / "sweep" / m / "S" / "metrics.json"
    if not met.exists():
        return {}
    return {int(k[4:]): v["val"] for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items()
            if k.startswith("step") and "val" in v}


def rank(x):
    """平均秩（并列取平均；∞ 照常排在最后）。"""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def own_ratio(d, m, st, ds, pairs):
    """一个存档：每对 Val 图（ds.labelled 顺序）的 (自身内点比例, 真实误差)。"""
    p = d / "sweep" / m / "S" / "preds" / f"step{st}"
    if not (p / "val_matches.npz").exists():
        return None
    preds = {}
    for line in (p / "val.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["pair"]] = r.get("A")
    ratio, err = [], []
    with np.load(p / "val_matches.npz") as z:
        for q in pairs:
            A, key = preds.get(q), q.replace("/", "__")
            if A is None or key not in z.files or len(z[key]) == 0:
                ratio.append(0.0)
            else:
                x, a = z[key].astype(np.float64), np.asarray(A, np.float64)
                res = np.linalg.norm(x[:, :2] @ a[:, :2].T + a[:, 2] - x[:, 2:4], axis=1)
                ratio.append(float((res <= THR_PX).mean()))
            err.append(pair_error(ds, "val", q, A))
    return np.array(ratio), np.array(err, float)


def train_hours(d, m):
    """训练的墙钟时间：args.json（开训时写）到最后一个存档的修改时间。启动记录的起止含等评测的时间，不用它。"""
    ck = d / "ckpt" / m
    last = max(ck.glob("ckpt_*.pt"), key=lambda q: int(q.stem[5:]), default=None)
    if last is None or not (ck / "args.json").exists():
        return None
    return (last.stat().st_mtime - (ck / "args.json").stat().st_mtime) / 3600


def main():
    (R / "extra").mkdir(exist_ok=True)
    val = {k: load_val(d, m) for k, (d, m, _, _) in RUNS.items()}
    # 1. Val
    write("val.csv", [{"run": k, "update": s, "pairs_learned": s * N,
                       **{x: round(val[k][s]["summary"][x], 3) for x in ("auc@5", "auc@3", "sr@3", "sr@5")},
                       "median_err": round(val[k][s]["summary"]["median"], 2)} for k in RUNS for s in sorted(val[k])])
    ss = sorted(set(val["E41"]) & set(val["E37"]))
    diff = []
    for s in ss:
        d_, ci = paired(val["E41"][s]["errors"], val["E37"][s]["errors"])
        diff.append({"update": s, "auc5_E41": round(val["E41"][s]["summary"]["auc@5"], 3),
                     "auc5_E37": round(val["E37"][s]["summary"]["auc@5"], 3),
                     "diff": round(d_, 3), "ci_lo": round(ci[0], 3), "ci_hi": round(ci[1], 3)})
    if diff:
        write("vs_e37.csv", diff)
        bp = []
        for k in range(1, 5):
            rs = [r for r in diff if (k - 1) * PASS < r["update"] <= (k * PASS if k < 4 else UPD)]
            if rs:
                bp.append({"segment": k, "n_ckpt": len(rs),
                           "auc5_E41": round(float(np.mean([r["auc5_E41"] for r in rs])), 4),
                           "auc5_E37": round(float(np.mean([r["auc5_E37"] for r in rs])), 4),
                           "mean_diff": round(float(np.mean([r["diff"] for r in rs])), 4),
                           "n_above": sum(r["ci_lo"] > 0 for r in rs), "n_below": sum(r["ci_hi"] < 0 for r in rs)})
        write("by_segment.csv", bp)

    # 2–3. 训练日志
    logs = {k: jl(d / "ckpt" / m / "log.jsonl") for k, (d, m, _, _) in RUNS.items()
            if (d / "ckpt" / m / "log.jsonl").exists()}
    scr = []
    for f in sorted((R / "ckpt" / "main").glob("screen_r*.jsonl")):
        scr += jl(f)
    cost = []
    for k, log in logs.items():
        drawn = np.array([r.get("scr_drawn", 1.0) for r in log], float) * N
        cost.append({"run": k, "updates": log[-1]["step"], "pairs_drawn": int(round(drawn.sum())),
                     "mean_drawn_per_update": round(float(drawn.mean()), 2),
                     "sec_sum_h": round(sum(r["sec"] for r in log) / 3600, 2),
                     "median_sec": round(float(np.median([r["sec"] for r in log])), 3),
                     "wall_h": None if train_hours(*RUNS[k][:2]) is None else round(train_hours(*RUNS[k][:2]), 2)})
    if cost:
        write("cost.csv", cost)
    if "E41" in logs:
        log = logs["E41"]
        st = np.array([r["step"] for r in log])
        keep = 1.0 / np.array([r.get("scr_drawn", np.nan) for r in log], float)
        seg = []
        for k in range(1, 5):
            m_ = (st > (k - 1) * PASS) & (st <= (k * PASS if k < 4 else UPD))
            sr = [r for r in scr if (k - 1) * PASS < r["step"] <= (k * PASS if k < 4 else UPD)]
            disc = np.array([r["ratio"] for r in sr if not r["keep"]])
            kept = np.array([r["ratio"] for r in sr if r["keep"]])
            seg.append({"segment": k, "keep_ratio": round(float(len(kept) / max(1, len(sr))), 3),
                        "drawn_per_update": round(float(len(sr) / max(1, m_.sum())), 2),
                        "kept_ratio_median": round(float(np.median(kept)), 3) if len(kept) else None,
                        "disc_ratio_median": round(float(np.median(disc)), 3) if len(disc) else None,
                        "disc_ratio_p90": round(float(np.quantile(disc, 0.9)), 3) if len(disc) else None,
                        "disc_fit_fail": int((disc == 0).sum())})
        write("screen_by_segment.csv", seg)
        early = []
        for lo, hi in ((0, 100), (100, 200), (200, 300), (300, 500), (500, 750), (750, 1000), (1000, 1500),
                       (1500, 1977), (1977, UPD)):
            sr = [r for r in scr if lo < r["step"] <= hi]
            ra = np.array([r["ratio"] for r in sr])
            early.append({"updates": f"{lo + 1}-{hi}", "drawn": len(sr), "discarded": sum(not r["keep"] for r in sr),
                          "keep_ratio": round(sum(r["keep"] for r in sr) / max(1, len(sr)), 3),
                          "ratio_median_all": round(float(np.median(ra)), 3)})
        write("screen_early.csv", early)

        fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
        ax[0].plot(st, roll(keep, 200, np.nanmean), color="C3", lw=1.1)
        for k in range(1, 4):
            ax[0].axvline(k * PASS, color="0.8", lw=0.8)
        ax[0].set_ylabel("保留比例（200 次更新滑动平均）")
        ax[0].set_xlabel("更新次数")
        bins = np.linspace(0, max(0.05, max(r["ratio"] for r in scr)), 41)
        for k in range(1, 5):
            disc = [r["ratio"] for r in scr if not r["keep"] and (k - 1) * PASS < r["step"] <= (k * PASS if k < 4 else UPD)]
            if disc:
                ax[1].hist(disc, bins=bins, histtype="step", density=True, lw=1.1, label=f"第 {k} 段")
        thr = json.loads((R / "extra" / "data" / "threshold.json").read_text(encoding="utf-8"))["threshold"]
        ax[1].axvline(thr, color="0.4", ls=":", lw=0.8)
        ax[1].set_xlabel("被丢掉的对的参照内点比例")
        ax[1].set_ylabel("密度")
        ax[1].legend(fontsize=8)
        ratio_all = np.array([r["ratio"] for r in scr])
        step_all = np.array([r["step"] for r in scr])
        ax[2].plot(np.unique(step_all), [np.median(ratio_all[step_all == s]) for s in np.unique(step_all)],
                   ",", color="0.6", alpha=0.3)
        med = roll(np.array([np.median(ratio_all[step_all == s]) for s in np.unique(step_all)]), 200)
        ax[2].plot(np.unique(step_all), med, color="C3", lw=1.1, label="取到的全部对（每步中位数，200 步滑动中位数）")
        ax[2].axhline(thr, color="0.4", ls=":", lw=0.8)
        ax[2].set_xlabel("更新次数")
        ax[2].set_ylabel("参照内点比例")
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "screen.png", dpi=110)

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.4))
    for k, (d, m, name, c) in RUNS.items():
        if k in logs:
            log = logs[k]
            st = np.array([r["step"] for r in log])
            for a, key in zip(ax[:3], ("d_ref", "in_frame", "n_inl")):
                v = [r.get(key, np.nan) / (5000.0 if key == "n_inl" else 1.0) for r in log]
                a.plot(st, roll(v, 50), color=c, lw=1.0, label=name)
        if val[k]:
            s_ = sorted(val[k])
            ax[3].plot(s_, [val[k][s]["summary"]["auc@5"] for s in s_], "o-", color=c, lw=1.1, ms=3, label=name)
    ax[0].set_yscale("log")
    for a, lab in zip(ax, ("离参照的距离（px）", "画面内格子比例", "训练图对上模型自身拟合的内点比例", "Val AUC@5")):
        a.set_ylabel(lab)
        a.set_xlabel("更新次数")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)

    # 4. 自身内点比例与真实误差的 Spearman
    ds = Dataset(DATA)
    pairs = ds.labelled("val")
    sp = []
    for k, (d, m, _, _) in RUNS.items():
        for s in sorted(val[k]):
            o = own_ratio(d, m, s, ds, pairs)
            if o is not None:
                sp.append({"run": k, "update": s, "spearman": round(spearman(*o), 3),
                           "own_ratio_median": round(float(np.median(o[0])), 3)})
    if sp:
        write("spearman.csv", sp)
        fig, a = plt.subplots(figsize=(7, 4.2))
        for k, (_, _, name, c) in RUNS.items():
            rs = [r for r in sp if r["run"] == k]
            a.plot([r["update"] for r in rs], [r["spearman"] for r in rs], "o-", color=c, ms=3, lw=1.0, label=name)
        a.set_xlabel("更新次数")
        a.set_ylabel("自身内点比例与真实误差的 Spearman 相关")
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "spearman.png", dpi=110)

    # 5. 被丢掉最多次的对
    if scr:
        n_draw = collections.Counter(r["pair"] for r in scr)
        n_disc = collections.Counter(r["pair"] for r in scr if not r["keep"])
        rat = collections.defaultdict(list)
        for r in scr:
            if not r["keep"]:
                rat[r["pair"]].append(r["ratio"])
        top = [{"pair": q, "n_disc": c, "n_drawn": n_draw[q], "disc_ratio_median": round(float(np.median(rat[q])), 3)}
               for q, c in n_disc.most_common(20)]
        write("most_discarded.csv", top)
        never = sum(1 for q in n_draw if q not in n_disc)
        print(f"取到过的不同对 {len(n_draw)}，从未被丢掉的 {never}，被丢掉过的 {len(n_disc)}")
        from baselines.data import Data
        from moonlib import inputs
        dd = Data(str(DATA))
        mo, ms = inputs.get("optical", "div255"), inputs.get("sar", "p2p98")
        fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
        for j, r in enumerate(top[:6]):
            ax[0, j].imshow(mo(dd.optical("train", r["pair"])), cmap="gray")
            ax[1, j].imshow(ms(dd.sar("train", r["pair"])), cmap="gray")
            ax[0, j].set_title(f"{r['pair']}\n丢 {r['n_disc']} / 取 {r['n_drawn']}，比例中位数 {r['disc_ratio_median']}",
                               fontsize=8)
            for a in ax[:, j]:
                a.axis("off")
        fig.tight_layout()
        fig.savefig(R / "extra" / "most_discarded.png", dpi=90)


if __name__ == "__main__":
    main()
