"""E41 表与图（#245 门槛 0.15，#247 加门槛 0.3 / 0.6 / 0.8）。在 gpfs worktree 里跑（ckpt、点对不同步到本机）：

    /opt/envs/loftr/bin/python runs/E41/code/summary.py

写 extra/data/*.csv、extra/*.png，并打印表格。
- 五个运行：E37 all（不筛选，门槛 0，对照，不重跑）、E41 main（门槛 0.15）、E41 main__screen0.3 / 0.6 / 0.8。
  每个方法的产物找两处：本 worktree 的 runs/E41（本票新训的方法），没有就找 gpfs 主 checkout（已结票的 main、E37）。
  横轴为更新次数（每次 4 对）。
1. Val：sweep/<方法>/S/metrics.json 各存档的逐对误差；AUC@5、AUC@3、SR@3、SR@5、误差中位数；评测这个存档的机器
   （sweep/<方法>/S/preds/step<N>/val.meta.json 的 host → GPU 型号）；同一存档位置减 E37 all 的 AUC@5 之差与配对自助法
   95% 区间（同一批 Val 图对有放回重抽 2000 次）；按训练集遍数（每 1977 次更新）分四段的平均。
   总表（overview.csv）：各段平均 AUC@5 与减不筛选之差、最后存档、最高点、全程丢掉的对数、训练墙钟时间。
2. 训练曲线：每次更新的保留比例 = 1 ÷ scr_drawn（日志里为各卡平均取到的对数），200 次更新滑动平均；
   筛选记录（ckpt/<方法>/screen_r*.jsonl）按更新次数分段：取到 / 丢掉的对数、保留比例、取到的对与被丢掉的对的
   参照内点比例；离参照的距离、画面内格子比例、训练图对上当前模型自身拟合的内点比例（日志 n_inl ÷ 5000）五个运行叠画。
3. 开销：日志 sec（每次更新的耗时，含被丢掉的对的参照推理）之和、中位数、训练墙钟时间、平均每次更新取了几对；
   前 500 次更新单独列出。
4. 每个存档在 Val 上，模型自身的内点比例与真实误差的 Spearman 相关：点对（val_matches.npz，按可信度取前 2000 个）
   到这对图估计仿射的距离 ≤ 3 px 的比例，估计失败记 0；真实误差同评测（失败记 ∞）。
5. 训练中被丢掉次数最多的对：各门槛前 20 对的次数、被取到的次数、丢掉时内点比例的中位数（csv），各门槛前 4 对的
   光学与 SAR 原图（图）。
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
MAIN = Y / "moon-exp" / "runs"
DATA = Y / "dataset" / "Moon"
FONT = str(Y / "fonts" / "simhei.ttf")
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
GPU = {"bcb626e99cea": "RTX 3090", "7b3282290ade": "V100"}     # 150、160 的容器 hostname


def where(exp, m):
    """方法的实验目录：本 worktree 里有这个方法的 ckpt 就用它，否则用 gpfs 主 checkout。"""
    here = REPO / "runs" / exp
    return here if (here / "ckpt" / m).exists() else MAIN / exp


# key: (实验, 方法, 门槛, 显示名, 颜色)
RUNS = {"0": ("E37", "all", 0.0, "不筛选（E37 五级）", "0.3"),
        "0.15": ("E41", "main", 0.15, "门槛 0.15", "C0"),
        "0.3": ("E41", "main__screen0.3", 0.3, "门槛 0.3", "C2"),
        "0.6": ("E41", "main__screen0.6", 0.6, "门槛 0.6", "C1"),
        "0.8": ("E41", "main__screen0.8", 0.8, "门槛 0.8", "C3")}
SCREENED = [k for k in RUNS if k != "0"]
N, PASS, UPD, THR_PX = 4, 1977, 7907, 3.0
BINS = ((0, 100), (100, 300), (300, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, UPD))
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
    """{更新次数: Val 记录}，记录里另加 gpu（评测这个存档的机器的 GPU 型号）。"""
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
    dirs = {k: where(e, m) for k, (e, m, *_) in RUNS.items()}
    val = {k: load_val(dirs[k], RUNS[k][1]) for k in RUNS}
    logs = {k: jl(dirs[k] / "ckpt" / RUNS[k][1] / "log.jsonl") for k in RUNS
            if (dirs[k] / "ckpt" / RUNS[k][1] / "log.jsonl").exists()}
    scr = {k: [r for f in sorted((dirs[k] / "ckpt" / RUNS[k][1]).glob("screen_r*.jsonl")) for r in jl(f)]
           for k in SCREENED}

    # 1. Val
    write("val.csv", [{"threshold": k, "update": s, **{x: round(val[k][s]["summary"][x], 3)
                                                        for x in ("auc@5", "auc@3", "sr@3", "sr@5")},
                       "median_err": round(val[k][s]["summary"]["median"], 2), "gpu": val[k][s]["gpu"]}
                      for k in RUNS for s in sorted(val[k])])
    diff = []
    for k in SCREENED:
        for s in sorted(set(val[k]) & set(val["0"])):
            d_, ci = paired(val[k][s]["errors"], val["0"][s]["errors"])
            diff.append({"threshold": k, "update": s, "auc5": round(val[k][s]["summary"]["auc@5"], 4),
                         "auc5_noscreen": round(val["0"][s]["summary"]["auc@5"], 4),
                         "diff": round(d_, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                         "gpu": val[k][s]["gpu"], "gpu_noscreen": val["0"][s]["gpu"]})
    write("vs_e37.csv", diff)
    bp = []
    for k in SCREENED:
        for g in range(1, 5):
            rs = [r for r in diff if r["threshold"] == k and seg_of(r["update"]) == g]
            if rs:
                bp.append({"threshold": k, "segment": g, "n_ckpt": len(rs),
                           "auc5": round(float(np.mean([r["auc5"] for r in rs])), 4),
                           "auc5_noscreen": round(float(np.mean([r["auc5_noscreen"] for r in rs])), 4),
                           "mean_diff": round(float(np.mean([r["diff"] for r in rs])), 4),
                           "n_above": sum(r["ci_lo"] > 0 for r in rs), "n_below": sum(r["ci_hi"] < 0 for r in rs)})
    write("by_segment.csv", bp)

    # 总表
    ov = []
    for k in RUNS:
        v = val[k]
        if not v:
            continue
        row = {"threshold": k, "n_ckpt": len(v)}
        for g in range(1, 5):
            ss = [s for s in v if seg_of(s) == g]
            row[f"seg{g}"] = round(float(np.mean([v[s]["summary"]["auc@5"] for s in ss])), 4) if ss else None
            row[f"seg{g}_diff"] = None if k == "0" else next(
                (r["mean_diff"] for r in bp if r["threshold"] == k and r["segment"] == g), None)
        last = max(v)
        best = max(v, key=lambda s: v[s]["summary"]["auc@5"])
        row.update(last_update=last, last_auc5=round(v[last]["summary"]["auc@5"], 4),
                   last_diff=next((r["diff"] for r in diff if r["threshold"] == k and r["update"] == last), None),
                   best_update=best, best_auc5=round(v[best]["summary"]["auc@5"], 4),
                   discarded=0 if k == "0" else sum(not r["keep"] for r in scr[k]),
                   drawn=(len(scr[k]) if k != "0" else
                          (logs[k][-1]["step"] * N if k in logs else None)),
                   wall_h=None if train_hours(dirs[k], RUNS[k][1]) is None else round(train_hours(dirs[k], RUNS[k][1]), 2))
        ov.append(row)
    write("overview.csv", ov)

    # 2–3. 训练日志、筛选
    cost = []
    for k, log in logs.items():
        drawn = np.array([r.get("scr_drawn", 1.0) for r in log], float) * N
        st = np.array([r["step"] for r in log])
        sec = np.array([r["sec"] for r in log], float)
        e = st <= 500
        cost.append({"threshold": k, "updates": int(st[-1]), "pairs_drawn": int(round(drawn.sum())),
                     "mean_drawn_per_update": round(float(drawn.mean()), 2),
                     "sec_sum_h": round(float(sec.sum()) / 3600, 2), "median_sec": round(float(np.median(sec)), 3),
                     "wall_h": None if train_hours(dirs[k], RUNS[k][1]) is None else round(train_hours(dirs[k], RUNS[k][1]), 2),
                     "first500_pairs_drawn": int(round(drawn[e].sum())),
                     "first500_mean_drawn": round(float(drawn[e].mean()), 2),
                     "first500_sec_sum_h": round(float(sec[e].sum()) / 3600, 3),
                     "first500_median_sec": round(float(np.median(sec[e])), 3),
                     "rest_median_sec": round(float(np.median(sec[~e])), 3) if (~e).any() else None})
    write("cost.csv", cost)
    seg = []
    for k in SCREENED:
        for lo, hi in BINS:
            sr = [r for r in scr[k] if lo < r["step"] <= hi]
            if not sr:
                continue
            ra = np.array([r["ratio"] for r in sr])
            disc = np.array([r["ratio"] for r in sr if not r["keep"]])
            seg.append({"threshold": k, "updates": f"{lo + 1}-{hi}", "drawn": len(sr), "discarded": len(disc),
                        "keep_ratio": round(1 - len(disc) / len(sr), 3),
                        "drawn_per_update": round(len(sr) / (hi - lo), 2),
                        "ratio_median_all": round(float(np.median(ra)), 3),
                        "disc_ratio_median": round(float(np.median(disc)), 3) if len(disc) else None,
                        "disc_fit_fail": int((disc == 0).sum())})
    write("screen_by_segment.csv", seg)

    have = [k for k in SCREENED if k in logs and scr[k]]
    if have:
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
        for k in have:
            log, c, name = logs[k], RUNS[k][4], RUNS[k][3]
            st = np.array([r["step"] for r in log])
            keep = 1.0 / np.array([r.get("scr_drawn", np.nan) for r in log], float)
            ax[0].plot(st, roll(keep, 200, np.nanmean), color=c, lw=1.1, label=name)
            disc = [r["ratio"] for r in scr[k] if not r["keep"]]
            if disc:
                ax[1].hist(disc, bins=np.linspace(0, 0.8, 41), histtype="step", density=True, lw=1.1, color=c,
                           label=f"{name}（{len(disc)} 对）")
            ra = np.array([r["ratio"] for r in scr[k]])
            sa = np.array([r["step"] for r in scr[k]])
            o = np.argsort(sa, kind="stable")
            us, first = np.unique(sa[o], return_index=True)
            med = [np.median(x) for x in np.split(ra[o], first[1:])]
            ax[2].plot(us, roll(med, 200), color=c, lw=1.1, label=name)
            ax[1].axvline(RUNS[k][2], color=c, ls=":", lw=0.8)
            ax[2].axhline(RUNS[k][2], color=c, ls=":", lw=0.8)
        for g in range(1, 4):
            ax[0].axvline(g * PASS, color="0.85", lw=0.8)
        for a in (ax[0], ax[2]):
            a.set_xscale("symlog", linthresh=100)
            a.set_xlim(0, UPD)
        ax[0].set(xlabel="更新次数（前 100 次线性、之后对数）", ylabel="保留比例（200 次更新滑动平均）")
        ax[1].set(xlabel="被丢掉的对的参照内点比例（全程）", ylabel="密度")
        ax[2].set(xlabel="更新次数（前 100 次线性、之后对数）", ylabel="取到的对的参照内点比例（每步中位数，200 步滑动中位数）")
        for a in ax:
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "screen.png", dpi=110)

    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
    for k, (_, _, _, name, c) in RUNS.items():
        if k in logs:
            log = logs[k]
            st = np.array([r["step"] for r in log])
            for a, key in zip(ax[:3], ("d_ref", "in_frame", "n_inl")):
                v = [r.get(key, np.nan) / (5000.0 if key == "n_inl" else 1.0) for r in log]
                a.plot(st, roll(v, 50), color=c, lw=1.0, label=name)
        if val[k]:
            s_ = sorted(val[k])
            ax[3].plot(s_, [val[k][s]["summary"]["auc@5"] for s in s_], "o-", color=c, lw=1.0, ms=2.5, label=name)
    ax[0].set_yscale("log")
    for a, lab in zip(ax, ("离参照的距离（px）", "画面内格子比例", "训练图对上模型自身拟合的内点比例", "Val AUC@5")):
        a.set(ylabel=lab, xlabel="更新次数")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "extra" / "curves.png", dpi=110)

    if diff:
        fig, a_ = plt.subplots(figsize=(9, 4.6))
        for j, k in enumerate(SCREENED):
            rs = [r for r in diff if r["threshold"] == k]
            if not rs:
                continue
            x = np.array([r["update"] for r in rs]) + (j - 1.5) * 40
            y = np.array([r["diff"] for r in rs])
            a_.errorbar(x, y, yerr=[y - [r["ci_lo"] for r in rs], [r["ci_hi"] for r in rs] - y], fmt="o",
                        color=RUNS[k][4], ms=3, capsize=1.5, lw=0.7, label=RUNS[k][3])
        a_.axhline(0, color="0.5", ls=":", lw=0.8)
        for g in range(1, 4):
            a_.axvline(g * PASS, color="0.85", lw=0.8)
        a_.set(xlabel="更新次数", ylabel="Val AUC@5 之差（减不筛选）")
        a_.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "diff.png", dpi=110)

    # 4. 自身内点比例与真实误差的 Spearman
    ds = Dataset(DATA)
    pairs = ds.labelled("val")
    sp = []
    for k in RUNS:
        for s in sorted(val[k]):
            o = own_ratio(dirs[k], RUNS[k][1], s, ds, pairs)
            if o is not None:
                sp.append({"threshold": k, "update": s, "spearman": round(spearman(*o), 3),
                           "own_ratio_median": round(float(np.median(o[0])), 3)})
    write("spearman.csv", sp)
    if sp:
        fig, a = plt.subplots(figsize=(8, 4.4))
        for k in RUNS:
            rs = [r for r in sp if r["threshold"] == k]
            if rs:
                a.plot([r["update"] for r in rs], [r["spearman"] for r in rs], "o-", color=RUNS[k][4], ms=2.5,
                       lw=1.0, label=RUNS[k][3])
        a.set(xlabel="更新次数", ylabel="自身内点比例与真实误差的 Spearman 相关")
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "spearman.png", dpi=110)

    # 5. 被丢掉最多次的对
    top, n_pairs = {}, []
    for k in SCREENED:
        if not scr[k]:
            continue
        n_draw = collections.Counter(r["pair"] for r in scr[k])
        n_disc = collections.Counter(r["pair"] for r in scr[k] if not r["keep"])
        rat = collections.defaultdict(list)
        for r in scr[k]:
            if not r["keep"]:
                rat[r["pair"]].append(r["ratio"])
        top[k] = [{"threshold": k, "pair": q, "n_disc": c, "n_drawn": n_draw[q],
                   "disc_ratio_median": round(float(np.median(rat[q])), 3)} for q, c in n_disc.most_common(20)]
        n_pairs.append({"threshold": k, "distinct_drawn": len(n_draw), "ever_discarded": len(n_disc),
                        "never_discarded": sum(1 for q in n_draw if q not in n_disc),
                        "discarded_every_time": sum(1 for q in n_disc if n_disc[q] == n_draw[q])})
    write("most_discarded.csv", [r for k in top for r in top[k]])
    write("discard_pairs.csv", n_pairs)
    if top:
        from baselines.data import Data
        from moonlib import inputs
        dd = Data(str(DATA))
        mo, ms = inputs.get("optical", "div255"), inputs.get("sar", "p2p98")
        ks = list(top)
        fig, ax = plt.subplots(2 * len(ks), 4, figsize=(12, 6.2 * len(ks)))
        for i, k in enumerate(ks):
            for j in range(4):
                for a in ax[2 * i:2 * i + 2, j]:
                    a.axis("off")
                if j >= len(top[k]):
                    continue
                r = top[k][j]
                ax[2 * i, j].imshow(mo(dd.optical("train", r["pair"])), cmap="gray")
                ax[2 * i + 1, j].imshow(ms(dd.sar("train", r["pair"])), cmap="gray")
                ax[2 * i, j].set_title(f"{RUNS[k][3]}：{r['pair']}\n丢 {r['n_disc']} / 取 {r['n_drawn']}，"
                                       f"比例中位数 {r['disc_ratio_median']}", fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "extra" / "most_discarded.png", dpi=80)


if __name__ == "__main__":
    main()
