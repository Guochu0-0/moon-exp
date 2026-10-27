"""E35 表与图（#230）：各 τ 组是否提前停、训练日志的判据曲线与当前模型离 EMA 参照的距离、各存档的 Val AUC@5，以及与 τ = 0.999 组的比较。
- 一组 = 一个 τ（方法 tau<值>），从头训到 4000 次更新，每 1000 次存档。
- 日志取 ckpt/<方法>/log.jsonl。判据量为离参照的距离 d_ref 与画面内格子比例 in_frame 的 50 次更新滑动中位数（同 E32）；
  当前模型与 EMA 参照的距离取 ref_gap（两者各自拟合的仿射把各格中心映射到的位置之间的平均距离，px，同 E30），
  报 50 次更新滑动中位数的曲线和最后 500 次更新的中位数。
- 提前停取 ckpt/<方法>/halt.json（停在第几次更新、哪个量越线）。
- Val 取 sweep/<方法>/S/metrics.json（各存档 step<N>），AUC@5 与配准误差中位数。
- 比较口径（#230，同 #228）：每组报最后一个存档的 Val AUC@5，和最后两个存档的平均值；与 τ = 0.999 组在同样的存档上比，
  差值用配对自助法给 95% 区间（同一批验证图有放回重抽 2000 次，同 E32、E34 的 summary.py），两个存档平均时每次重抽
  算两个存档各自的 AUC@5 再平均。区间跨过 0 记为「分不出高下」。
在 gpfs 上跑（ckpt 不同步到本机）：

    /opt/envs/loftr/bin/python runs/E35/code/summary.py
写 extra/data/summary.csv、extra/data/val.csv、extra/curves.png，并打印表格。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parents[1]))
from workbench.protocol import auc  # noqa: E402

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
REF = "0.999"
D_MAX, IN_MIN, WIN = 100.0, 0.5, 50
KEYS = ("d_ref", "in_frame", "ref_gap")
GAP_TAIL = 500     # 报最后这么多次更新内 ref_gap 的中位数


def groups():
    """τ（字符串，如 0.999）→ 方法名列表（每组一个方法 tau<值>）。按记忆长度 1/(1−τ) 从短到长排。"""
    out = {p.stem[3:]: [p.stem] for p in (R / "configs").glob("tau*.toml")}
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


def load_log(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.startswith("{")]
    return np.array([r["step"] for r in rows]), {k: np.array([r.get(k, np.nan) for r in rows], float) for k in KEYS}


def roll(x, win):
    return np.array([np.nanmedian(x[max(0, i - win + 1):i + 1]) if np.isfinite(x[max(0, i - win + 1):i + 1]).any()
                     else np.nan for i in range(len(x))])


def fix(e):
    return np.array([np.inf if x is None else x for x in e], float)


def paired(e1s, e0s, n=2000, seed=0):
    """mean_k AUC@5(e1s[k]) − mean_k AUC@5(e0s[k]) 与配对自助法 95% 区间。e1s、e0s 为同样几个存档的逐对误差，
    各存档用同一组重抽下标。"""
    e1s, e0s = [fix(e) for e in e1s], [fix(e) for e in e0s]
    f = lambda es, i: np.mean([auc(e[i], 5.0) for e in es])
    full = np.arange(len(e1s[0]))
    idx = np.random.default_rng(seed).integers(0, len(full), (n, len(full)))
    diff = [f(e1s, i) - f(e0s, i) for i in idx]
    return f(e1s, full) - f(e0s, full), np.percentile(diff, [2.5, 97.5])


def r3(x):
    return "" if x is None or not np.isfinite(x) else round(float(x), 3)


def fmt(d, ci):
    tag = "" if ci[0] > 0 or ci[1] < 0 else "，分不出高下"
    return f"{d:+.3f}（{ci[0]:+.3f}，{ci[1]:+.3f}{tag}）"


def collect(ms):
    val, steps, logs, halt = {}, [], [], None
    for m in ms:
        ck = R / "ckpt" / m
        if (ck / "log.jsonl").exists():
            logs.append(load_log(ck / "log.jsonl"))
        if (ck / "halt.json").exists():
            halt = json.loads((ck / "halt.json").read_text())
        met = R / "sweep" / m / "S" / "metrics.json"
        if met.exists():
            for k, v in json.loads(met.read_text(encoding="utf-8"))["methods"].items():
                if k.startswith("step") and "val" in v:
                    val[int(k[4:])] = v["val"]
    if logs:
        steps = np.concatenate([s for s, _ in logs])
        x = {k: np.concatenate([d[k] for _, d in logs]) for k in KEYS}
    else:
        x = {k: np.array([]) for k in KEYS}
    return val, steps, x, halt


def main():
    G = groups()
    data = {g: collect(ms) for g, ms in G.items()}
    ref_val = data[REF][0] if REF in data else {}
    rows, vrows = [], []
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
    for i, (g, (val, steps, x, halt)) in enumerate(data.items()):
        name = f"τ = {g}（记忆约 {1 / (1 - float(g)):.0f} 次）"
        if len(steps):
            d_s, in_s = roll(x["d_ref"], WIN), roll(x["in_frame"], WIN)
            ax[0].plot(steps, d_s, color=f"C{i}", lw=1.1, label=name)
            ax[1].plot(steps, in_s, color=f"C{i}", lw=1.1, label=name)
            ax[2].plot(steps, roll(x["ref_gap"], WIN), color=f"C{i}", lw=1.1, label=name)
            tail = x["ref_gap"][steps > steps[-1] - GAP_TAIL]
            gap_tail = np.nanmedian(tail) if np.isfinite(tail).any() else np.nan
        else:
            d_s = in_s = np.array([np.nan])
            gap_tail = np.nan
        ss = sorted(val)
        for s in ss:
            vrows.append({"tau": g, "update": s, "auc5": r3(val[s]["summary"]["auc@5"]),
                          "median_px": round(val[s]["summary"]["median"], 2)})
        if ss:
            ax[3].plot(ss, [val[s]["summary"]["auc@5"] for s in ss], "o-", color=f"C{i}", lw=1.1, label=name)
        if halt is not None:
            for a in ax[:3]:
                a.axvline(halt["step"], color=f"C{i}", ls=":", lw=0.8)
        row = {"tau": g, "memory": round(1 / (1 - float(g))), "methods": "+".join(G[g]), "updates": int(steps[-1]) if len(steps) else 0,
               "halt": "" if halt is None else f"第 {halt['step']} 次：{halt['key']} {halt['op']} {halt['thr']}（中位数 {halt['median']}）",
               "d_ref_max_smooth": r3(np.nanmax(d_s)) if np.isfinite(d_s).any() else "",
               "in_frame_min_smooth": r3(np.nanmin(in_s)) if np.isfinite(in_s).any() else "",
               f"ref_gap_last{GAP_TAIL}_median": r3(gap_tail)}
        for s in (1000, 2000, 3000, 4000):
            row[f"auc5_{s}"] = r3(val[s]["summary"]["auc@5"]) if s in val else ""
        last, last2 = ss[-1:], ss[-2:]
        row["last_ckpt"] = last[0] if last else ""
        row["auc5_last"] = r3(val[last[0]]["summary"]["auc@5"]) if last else ""
        row["auc5_last2_mean"] = r3(np.mean([val[s]["summary"]["auc@5"] for s in last2])) if len(last2) == 2 else ""
        for tag, sel in (("last", last), ("last2", last2)):
            ok = sel and len(sel) == (1 if tag == "last" else 2) and all(s in ref_val for s in sel) and g != REF
            row[f"vs_{REF}_{tag}"] = fmt(*paired([val[s]["errors"] for s in sel], [ref_val[s]["errors"] for s in sel])) \
                if ok else ""
        rows.append(row)
    ax[0].axhline(D_MAX, color="0.5", ls="--", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("离参照的距离（px，50 次更新滑动中位数）")
    ax[1].axhline(IN_MIN, color="0.5", ls="--", lw=0.8)
    ax[1].set_ylabel("画面内格子比例（50 次更新滑动中位数）")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("当前模型与 EMA 参照的距离（px，50 次更新滑动中位数）")
    ax[3].set_ylabel("Val AUC@5")
    for a in ax:
        a.set_xlabel("参数更新次数")
        a.legend(fontsize=8)
    fig.tight_layout()
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    fig.savefig(R / "extra" / "curves.png", dpi=110)
    for fn, rs in (("summary.csv", rows), ("val.csv", vrows)):
        with open(R / "extra" / "data" / fn, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    for r in rows:
        print(" | ".join(f"{k}={r[k]}" for k in r))


if __name__ == "__main__":
    main()
