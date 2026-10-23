"""E22 汇总：各方法逐存档的 Val AUC@5、最好存档与 E19 gauss8（只训粗级的对照）的配对自助差，
发散起点，训练日志里粗级与各加噪级的奖励、离参照距离、各部分每步相对改变量的窗口中位数；
第一次启动（因监控代码出错中途退出）的同名方法也列出发散起点。并画两张图。
用法（在仓库根目录、wb 环境）：PYTHONPATH=. python runs/E22/code/summary.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workbench.protocol import auc

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 同 E14、E21
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")   # 对数轴刻度写成纯文本（SimHei 缺数学负号）

E = Path(__file__).resolve().parents[1]
E19 = E.parent / "E19"
MULT = {"f0002": 0.002, "f001": 0.01, "f01": 0.1, "f1": 1, "f10": 10}
AT = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3900]     # 从该步起 100 步的窗口
KEYS = ["d_ref", "rew", "rew_16", "rew_8", "rew_4", "d_16", "d_8", "d_4", "d90_16", "d90_8", "d90_4",
        "upd_coarse", "upd_lora", "upd_fine", "upd_ref16", "upd_ref8", "upd_ref4", "upd_vgg"]


def sweep(d):
    m = json.loads((d / "S" / "metrics.json").read_text())["methods"]
    return {int(k[4:]): v["val"] for k, v in sorted(m.items(), key=lambda kv: int(kv[0][4:])) if "val" in v}


def rows(p):
    r = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [x for x in r if "step" in x]


def onset(rs):
    """离参照距离（d_ref）的 50 步滑动中位数首次超过 100 px 的步；参照拟合失败的步记为缺失。"""
    dr = np.array([r.get("d_ref", np.nan) for r in rs])
    for i in range(len(dr)):
        w = dr[max(0, i - 49):i + 1]
        if np.isfinite(w).any() and np.nanmedian(w) > 100:
            return rs[i]["step"]
    return None


def paired(e1, e0, n=2000, seed=0):
    """同一批验证图对上 AUC@5(e1) − AUC@5(e0) 的配对自助法 95% 区间。"""
    fix = lambda e: np.array([np.inf if x is None else x for x in e], float)   # 没有输出的图对：误差记为无穷
    e1, e0 = fix(e1), fix(e0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return auc(e1, 5.0) - auc(e0, 5.0), np.percentile(diff, [2.5, 97.5])


o = sweep(E19 / "sweep" / "gauss8")
ref = o[max(o, key=lambda k: o[k]["summary"]["auc@5"])]
print("== E19 gauss8 val auc@5:", {k: round(v["summary"]["auc@5"], 3) for k, v in o.items()})
curves, logs = {}, {}
for m in sorted(MULT, key=MULT.get):
    if not (E / "sweep" / m / "S" / "metrics.json").exists():
        print(f"== {m}: 没有评测结果")
        continue
    s = sweep(E / "sweep" / m)
    curves[m] = {k: v["summary"]["auc@5"] for k, v in s.items()}
    st = max(s, key=lambda k: s[k]["summary"]["auc@5"])
    d, ci = paired(s[st]["errors"], ref["errors"])
    rs = rows(E / "ckpt" / m / "log.jsonl")
    logs[m] = rs
    print(f"== {m}（锚点 × {MULT[m]}）  发散起点 {onset(rs)}  参照拟合失败步数 {sum(1 for r in rs if 'd_ref' not in r)}")
    print("  val auc@5 :", {k: round(v, 3) for k, v in curves[m].items()})
    print("  val fail_rate:", {k: round(v["summary"]["fail_rate"], 3) for k, v in s.items()})
    print(f"  最好存档 step {st}: auc@5 {s[st]['summary']['auc@5']:.3f}  − E19 gauss8: {d:+.4f} ci {np.round(ci, 4)}")
    by = {r["step"]: r for r in rs}
    for a in AT:
        w = [by[t] for t in range(a, a + 100) if t in by]
        v = {k: np.median([r[k] for r in w if k in r]) for k in KEYS if any(k in r for r in w)}
        print(f"  step {a:4d}: " + " ".join(f"{k}={x:.3g}" for k, x in v.items()))

print("== 第一次启动（同配置、同种子，监控代码出错后中途退出）")
for d in sorted((E / "ckpt" / "_crash1").glob("f*")):
    if (d / "log.jsonl").exists():
        rs = rows(d / "log.jsonl")
        print(f"  {d.name:6s} 跑到第 {rs[-1]['step']} 步  发散起点 {onset(rs)}")

# 图 1：逐存档的 Val AUC@5
fig, ax = plt.subplots(figsize=(6, 3.6))
for m in sorted(curves, key=MULT.get):
    ax.plot(list(curves[m]), list(curves[m].values()), "o-", ms=3, label=f"× {MULT[m]}")
ax.plot(list(o), [v["summary"]["auc@5"] for v in o.values()], "k--", lw=1, label="只训粗级（E19）")
ax.set_xlabel("训练步数")
ax.set_ylabel("Val AUC@5")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(E / "extra" / "val_auc.png", dpi=150)

# 图 2：离参照距离的 50 步滑动中位数
fig, ax = plt.subplots(figsize=(6, 3.6))
for m in sorted(logs, key=MULT.get):
    rs = logs[m]
    dr = np.array([r.get("d_ref", np.nan) for r in rs])
    roll = [np.nanmedian(dr[max(0, i - 49):i + 1]) if np.isfinite(dr[max(0, i - 49):i + 1]).any() else np.nan
            for i in range(len(dr))]
    ax.plot([r["step"] for r in rs], roll, lw=1, label=f"× {MULT[m]}")
ax.axhline(100, color="k", ls=":", lw=1)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(PLAIN)
ax.set_xlabel("训练步数")
ax.set_ylabel("离参照距离，50 步滑动中位数（px）")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(E / "extra" / "d_ref.png", dpi=150)
