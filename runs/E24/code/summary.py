"""E24 汇总（同 E22 summary.py 的口径）：各方法逐存档的 Val AUC@5、最好存档与 E19 gauss8 的配对自助差，发散起点，
训练日志里粗级与各级的奖励、离参照距离（含正常推理轨迹上各级 μ 的距离 dmu_s）、各部分每步相对改变量、
局部抽签与精确梯度的余弦相似度的窗口中位数；E22 f01、f001 直接引用。画图。
用法（在仓库根目录、wb 环境）：PYTHONPATH=. python runs/E24/code/summary.py
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
E22 = E.parent / "E22"
METHODS = ["g1_s0", "g1_s1", "g01_s0", "g01_s1", "l1_s0", "l1_s1", "l01_s0", "l01_s1", "c01_s0"]
AT = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3900]     # 从该步起 100 步的窗口
KEYS = ["d_ref", "rew", "rew_16", "rew_8", "rew_4", "d_16", "d_8", "d_4", "dmu_16", "dmu_8", "dmu_4", "dmu_2", "dmu_1",
        "upd_coarse", "upd_lora", "upd_fine", "upd_ref16", "upd_ref8", "upd_ref4", "upd_vgg", "gnorm",
        "cos", "cos_16", "cos_8", "cos_4", "cosf_16", "cosf_8", "cosf_4"]


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
for m in ["E22/f01", "E22/f001"] + METHODS:
    root = (E22 / "sweep" / m[4:]) if m.startswith("E22/") else (E / "sweep" / m)
    if not (root / "S" / "metrics.json").exists():
        print(f"== {m}: 没有评测结果")
        continue
    s = sweep(root)
    curves[m] = {k: v["summary"]["auc@5"] for k, v in s.items()}
    st = max(s, key=lambda k: s[k]["summary"]["auc@5"])
    d, ci = paired(s[st]["errors"], ref["errors"])
    print(f"== {m}")
    print("  val auc@5 :", {k: round(v, 3) for k, v in curves[m].items()})
    print("  val fail_rate:", {k: round(v["summary"]["fail_rate"], 3) for k, v in s.items()})
    print(f"  最好存档 step {st}: auc@5 {s[st]['summary']['auc@5']:.3f}  − E19 gauss8: {d:+.4f} ci {np.round(ci, 4)}")
    if m.startswith("E22/"):
        continue
    rs = rows(E / "ckpt" / m / "log.jsonl")
    logs[m] = rs
    print(f"  发散起点 {onset(rs)}  参照拟合失败步数 {sum(1 for r in rs if 'd_ref' not in r)}"
          f"  每步耗时中位数 {np.median([r['sec'] for r in rs]):.2f}s  显存 {rs[0].get('mem_gb')}")
    by = {r["step"]: r for r in rs}
    for a in AT:
        w = [by[t] for t in range(a, a + 100) if t in by]
        v = {k: np.median([r[k] for r in w if k in r]) for k in KEYS if any(k in r for r in w)}
        print(f"  step {a:4d}: " + " ".join(f"{k}={x:.3g}" for k, x in v.items()))
    c = [r["cos"] for r in rs if "cos" in r]
    if c:
        print(f"  cos：{len(c)} 次，中位数 {np.median(c):.3f}，最小 {min(c):.3f}，> 0 的比例 {np.mean(np.array(c) > 0):.3f}")

STY = {"g": "-", "l": "--", "c": ":"}
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for m in curves:
    ax.plot(list(curves[m]), list(curves[m].values()), STY.get(m[0], "-."), marker="o", ms=3, lw=1, label=m)
ax.plot(list(o), [v["summary"]["auc@5"] for v in o.values()], "k-", lw=2, alpha=0.4, label="E19 gauss8")
ax.set_xlabel("训练步数")
ax.set_ylabel("Val AUC@5")
ax.legend(fontsize=6, ncol=3)
fig.tight_layout()
fig.savefig(E / "extra" / "val_auc.png", dpi=150)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for m, rs in logs.items():
    for ax, key in zip(axs, ("d_ref", "dmu_4")):
        dr = np.array([r.get(key, np.nan) for r in rs])
        roll = [np.nanmedian(dr[max(0, i - 49):i + 1]) if np.isfinite(dr[max(0, i - 49):i + 1]).any() else np.nan
                for i in range(len(dr))]
        ax.plot([r["step"] for r in rs], roll, STY.get(m[0], "-"), lw=1, label=m)
for ax, t in zip(axs, ("粗匹配离参照距离", "正常推理尺度 4 输出离参照距离")):
    ax.axhline(100, color="k", ls=":", lw=0.8) if t.startswith("粗") else None
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("训练步数")
    ax.set_ylabel(t + "，50 步滑动中位数（px）", fontsize=8)
axs[0].legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(E / "extra" / "dist.png", dpi=150)
