"""E23 汇总：各方法的发散起点、逐存档 Val AUC@5、最好存档与 E19 gauss8 的配对自助差，
训练日志里粗匹配分数、离参照距离、各部分每步相对改变量的窗口中位数；并画三张图。
对照：E19 gauss8（只训粗匹配、每对 16 次、种子 0）与 E22 f0002（细化学习率 = 锚点 × 0.002、种子 0），
从 gpfs 主 checkout 读（环境变量 MAIN_RUNS 可覆盖）。
用法（在仓库根目录、wb 环境）：PYTHONPATH=. python runs/E23/code/summary.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workbench.protocol import auc

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 同 E14、E21、E22
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")

E = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get("MAIN_RUNS", "/remote-home/xufang/YGC/moon-exp/runs"))
# 方法 -> (实验目录, 方法名, 图例)
RUNS = {
    "c16_s0": (MAIN / "E19", "gauss8", "16 次 种子 0（E19）"),
    "c16_s1": (E, "c16_s1", "16 次 种子 1"),
    "c16_s2": (E, "c16_s2", "16 次 种子 2"),
    "c8_s0": (E, "c8_s0", "8 次 种子 0"),
    "c8_s1": (E, "c8_s1", "8 次 种子 1"),
    "c8_s2": (E, "c8_s2", "8 次 种子 2"),
    "f0002_s0": (MAIN / "E22", "f0002", "× 0.002 种子 0（E22）"),
    "f0002_s1": (E, "f0002_s1", "× 0.002 种子 1"),
    "f0002_s2": (E, "f0002_s2", "× 0.002 种子 2"),
}
AT = [1, 400, 500, 600, 700, 1000, 2000, 3000, 3900]     # 从该步起 100 步的窗口
FINE = ["c16_s1", "c16_s2"]
KEYS = ["d_ref", "rew", "H", "pmode", "gnorm", "upd_coarse", "upd_lora", "upd_fine", "upd_ref16", "upd_vgg"]


def sweep(d):
    p = d / "S" / "metrics.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())["methods"]
    return {int(k[4:]): v["val"] for k, v in sorted(m.items(), key=lambda kv: int(kv[0][4:])) if "val" in v}


def rows(p):
    if not p.exists():
        return []
    r = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [x for x in r if "step" in x]


def roll(rs, k="d_ref", w=50):
    x = np.array([r.get(k, np.nan) for r in rs], float)
    return np.array([np.nanmedian(x[max(0, i - w + 1):i + 1]) if np.isfinite(x[max(0, i - w + 1):i + 1]).any()
                     else np.nan for i in range(len(x))])


def onset(rs):
    """E20 的定义：离参照距离（d_ref）的 50 步滑动中位数首次超过 100 px 的步；参照拟合失败的步记为缺失。"""
    r = roll(rs)
    i = np.flatnonzero(r > 100)
    return rs[i[0]]["step"] if len(i) else None


def paired(e1, e0, n=2000, seed=0):
    """同一批验证图对上 AUC@5(e1) − AUC@5(e0) 的配对自助法 95% 区间。"""
    fix = lambda e: np.array([np.inf if x is None else x for x in e], float)   # 没有输出的图对：误差记为无穷
    e1, e0 = fix(e1), fix(e0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return auc(e1, 5.0) - auc(e0, 5.0), np.percentile(diff, [2.5, 97.5])


o = sweep(MAIN / "E19" / "sweep" / "gauss8")
ref = o[max(o, key=lambda k: o[k]["summary"]["auc@5"])]
curves, logs = {}, {}
for key, (X, m, lab) in RUNS.items():
    rs = rows(X / "ckpt" / m / "log.jsonl")
    s = sweep(X / "sweep" / m)
    if not rs:
        print(f"== {key}: 没有训练日志")
        continue
    logs[key] = rs
    print(f"== {key}（{lab}）  跑到第 {rs[-1]['step']} 步  发散起点 {onset(rs)}  "
          f"参照拟合失败步数 {sum(1 for r in rs if 'd_ref' not in r)}  "
          f"d_ref 50 步中位数最高 {np.nanmax(roll(rs)):.1f} px（第 {rs[int(np.nanargmax(roll(rs)))]['step']} 步）  "
          f"每步耗时中位数 {np.median([r['sec'] for r in rs if 'sec' in r]):.3f} s  显存 {rs[0].get('mem_gb')}")
    if s:
        curves[key] = {k: v["summary"]["auc@5"] for k, v in s.items()}
        st = max(s, key=lambda k: s[k]["summary"]["auc@5"])
        d, ci = paired(s[st]["errors"], ref["errors"])
        print("  val auc@5 :", {k: round(v, 3) for k, v in curves[key].items()})
        print("  val 误差中位数:", {k: round(v["summary"]["median"], 2) for k, v in s.items()})
        print("  val fail_rate:", {k: round(v["summary"]["fail_rate"], 3) for k, v in s.items()})
        print(f"  最好存档 step {st}: auc@5 {s[st]['summary']['auc@5']:.3f}  − E19 gauss8: {d:+.4f} ci {np.round(ci, 4)}")
    by = {r["step"]: r for r in rs}
    for a in AT:
        w = [by[t] for t in range(a, a + 100) if t in by]
        v = {k: np.median([r[k] for r in w if k in r]) for k in KEYS if any(k in r for r in w)}
        if v:
            print(f"  step {a:4d}: " + " ".join(f"{k}={x:.3g}" for k, x in v.items()))
    if key in FINE:   # 第 500–1000 步逐 50 步看 50 步滑动中位数
        r = {k: roll(rs, k) for k in ["d_ref", "rew", "H", "pmode", "gnorm", "upd_lora"]}
        for t in range(500, 1001, 50):
            i = [j for j, x in enumerate(rs) if x["step"] == t][0]
            print(f"  滑动中位数 step {t:4d}: " + " ".join(f"{k}={v[i]:.3g}" for k, v in r.items()))

COL = {"c16": "tab:blue", "c8_": "tab:red", "f00": "tab:green"}
LS = {"s0": "-", "s1": "--", "s2": ":"}
sty = lambda k: dict(color=COL[k[:3]], ls=LS[k[-2:]], lw=1.2)

# 图 1：离参照距离的 50 步滑动中位数
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for k, rs in logs.items():
    ax.plot([r["step"] for r in rs], roll(rs), label=RUNS[k][2], **sty(k))
ax.axhline(100, color="k", ls=":", lw=0.8)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(PLAIN)
ax.set_xlabel("训练步数")
ax.set_ylabel("离参照距离，50 步滑动中位数（px）")
ax.legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(E / "extra" / "d_ref.png", dpi=150)

# 图 2：逐存档的 Val AUC@5
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for k, c in curves.items():
    ax.plot(list(c), list(c.values()), marker="o", ms=3, label=RUNS[k][2], **sty(k))
ax.set_xlabel("训练步数")
ax.set_ylabel("Val AUC@5")
ax.legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(E / "extra" / "val_auc.png", dpi=150)

# 图 3：粗匹配分数与粗匹配组每步相对改变量（50 步滑动中位数），只画本实验的只训粗匹配运行与 E19
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
for k, rs in logs.items():
    if k.startswith("f"):
        continue
    t = [r["step"] for r in rs]
    for ax, key in zip(axs, ["rew", "upd_coarse", "upd_lora"]):
        if any(key in r for r in rs):
            ax.plot(t, roll(rs, key), label=RUNS[k][2], **sty(k))
for ax, lab in zip(axs, ["粗匹配分数", "粗匹配解码器每步相对改变量", "LoRA 每步相对改变量"]):
    ax.set_xlabel("训练步数")
    ax.set_title(lab, fontsize=9)
for ax in axs[1:]:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PLAIN)
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(E / "extra" / "curves.png", dpi=150)
