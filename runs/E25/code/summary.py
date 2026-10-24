"""E25 汇总：各方法逐存档的 Val AUC@5、最好存档与 E19 gauss8 的配对自助差；发散起点（离参照距离 d_ref 的 50 步
滑动中位数首次超过 100 px；冻结参照的运行另按离当前模型仿射的距离 d_cur 算一次）；训练日志的窗口中位数；画两张图。
E22 f1（参照随当前模型，种子 0）作对照，日志从 gpfs 主 checkout 读。
用法（在仓库根目录、wb 环境）：PYTHONPATH=. python runs/E25/code/summary.py
"""
import json
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
MAIN = Path("/remote-home/xufang/YGC/moon-exp/runs")   # E19、E22 的产物（结票后在主 checkout）
LABEL = {"fr1_s0": "冻结参照，种子 0", "fr1_s1": "冻结参照，种子 1", "f1_s1": "参照随模型，种子 1",
         "E22/f1": "参照随模型，种子 0（E22）"}
AT = [1, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 3900]     # 从该步起 100 步的窗口
KEYS = ["d_ref", "d_cur", "ref_gap", "rew", "rew_16", "rew_8", "rew_4", "d_16", "d_8", "d_4",
        "upd_coarse", "upd_lora", "upd_ref16", "upd_ref8", "upd_ref4", "upd_vgg", "grad_norm"]


def sweep(d):
    m = json.loads((d / "S" / "metrics.json").read_text())["methods"]
    return {int(k[4:]): v["val"] for k, v in sorted(m.items(), key=lambda kv: int(kv[0][4:])) if "val" in v}


def rows(p):
    r = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [x for x in r if "step" in x]


def roll(rs, key):
    x = np.array([r.get(key, np.nan) for r in rs], float)
    return np.array([np.nanmedian(x[max(0, i - 49):i + 1]) if np.isfinite(x[max(0, i - 49):i + 1]).any() else np.nan
                     for i in range(len(x))])


def onset(rs, key="d_ref"):
    """key 的 50 步滑动中位数首次超过 100 px 的步；没有该量的步（拟合失败）记为缺失。"""
    m = roll(rs, key)
    hit = np.nonzero(m > 100)[0]
    return rs[hit[0]]["step"] if len(hit) else None


def paired(e1, e0, n=2000, seed=0):
    fix = lambda e: np.array([np.inf if x is None else x for x in e], float)
    e1, e0 = fix(e1), fix(e0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e1), (n, len(e1)))
    diff = [auc(e1[i], 5.0) - auc(e0[i], 5.0) for i in idx]
    return auc(e1, 5.0) - auc(e0, 5.0), np.percentile(diff, [2.5, 97.5])


o = sweep(MAIN / "E19" / "sweep" / "gauss8")
ref = o[max(o, key=lambda k: o[k]["summary"]["auc@5"])]
print("== E19 gauss8 val auc@5:", {k: round(v["summary"]["auc@5"], 3) for k, v in o.items()})
src = {m: (E / "sweep" / m, E / "ckpt" / m / "log.jsonl") for m in ("fr1_s0", "fr1_s1", "f1_s1")}
e22 = MAIN / "E22" / "ckpt" / "f1" / "log.jsonl"
if not e22.exists():   # A6000 的主 checkout 没有 E22 的训练日志，从 gpfs 拷到这里
    e22 = E / "ckpt" / "_ref" / "E22_f1_log.jsonl"
src["E22/f1"] = (MAIN / "E22" / "sweep" / "f1", e22)
curves, logs = {}, {}
for m, (sw, lg) in src.items():
    if lg.exists():
        logs[m] = rows(lg)
        rs = logs[m]
        print(f"== {m}  跑到第 {rs[-1]['step']} 步  发散起点（d_ref）{onset(rs)}  （d_cur）{onset(rs, 'd_cur')}"
              f"  参照拟合失败步数 {sum(1 for r in rs if 'd_ref' not in r)}"
              f"  第 1 步 mem_gb {rs[0].get('mem_gb')}  每步秒数中位数 {np.median([r['sec'] for r in rs]):.2f}")
        print(f"  第 1 步 d_ref={rs[0].get('d_ref')} d_cur={rs[0].get('d_cur')} ref_gap={rs[0].get('ref_gap')}")
        by = {r["step"]: r for r in rs}
        for a in AT:
            w = [by[t] for t in range(a, a + 100) if t in by]
            v = {k: np.median([r[k] for r in w if k in r]) for k in KEYS if any(k in r for r in w)}
            print(f"  step {a:4d}: " + " ".join(f"{k}={x:.3g}" for k, x in v.items()))
    if (sw / "S" / "metrics.json").exists():
        s = sweep(sw)
        curves[m] = {k: v["summary"]["auc@5"] for k, v in s.items()}
        st = max(s, key=lambda k: s[k]["summary"]["auc@5"])
        d, ci = paired(s[st]["errors"], ref["errors"])
        print("  val auc@5 :", {k: round(v, 3) for k, v in curves[m].items()})
        print("  val 误差中位数:", {k: round(float(np.median([np.inf if x is None else x for x in v["errors"]])), 2)
                                  for k, v in s.items()})
        print(f"  最好存档 step {st}: auc@5 {s[st]['summary']['auc@5']:.3f}  − E19 gauss8: {d:+.4f} ci {np.round(ci, 4)}")
    else:
        print(f"  {m}: 没有评测结果")

(E / "extra").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 3.6))
for m in curves:
    ax.plot(list(curves[m]), list(curves[m].values()), "o-", ms=3, label=LABEL[m])
ax.plot(list(o), [v["summary"]["auc@5"] for v in o.values()], "k--", lw=1, label="只训粗匹配（E19）")
ax.set_xlabel("训练步数")
ax.set_ylabel("Val AUC@5")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(E / "extra" / "val_auc.png", dpi=150)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for m, rs in logs.items():
    st = [r["step"] for r in rs]
    axs[0].plot(st, roll(rs, "d_ref"), lw=1, label=LABEL[m])
    axs[1].plot(st, roll(rs, "d_cur" if m.startswith("fr") else "d_ref"), lw=1, label=LABEL[m])
for ax, t in zip(axs, ("离打分所用参照的距离", "离当前模型正常推理的仿射的距离")):
    ax.axhline(100, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("训练步数")
    ax.set_title(t, fontsize=9)
axs[0].set_ylabel("粗匹配距离，50 步滑动中位数（px）")
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(E / "extra" / "d_ref.png", dpi=150)
