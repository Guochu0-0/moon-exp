"""E9 汇总（#150）：各变体相对同模型参照 <模型>__s1 的逐对配对比较，写 extra/data/summary.csv 与 extra/delta_auc5.png。

    PYTHONPATH=. /opt/envs/loftr/bin/python runs/E9/code/summary.py   # 在 worktree 根目录跑（wb 环境没有 matplotlib）；先 workbench eval E9

配对比较：同一批 Val pair 上，ΔAUC@5 = 变体 − 参照，95% CI 按 pair 重采样 2000 次（两方法用同一组下标）。
「变好 / 变差」= pair 误差减小 / 增大超过 1 px 的对数。x / y 残差 = 检查点经 A 映射后与 SAR 检查点之差，先在 pair 内取均值，
再报 |·| 的中位数（失败 pair 不计）。
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

from workbench.dataset import Dataset
from workbench.protocol import auc, checkpoint_residuals

R = Path("runs/E9")
REF = {"anymatch_loftr": "anymatch_loftr__s1", "anymatch_roma": "anymatch_roma__s1"}
ds = Dataset(os.environ.get("MOON_DATA", "/remote-home/xufang/YGC/dataset/Moon"))
met = json.loads((R / "metrics.json").read_text(encoding="utf-8"))["methods"]
pairs = ds.labelled("val")


def errors(m):
    """pair 误差，按 ds.labelled("val") 的顺序（workbench/evaluate.py），失败为 ∞。"""
    return np.array([np.inf if v is None else v for v in met[m]["val"]["errors"]], np.float64)


def residuals(m):
    A = {}
    for l in (R / "preds" / m / "val.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            A[d["pair"]] = d.get("A")
    out = []
    for p in pairs:
        r = checkpoint_residuals(A.get(p), *ds.checkpoints("val", p))
        out.append(r.mean(0) if r is not None else (np.nan, np.nan))
    return np.array(out)


def n_inliers(m):
    n = [json.loads(l).get("n_inliers", 0) for l in (R / "preds" / m / "val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return float(np.median(n))


rng = np.random.default_rng(0)
idx = rng.integers(0, len(pairs), size=(2000, len(pairs)))
rows = []
for m in sorted(met):
    if "val" not in met[m]:
        continue
    model = m.split("__")[0]
    e, er = errors(m), errors(REF[model])
    s = met[m]["val"]["summary"]
    res = residuals(m)
    d5 = np.clip(1 - e[idx] / 5, 0, None).mean(1) - np.clip(1 - er[idx] / 5, 0, None).mean(1)
    de = e - er
    rows.append({
        "method": m, "model": model, "variant": m.split("__", 1)[1],
        "auc@3": s["auc@3"], "auc@5": s["auc@5"], "auc@10": s["auc@10"],
        "sr@3": s["sr@3"], "sr@5": s["sr@5"], "sr@10": s["sr@10"], "fail_rate": s["fail_rate"], "median": s["median"],
        "d_auc@5": auc(e, 5) - auc(er, 5), "d_auc@5_lo": float(np.percentile(d5, 2.5)), "d_auc@5_hi": float(np.percentile(d5, 97.5)),
        "better": int((de < -1).sum()), "worse": int((de > 1).sum()),
        "med_abs_rx": float(np.nanmedian(np.abs(res[:, 0]))), "med_abs_ry": float(np.nanmedian(np.abs(res[:, 1]))),
        "med_inliers": n_inliers(m),
    })

out = R / "extra" / "data"
out.mkdir(parents=True, exist_ok=True)
with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("| 方法 | AUC@3 | AUC@5 | AUC@10 | SR@3 | SR@10 | 失败率 | ΔAUC@5 [95% CI] | 变好/变差 | 中位 |rx| | 中位 |ry| | 内点中位 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['method']} | {r['auc@3']:.3f} | {r['auc@5']:.3f} | {r['auc@10']:.3f} | {r['sr@3']:.3f} | {r['sr@10']:.3f} | "
          f"{r['fail_rate']:.3f} | {r['d_auc@5']:+.3f} [{r['d_auc@5_lo']:+.3f}, {r['d_auc@5_hi']:+.3f}] | "
          f"{r['better']}/{r['worse']} | {r['med_abs_rx']:.2f} | {r['med_abs_ry']:.2f} | {r['med_inliers']:.0f} |")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
for ax, model, title in zip(axes, REF, ("AnyMatch-LoFTR", "AnyMatch-RoMa")):
    rs = sorted((r for r in rows if r["model"] == model and r["method"] != REF[model]), key=lambda r: r["d_auc@5"])
    y = np.arange(len(rs))
    ax.errorbar([r["d_auc@5"] for r in rs], y, xerr=[[r["d_auc@5"] - r["d_auc@5_lo"] for r in rs], [r["d_auc@5_hi"] - r["d_auc@5"] for r in rs]],
                fmt="o", capsize=3)
    ax.set_yticks(y, [r["variant"] for r in rs])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{title} (S1 reference AUC@5 = {next(r['auc@5'] for r in rows if r['method'] == REF[model]):.3f})")
    ax.set_xlabel("ΔAUC@5 (variant − S1), 95% paired bootstrap CI")   # 服务器没有中文字体
fig.tight_layout()
fig.savefig(R / "extra" / "delta_auc5.png", dpi=120)
print("wrote", out / "summary.csv", R / "extra" / "delta_auc5.png")
