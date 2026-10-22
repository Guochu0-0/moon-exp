"""合并 probe.py 的分片，写汇总与图（#195）。

    python runs/E21/code/report.py

读 raw/shard*.npz、shard*_pairs.csv；写 extra/data/summary.json、pairs.csv、tables.md，extra/geometry.png、
extra/right_by_bs.png，以及逐像素子样本 extra/data/pixels.npz。
比例：对 / 错 / 无信号各占像素的比例；净比例 = (Σ 对的强度 − Σ 错的强度) / Σ 强度。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path(__file__).resolve().parents[1]
DATA = RUN / "extra" / "data"
RAW = RUN / "raw"                     # probe.py 的分片（不进 git）
FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 同 E14
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"      # 刻度里的负号（SimHei 缺 U+2212）

LAYERS = ("a", "b", "c")
GROUPS = ("all", "success", "fail")
GEO = ("b", "mt", "mr", "s")          # |R − T|、|M − T|、|M − R|、散开程度 s


def stats(x):
    """x (..., 6) 计数行 → 比例与净比例。"""
    n, r, w, z, sr, sw = (float(v) for v in x)
    tot = sr + sw
    return {"n": int(n), "right": round(r / n, 4) if n else None, "wrong": round(w / n, 4) if n else None,
            "zero": round(z / n, 4) if n else None, "net": round((sr - sw) / tot, 4) if tot > 0 else None}


def pct(x):
    if not len(x):
        return None
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {f"p{p}": round(float(v), 3) for p, v in zip((10, 25, 50, 75, 90), q)} | {"n": int(len(x))}


def label(c):
    if c == "coarse":
        return "粗级"
    s, k = c[1:].split("_k")
    return f"尺度 {s}，σ = {k}·s"


def main():
    shards = sorted(RAW.glob("shard*.npz"))
    zs = [np.load(f) for f in shards]
    configs = [str(c) for c in zs[0]["configs"]]
    bins = zs[0]["bins"]
    acc = {c: sum(z[f"acc_{c}"] for z in zs) for c in configs}
    keep = {c: np.concatenate([z[f"keep_{c}"] for z in zs]) for c in configs}
    n_pairs = int(sum(int(z["n_pairs"]) for z in zs))
    n_ref_fail = int(sum(int(z["n_ref_fail"]) for z in zs))
    rows = [r for f in sorted(RAW.glob("shard*_pairs.csv")) for r in csv.DictReader(open(f, encoding="utf-8"))]
    with open(DATA / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    np.savez_compressed(DATA / "pixels.npz", columns=np.array(["pair", "fail", *GEO, "dE_a", "dE_b", "dE_c"]),
                        **{c: keep[c] for c in configs})

    succ = {r["pair"] for r in rows if r["success"] == "True"}
    summ = {"n_pairs": n_pairs, "n_ref_fail": n_ref_fail, "n_success": len(succ),
            "n_fail": len({r["pair"] for r in rows}) - len(succ), "bins_b_over_s": [float(b) for b in bins],
            "by_config": {}}
    for c in configs:
        A = acc[c]                                       # (3, 2, nbin, 6)
        k = keep[c]
        d = {"geometry": {g: {"all": pct(k[:, 2 + i]), "success": pct(k[k[:, 1] == 0, 2 + i]),
                              "fail": pct(k[k[:, 1] == 1, 2 + i])} for i, g in enumerate(GEO)},
             "b_over_s": {"all": pct(k[:, 2] / k[:, 5])}, "layers": {}}
        for li, ln in enumerate(LAYERS):
            per = {"all": A[li].sum((0, 1)), "success": A[li, 0].sum(0), "fail": A[li, 1].sum(0)}
            d["layers"][ln] = {g: stats(v) for g, v in per.items()}
            d["layers"][ln]["by_bin"] = {g: [stats(v) for v in (A[li].sum(0) if g == "all" else A[li, gi - 1])]
                                         for gi, g in enumerate(GROUPS)}
        summ["by_config"][c] = d
    (DATA / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")

    # 表：每个配置一行，三层 × (对、错、无信号、净比例)；另按成功 / 失败
    f = lambda v: "—" if v is None else f"{v:.2f}"
    lines = []
    for g in GROUPS:
        lines += [f"\n### {g}\n", "| 配置 | " + " | ".join(f"{ln} 对 | {ln} 错 | {ln} 无 | {ln} 净" for ln in LAYERS) + " |",
                  "|---|" + "---|" * 12]
        for c in configs:
            L = summ["by_config"][c]["layers"]
            lines.append(f"| {label(c)} | " + " | ".join(
                f"{f(L[ln][g]['right'])} | {f(L[ln][g]['wrong'])} | {f(L[ln][g]['zero'])} | {f(L[ln][g]['net'])}"
                for ln in LAYERS) + " |")
    lines += ["\n### 几何量中位数（px）与 b/s 中位数\n", "| 配置 | b | \\|M−T\\| | \\|M−R\\| | s | b/s |", "|---|---|---|---|---|---|"]
    for c in configs:
        G = summ["by_config"][c]["geometry"]
        lines.append(f"| {label(c)} | " + " | ".join(f"{G[g]['all']['p50']:.2f}" for g in GEO)
                     + f" | {summ['by_config'][c]['b_over_s']['all']['p50']:.2f} |")
    lines += ["\n### 按 b/s 分档（全部图对）：b 层 对的比例 / 净比例 / 像素占比\n",
              "| 配置 | " + " | ".join(f"[{bins[i]:g}, {bins[i + 1]:g})" for i in range(len(bins) - 1)) + " |",
              "|---|" + "---|" * (len(bins) - 1)]
    for c in configs:
        bb = summ["by_config"][c]["layers"]["b"]["by_bin"]["all"]
        tot = sum(x["n"] for x in bb)
        lines.append(f"| {label(c)} | " + " | ".join(
            "—" if not x["n"] else f"{f(x['right'])} / {f(x['net'])} / {x['n'] / tot:.2f}" for x in bb) + " |")
    for g in ("success", "fail"):
        lines += [f"\n### 按 b/s 分档（{g}）：b 层 对的比例 / 净比例 / 像素占比\n",
                  "| 配置 | " + " | ".join(f"[{bins[i]:g}, {bins[i + 1]:g})" for i in range(len(bins) - 1)) + " |",
                  "|---|" + "---|" * (len(bins) - 1)]
        for c in configs:
            bb = summ["by_config"][c]["layers"]["b"]["by_bin"][g]
            tot = sum(x["n"] for x in bb)
            lines.append(f"| {label(c)} | " + " | ".join(
                "—" if not x["n"] else f"{f(x['right'])} / {f(x['net'])} / {x['n'] / tot:.2f}" for x in bb) + " |")
    (DATA / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # 图 1：几何量分布（成功 / 失败），每级一行（细化取 k = 0.5 的 s；b、|M−T|、|M−R| 与 k 无关）
    lv = ["coarse"] + [c for c in configs if c.endswith("_k0.5")]
    fig, axes = plt.subplots(len(lv), 4, figsize=(13, 2.1 * len(lv)), constrained_layout=True)
    names = ("b = |R - T|", "|M - T|", "|M - R|", "散开程度 s")
    edges = np.logspace(-2, 3.5, 56)
    for i, c in enumerate(lv):
        k = keep[c]
        for j in range(4):
            ax = axes[i, j]
            if j == 3 and c != "coarse":       # 细化的 s 就是 σ，是常数
                ax.axis("off")
                continue
            for gi, (gl, col) in enumerate((("配准成功", "C0"), ("配准失败", "C3"))):
                x = np.clip(k[k[:, 1] == gi, 2 + j], edges[0], edges[-1])
                ax.hist(x, bins=edges, density=True, histtype="step", color=col, label=gl)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(names[j] + "（px）")
            if j == 0:
                ax.set_ylabel(label(c).split("，")[0])
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.savefig(RUN / "extra" / "geometry.png", dpi=120)

    # 图 2：对的比例随 b/s 分档（全部图对），三层不同线型；细化取 σ = 0.5·s
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2), sharey=True, constrained_layout=True)
    mid = np.sqrt(np.r_[bins[1] / 2, bins[1:-1]] * np.r_[bins[1:-1], bins[-2] * 2])
    for ax, c in zip(axes, lv):
        for ln, ls, lab in zip(LAYERS, ("--", "-", ":"), ("a 几何模型", "b 精确期望", "c 16 次实测")):
            bb = summ["by_config"][c]["layers"][ln]["by_bin"]["all"]
            y = [np.nan if x["n"] < 50 else x["right"] for x in bb]
            ax.plot(mid, y, ls=ls, c="k", marker=".", label=lab)
        ax.set_xscale("log")
        ax.axhline(0.5, c="gray", lw=0.6)
        ax.set_title(label(c))
        ax.set_xlabel("b / s")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("信号对的像素比例")
    fig.savefig(RUN / "extra" / "right_by_bs.png", dpi=120)


if __name__ == "__main__":
    main()
