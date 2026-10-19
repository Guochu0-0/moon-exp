"""E10 的统计与画图：读 reward_validity.py 的逐版本表，算每个 reward 在各版本族内与真实误差的对内一致率。

    python runs/E10/code/summarize.py [--plot-only]     # --plot-only：只按 summary.json 重画图

对内一致率（C-index，Harrell et al., JAMA 1982 的一致性指数用在对内排序上）：同一 patch 对的同一族版本两两比较，
只比真实误差相差 ≥ DELTA 的两份；reward 高的那份误差也小记 1，reward 相等记 0.5，反之记 0。
先在对内平均，再对有可比版本的对平均；0.5 = 随机，1 = 完全一致。区间为按 patch 对自助抽样 1000 次的 95% 区间。
另报 reward 相等的比较所占比例（「看不见」）。

族：base + 一种合成误差的 4 档（shift / rot / scale / noise / outlier）；mixed 为 base + 全部 20 个合成版本；
s1_1 / s1_2 为 8 份策略样本；real_roma 为 3 个 RoMa 模型的实际输出，real_all 为 8 个模型。
分层：按 zero-shot RoMa 在这一对上的误差（base 版本）分 < 3、3–10、≥ 10 px。

产物：runs/E10/extra/summary.json、cindex.png、strata.png。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
DATA = RUN / "extra" / "data"
DELTA = 0.1
B = 1000
REWARDS = ["r1_inl", "r1_grad", "r2_cfog", "r2_ncc", "r3_shift64", "r3_rot10", "r3_scale10", "r3_mean",
           "r4_boot", "rand"]
LABEL = {"r1_inl": "R1 内点 +1 / 外点 −0.1", "r1_grad": "R1 分级残差", "r2_cfog": "R2 CFOG-NCC",
         "r2_ncc": "R2 灰度 NCC", "r3_shift64": "R3 平移 64 px", "r3_rot10": "R3 旋转 10°",
         "r3_scale10": "R3 缩放 10%", "r3_mean": "R3 三者平均", "r4_boot": "R4 子集重拟合", "rand": "随机（对照）"}
SYN = ["shift", "rot", "scale", "noise", "outlier"]
ROMA = ["anymatch_roma", "erase_geo__r2", "roma_vgg"]
FAMILIES = [*SYN, "mixed", "s1_1", "s1_2", "real_roma", "real_all"]
FLABEL = {"shift": "整体平移", "rot": "旋转", "scale": "缩放", "noise": "逐点噪声", "outlier": "离群点",
          "mixed": "合成混合", "s1_1": "策略样本 σ1", "s1_2": "策略样本 σ2", "real_roma": "真实·RoMa",
          "real_all": "真实·全部"}
STRATA = [(0, 3), (3, 10), (10, np.inf)]


def family_sets(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = d[d.family == "base"]
    out = {f: pd.concat([base, d[d.family == f]]) for f in SYN}
    out["mixed"] = d[d.family.isin(["base", *SYN])]
    out["s1_1"] = d[d.family == "s1_1"]
    out["s1_2"] = d[d.family == "s1_2"]
    real = d[d.family == "real"]
    out["real_roma"] = real[real.version.isin(ROMA)]
    out["real_all"] = real
    return out


def per_pair(df: pd.DataFrame, reward: str) -> pd.DataFrame:
    """每个 patch 对：一致得分之和、可比较数、reward 相等数。"""
    rows = []
    for pair, g in df.groupby("pair", sort=False):
        g = g[np.isfinite(g.err) & np.isfinite(g[reward])]
        e, r = g.err.to_numpy(), g[reward].to_numpy()
        i, j = np.triu_indices(len(e), 1)
        de, dr = e[j] - e[i], r[j] - r[i]
        ok = np.abs(de) >= DELTA
        if not ok.any():
            continue
        de, dr = de[ok], dr[ok]
        tie = np.abs(dr) <= 1e-9 * np.maximum(1, np.abs(r).max())
        score = np.where(tie, 0.5, (np.sign(dr) == -np.sign(de)).astype(float))
        rows.append((pair, score.mean(), tie.mean(), ok.sum()))
    return pd.DataFrame(rows, columns=["pair", "c", "tie", "n"]).set_index("pair")


def boot(x: np.ndarray, rng) -> tuple[float, float, float]:
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    d = pd.read_csv(DATA / "versions.csv.gz")
    base_err = d[d.family == "base"].set_index("pair").err
    sets = family_sets(d)
    rng = np.random.default_rng(0)
    res = {"delta": DELTA, "n_pairs": int(d.pair.nunique()),
           "base_err_median": float(base_err[np.isfinite(base_err)].median()), "families": {}, "strata": {}}
    table = {}
    for f in FAMILIES:
        res["families"][f] = {}
        for r in REWARDS:
            pp = per_pair(sets[f], r)
            m, lo, hi = boot(pp.c.to_numpy(), rng)
            res["families"][f][r] = {"c": m, "lo": lo, "hi": hi, "tie": float(pp.tie.mean()),
                                     "n_pairs": int(len(pp)), "n_cmp": int(pp.n.sum())}
            table[(f, r)] = (m, lo, hi)
            if f in ("mixed", "real_all", "s1_1"):
                for a, b in STRATA:
                    sel = pp[pp.index.isin(base_err[(base_err >= a) & (base_err < b)].index)]
                    if len(sel) < 5:
                        continue
                    sm, slo, shi = boot(sel.c.to_numpy(), rng)
                    res["strata"].setdefault(f, {}).setdefault(r, {})[f"{a}-{b}"] = \
                        {"c": sm, "lo": slo, "hi": shi, "n_pairs": int(len(sel))}
    res["strata_n"] = {f"{a}-{b}": int(((base_err >= a) & (base_err < b)).sum()) for a, b in STRATA}
    # 真实误差在族内的跨度：判断「可比较」的版本差有多大
    res["err_spread"] = {f: float(sets[f].groupby("pair").err.agg(lambda e: np.ptp(e[np.isfinite(e)]) if
                                                                    np.isfinite(e).sum() > 1 else np.nan).median())
                         for f in FAMILIES}
    p = pd.read_csv(DATA / "points.csv")
    res["points"] = {k: float(p[k].mean()) for k in
                     ["spearman", "inl_frac", "true_ok_in_inl", "true_ok_in_out", "true_ok_all", "gt_fit_err"]}
    res["points"]["spearman_median"] = float(p.spearman.median())
    res["points"]["gt_fit_err_median"] = float(p.gt_fit_err.median())
    (RUN / "extra" / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    plot(table, res)
    for f in FAMILIES:
        print(f"{FLABEL[f]:8s}", " ".join(f"{res['families'][f][r]['c']:.2f}" for r in REWARDS))


def plot(table, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    M = np.array([[table[(f, r)][0] for f in FAMILIES] for r in REWARDS])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1)
    for i, r in enumerate(REWARDS):
        for j, f in enumerate(FAMILIES):
            m, lo, hi = table[(f, r)]
            sig = lo > 0.5 or hi < 0.5
            ax.text(j, i, f"{m:.2f}" + ("" if sig else "°"), ha="center", va="center", fontsize=9,
                    color="white" if abs(m - 0.5) > 0.3 else "black")
    ax.set_xticks(range(len(FAMILIES)), [FLABEL[f] for f in FAMILIES], rotation=30, ha="right")
    ax.set_yticks(range(len(REWARDS)), [LABEL[r] for r in REWARDS])
    fig.colorbar(im, ax=ax, label="对内一致率（0.5 = 随机）")
    ax.set_title("° 表示 95% 区间含 0.5")
    fig.tight_layout()
    fig.savefig(RUN / "extra" / "cindex.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, f in zip(axes, ("mixed", "s1_1", "real_all")):
        for k, (a, b) in enumerate(STRATA):
            key = f"{a}-{b}"
            ys = [res["strata"].get(f, {}).get(r, {}).get(key) for r in REWARDS]
            x = np.arange(len(REWARDS)) + (k - 1) * 0.25
            ok = [y is not None for y in ys]
            ax.errorbar(x[ok], [y["c"] for y in ys if y], fmt="o", ms=4, capsize=2,
                        yerr=np.array([[y["c"] - y["lo"], y["hi"] - y["c"]] for y in ys if y]).T,
                        label=f"zero-shot 误差 {a}–{b if np.isfinite(b) else '∞'} px（{res['strata_n'][key]} 对）")
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.set_xticks(range(len(REWARDS)), [LABEL[r] for r in REWARDS], rotation=60, ha="right", fontsize=8)
        ax.set_title(FLABEL[f])
    axes[0].set_ylabel("对内一致率")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(RUN / "extra" / "strata.png", dpi=130)
    plt.close(fig)


def plot_only():
    """只按已有的 summary.json 重画图（服务器上没有中文字体时，在有字体的机器上重画）。"""
    res = json.loads((RUN / "extra" / "summary.json").read_text(encoding="utf-8"))
    table = {(f, r): (v["c"], v["lo"], v["hi"]) for f, rs in res["families"].items() for r, v in rs.items()}
    plot(table, res)


if __name__ == "__main__":
    plot_only() if "--plot-only" in sys.argv else main()
