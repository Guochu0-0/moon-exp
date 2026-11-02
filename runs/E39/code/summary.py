"""E39（#240）：把搜索结果和反向估计汇总成表、写 extra/data/summary.json。CPU，在服务器 worktree 根目录运行。

    MOON_DATA=... python runs/E39/code/summary.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from workbench import protocol
from workbench.dataset import Dataset

R = Path("runs/E39")
D = R / "extra" / "data"
MODELS = ("zero_shot", "e37")
ROWS = ("model", "trans", "full", "identity", "gt_linear")
NAMES = {"model": "模型的答案", "trans": "只调平移", "full": "6 个数一起调", "identity": "线性部分设为无变形",
         "gt_linear": "标注拟合的线性部分"}
LIM = {"平移": (("tx", "ty"), 8.0, 1 / 16), "旋转": (("rot",), 2.0, np.degrees(1 / 362) / 16),
       "缩放": (("sx", "sy"), 0.05, 1 / 256 / 16), "剪切": (("shear",), 2.0, np.degrees(1 / 256) / 16)}
BINS = [(0, 3, "< 3"), (3, 5, "3–5"), (5, 10, "5–10"), (10, np.inf, "≥ 10")]
CH_BINS = [(0, 1, "< 1"), (1, 2, "1–2"), (2, 4, "2–4"), (4, 8, "4–8"), (8, np.inf, "≥ 8")]
CORNERS = np.array([[0, 0], [512, 0], [0, 512], [512, 512]], float)
NBOOT = 2000


def rank(x):
    """平均秩。"""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=float)
    u, inv, c = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(u))
    np.add.at(s, inv, r)
    return (s / c)[inv]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1]), int(ok.sum())


def corner_dist(A, B):
    if A is None or B is None:
        return np.inf
    A, B = np.asarray(A, float), np.asarray(B, float)
    return float(np.linalg.norm(CORNERS @ A[:, :2].T + A[:, 2] - (CORNERS @ B[:, :2].T + B[:, 2]), axis=1).mean())


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {k: float(np.percentile(x, p)) for k, p in (("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))} | {"mean": float(x.mean())}


def load(path):
    return {r["pair"]: r for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())}


def errors(recs, pairs, ds, row):
    return np.array([protocol.pair_error(recs[p].get(row, {}).get("A"), *ds.checkpoints("val", p)) for p in pairs])


def row_report(e0, e1, recs, pairs, row, rng):
    idx = rng.integers(0, len(e0), size=(NBOOT, len(e0)))
    a5 = lambda e: np.clip(1 - e / 5, 0, None)
    d = a5(e1) - a5(e0)
    boots = d[idx].mean(1)
    fin = np.isfinite(e0) & np.isfinite(e1)
    out = {"summary": {k: float(v) for k, v in protocol._point(e1).items() if k in ("auc@3", "auc@5", "sr@3", "sr@5", "median", "fail_rate")},
           "d_auc5": float(d.mean()), "d_auc5_ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
           "better": int((e1[fin] < e0[fin]).sum()), "worse": int((e1[fin] > e0[fin]).sum()),
           "bins": {}}
    for lo, hi, name in BINS:
        m = (e0 >= lo) & (e0 < hi) & fin
        out["bins"][name] = {"n": int(m.sum()), "better": float((e1[m] < e0[m]).mean()) if m.any() else None,
                             "median_change": float(np.median(e1[m] - e0[m])) if m.any() else None}
    As = [(recs[p]["model"]["A"], recs[p][row]["A"]) for p in pairs if row in recs[p]]
    ch = np.array([corner_dist(a, b) for a, b in As])
    imp = (e0 - e1)[[row in recs[p] for p in pairs]]
    out["change"] = q(ch)
    out["change_vs_gain"] = {"spearman": spearman(ch, imp)[0], "bins": {}}
    for lo, hi, name in CH_BINS:
        m = (ch >= lo) & (ch < hi)
        out["change_vs_gain"]["bins"][name] = {"n": int(m.sum()), "better": float((imp[m] > 0).mean()) if m.any() else None,
                                                "median_gain": float(np.median(imp[m])) if m.any() else None}
    if row != "gt_linear":
        edge = {}
        ps = [recs[p][row]["p"] for p in pairs if row in recs[p]]
        for name, (keys, lim, tol) in LIM.items():
            if keys[0] not in ps[0]:
                continue
            edge[name] = float(np.mean([any(abs(x.get(k, 0.0)) >= lim - tol for k in keys) for x in ps]))
        out["edge"] = edge
        n0 = np.array([recs[p]["model"]["ncc"] for p in pairs if row in recs[p]])
        n1 = np.array([recs[p][row]["ncc"] for p in pairs if row in recs[p]])
        out["ncc"] = {"model_median": float(np.median(n0)), "row_median": float(np.median(n1)),
                      "row_ge_model": float((n1 >= n0 - 1e-6).mean())}
    return out


def conv_report(ds):
    """步长核对：粗 / 用的 / 细 三种设置在每 16 对取一对的子集上互相比。"""
    out = {}
    for m in MODELS:
        S = {k: load(D / "conv" / f"{m}_{k}.jsonl") for k in ("coarse", "used", "fine") if (D / "conv" / f"{m}_{k}.jsonl").exists()}
        if len(S) < 3:
            continue
        pairs = sorted(S["used"])
        out[m] = {"n": len(pairs)}
        for row in ("trans", "full", "identity"):
            e = {k: errors(S[k], pairs, ds, row) for k in S}
            auc = {k: protocol.auc(e[k], 5.0) for k in S}
            d = lambda a, b: [corner_dist(S[a][p][row]["A"], S[b][p][row]["A"]) for p in pairs]
            dn = lambda a, b: [S[a][p][row]["ncc"] - S[b][p][row]["ncc"] for p in pairs]
            out[m][row] = {"auc5": auc,
                           "corner_used_vs_fine": q(d("used", "fine")), "corner_coarse_vs_fine": q(d("coarse", "fine")),
                           "frac_used_within_1px_of_fine": float(np.mean(np.array(d("used", "fine")) <= 1.0)),
                           "ncc_fine_minus_used": q(dn("fine", "used")), "ncc_fine_minus_coarse": q(dn("fine", "coarse"))}
    return out


def swap_report(recs, pairs, ds, m):
    f = D / f"swap_{m}.jsonl"
    if not f.exists():
        return None
    S = load(f)
    gap, err = [], []
    for p in pairs:
        A = recs[p].get("model", {}).get("A")
        B = S.get(p, {}).get("A_inv")
        gap.append(corner_dist(A, B))
        err.append(protocol.pair_error(A, *ds.checkpoints("val", p)))
    gap, err = np.array(gap), np.array(err)
    rho, n = spearman(gap, err)
    out = {"n": n, "fail_swap": int((~np.isfinite(gap)).sum()), "gap": q(gap), "spearman_vs_error": rho, "bins": {}}
    for lo, hi, name in CH_BINS:
        k = (gap >= lo) & (gap < hi)
        out["bins"][name] = {"n": int(k.sum()), "median_error": float(np.median(err[k])) if k.any() else None,
                             "auc5": protocol.auc(err[k], 5.0) if k.any() else None}
    return out


def main():
    ds = Dataset(os.environ["MOON_DATA"])
    pairs = ds.labelled("val")
    out = {"n": len(pairs), "conv": conv_report(ds), "models": {}, "minmax": {}}
    rng = np.random.default_rng(0)
    for m in MODELS:
        recs = load(D / f"rows_{m}.jsonl")
        e0 = errors(recs, pairs, ds, "model")
        rep = {"model": {"summary": {k: float(v) for k, v in protocol._point(e0).items() if k in ("auc@3", "auc@5", "sr@3", "sr@5", "median", "fail_rate")},
                         "ncc_median": float(np.median([recs[p]["model"]["ncc"] for p in pairs if "model" in recs[p]]))}}
        for row in ROWS[1:]:
            rep[row] = row_report(e0, errors(recs, pairs, ds, row), recs, pairs, row, rng)
        rep["swap"] = swap_report(recs, pairs, ds, m)
        out["models"][m] = rep
        fm = D / f"rows_{m}_minmax.jsonl"
        if fm.exists():
            rm = load(fm)
            out["minmax"][m] = {r: protocol.auc(errors(rm, pairs, ds, r), 5.0) for r in ROWS}
    (D / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    for m in MODELS:
        print(f"\n## {m}")
        for r in ROWS:
            x = out["models"][m][r]
            s = x["summary"]
            line = f"{NAMES[r]:12s} auc5 {s['auc@5']:.3f} auc3 {s['auc@3']:.3f} sr3 {s['sr@3']:.3f} sr5 {s['sr@5']:.3f} med {s['median']:.2f}"
            if r != "model":
                line += f" | d {x['d_auc5']:+.3f} [{x['d_auc5_ci'][0]:+.3f},{x['d_auc5_ci'][1]:+.3f}] better {x['better']} worse {x['worse']}"
            print(line)


if __name__ == "__main__":
    main()
