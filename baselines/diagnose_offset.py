"""诊断：3 px 以内的失败是系统性偏移还是随机噪声（「诊断：3 px 失败是系统性偏移还是随机噪声」#23）。

    python -m baselines.diagnose_offset $MOON_RESULTS/baselines --methods anymatch_loftr anymatch_roma \
        --split val --out runs/B0/extra/offset

逐 pair 比较两类误差（RANSAC 口径与 baselines.fit 相同，坐标同样 +0.5）：
- 自洽误差 r_self：RANSAC 内点在估计仿射 A 下的残差中位数；
- 真值误差 e_gt：检查点重投影误差（protocol.pair_error），并拆成
  共同偏移 |b|（检查点残差向量的均值）和去偏后的散布 s。
另外两个对照量：
- 标注地板 e_floor：对检查点自身最小二乘拟一个仿射 A_gt，它在检查点上的平均残差（标注噪声 + 非仿射成分）；
- 匹配集偏移 |u|：内点在 A_gt 下残差向量的均值，即整套匹配相对真值的共同平移。
写出 pairs.csv（逐对）、summary.json（分档统计）与图。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

BANDS = [(0, 3), (3, 5), (5, 10), (10, 20), (20, np.inf)]


def lstsq_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    X = np.hstack([src, np.ones((len(src), 1))])
    return np.linalg.lstsq(X, dst, rcond=None)[0].T  # 2×3


def apply(A, p):
    return p @ A[:, :2].T + A[:, 2]


def analyse(method_dir: Path, ds, split: str, ransac: float) -> list[dict]:
    from baselines.fit import fit_affine  # 需要 cv2；--replot 不走这里
    from workbench import protocol

    log = {}
    for line in (method_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            log[d["pair"]] = d
    rows = []
    with np.load(method_dir / f"{split}.npz") as z:
        for pair in ds.labelled(split):
            opt, sar = ds.checkpoints(split, pair)
            row = {"method": method_dir.name, "pair": pair, "roi": pair.split("/")[0], "n_cp": len(opt)}
            A_gt = lstsq_affine(opt, sar) if len(opt) >= 3 else None
            row["e_floor"] = float(np.linalg.norm(apply(A_gt, opt) - sar, axis=1).mean()) if A_gt is not None else np.nan
            k = pair.replace("/", "__")
            rec = log.get(pair)
            M = z[k] if (rec is not None and "error" not in rec and k in z.files) else np.zeros((0, 5), np.float32)
            A, inl, fail = fit_affine(M, ransac)
            if A is None:
                row.update(e_gt=np.inf)
                rows.append(row)
                continue
            src = M[:, :2].astype(np.float64) + 0.5
            dst = M[:, 2:4].astype(np.float64) + 0.5
            si, di = src[inl], dst[inl]
            d = protocol.checkpoint_residuals(A, opt, sar)
            b = d.mean(axis=0)
            row.update(
                n_inl=int(inl.sum()),
                r_self=float(np.median(np.linalg.norm(apply(A, si) - di, axis=1))),
                e_gt=protocol.pair_error(A, opt, sar),
                bx=float(b[0]), by=float(b[1]), b=float(np.linalg.norm(b)),
                s=float(np.linalg.norm(d - b, axis=1).mean()),
            )
            if A_gt is not None:
                u = apply(A_gt, si) - di
                um = u.mean(axis=0)
                row.update(ux=float(um[0]), uy=float(um[1]), u=float(np.linalg.norm(um)),
                           u_spread=float(np.median(np.linalg.norm(u - um, axis=1))))
            rows.append(row)
    return rows


def band_stats(rows: list[dict]) -> list[dict]:
    out = []
    e = np.array([r["e_gt"] for r in rows])
    for lo, hi in BANDS:
        sel = [r for r, x in zip(rows, e) if lo < x <= hi] if lo else [r for r, x in zip(rows, e) if x <= hi]
        if not sel:
            out.append({"band": f"({lo},{hi}]", "n": 0})
            continue
        med = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in sel]))
        out.append({
            "band": f"({lo},{hi}]", "n": len(sel),
            "e_gt": med("e_gt"), "r_self": med("r_self"), "b": med("b"), "s": med("s"),
            "b_over_e": float(np.nanmedian([r["b"] / r["e_gt"] for r in sel])),
            "e_floor": med("e_floor"), "u": med("u"), "u_spread": med("u_spread"),
        })
    return out


def roi_stats(rows: list[dict]) -> list[dict]:
    """3–10 px 档里，按 ROI 看共同偏移的方向是否一致：|mean b| / mean |b|，1 = 同向，≈0 = 方向随机。"""
    out = []
    mid = [r for r in rows if 3 < r["e_gt"] <= 10]
    for roi in sorted({r["roi"] for r in rows}):
        allr = [r for r in rows if r["roi"] == roi]
        sel = [r for r in mid if r["roi"] == roi]
        ok = [r for r in allr if np.isfinite(r["e_gt"])]
        item = {"roi": roi, "n": len(allr), "sr3": float(np.mean([r["e_gt"] <= 3 for r in allr])),
                "n_mid": len(sel), "e_floor": float(np.nanmedian([r["e_floor"] for r in allr]))}
        if ok:
            B = np.array([[r["bx"], r["by"]] for r in ok])
            item.update(mean_b=B.mean(axis=0).round(2).tolist(),
                        coherence=float(np.linalg.norm(B.mean(axis=0)) / np.linalg.norm(B, axis=1).mean()))
        out.append(item)
    return out


def plot(rows: list[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if np.isfinite(r["e_gt"])]
    methods = sorted({r["method"] for r in ok})
    fig, ax = plt.subplots(len(methods), 3, figsize=(15, 4.6 * len(methods)), squeeze=False)
    rois = sorted({r["roi"] for r in ok})
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(methods):
        R = [r for r in ok if r["method"] == m]
        e = np.array([r["e_gt"] for r in R]).clip(max=50)
        a = ax[i, 0]
        a.scatter([r["r_self"] for r in R], e, s=6, alpha=.5)
        a.axhline(3, c="r", lw=.8); a.axhline(10, c="r", lw=.8, ls="--")
        a.set_yscale("log"); a.set_xlabel("r_self (inlier residual median, px)"); a.set_ylabel("e_gt (px, clip 50)")
        a.set_title(f"{m}: self vs GT error")
        a = ax[i, 1]
        a.scatter([r["b"] for r in R], [r["s"] for r in R], s=6, alpha=.5, c=np.log10(e))
        lim = 15
        a.plot([0, lim], [0, lim], "k:", lw=.8); a.set_xlim(0, lim); a.set_ylim(0, lim)
        a.set_xlabel("|b| common offset (px)"); a.set_ylabel("s scatter after de-bias (px)")
        a.set_title("GT residual: offset vs scatter (color = log e_gt)")
        a = ax[i, 2]
        mid = [r for r in R if 3 < r["e_gt"] <= 10]
        for j, roi in enumerate(rois):
            S = [r for r in mid if r["roi"] == roi]
            if S:
                a.scatter([r["bx"] for r in S], [r["by"] for r in S], s=8, color=cmap(j % 10), label=roi)
        a.axhline(0, c="k", lw=.5); a.axvline(0, c="k", lw=.5); a.set_aspect("equal")
        a.set_xlim(-10, 10); a.set_ylim(-10, 10); a.legend(fontsize=7)
        a.set_title("offset vector b, pairs with 3<e<=10, by ROI")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k in ("method", "pair", "roi") else float(v) if v else np.nan) for k, v in r.items()})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", help="baselines.match 的 --out 目录")
    ap.add_argument("--methods", nargs="+", default=["anymatch_loftr", "anymatch_roma"])
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--no-plot", action="store_true", help="只写 csv/json（服务器 wb 环境没有 matplotlib）")
    ap.add_argument("--replot", action="store_true", help="只从已有 pairs_<split>.csv 出图")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        plot(read_csv(out / f"pairs_{args.split}.csv"), out / f"offset_{args.split}.png")
        return
    from workbench.dataset import Dataset

    ds = Dataset(args.data)
    rows, summary = [], {"split": args.split, "ransac": args.ransac, "methods": {}}
    for m in args.methods:
        R = analyse(Path(args.raw_root) / m, ds, args.split, args.ransac)
        rows += R
        summary["methods"][m] = {"bands": band_stats(R), "rois": roi_stats(R)}
    keys = ["method", "pair", "roi", "n_cp", "n_inl", "r_self", "e_gt", "bx", "by", "b", "s",
            "e_floor", "ux", "uy", "u", "u_spread"]
    with open(out / f"pairs_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    (out / f"summary_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not args.no_plot:
        plot(rows, out / f"offset_{args.split}.png")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
