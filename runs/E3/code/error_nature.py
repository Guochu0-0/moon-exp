"""误差的性质：整体偏移的统计描述与两种来源的检验（「【误差分析】误差的性质」#100）。只推理、只用 Val。

    # 1) 采集（服务器，RoMa 稠密对应要 GPU；其余只读已有预测与点对）
    GPU=1 /opt/envs/loftr/bin/python runs/E3/code/error_nature.py collect
    # 2) 统计与图（纯 numpy + matplotlib，读 1) 的产物）
    /opt/envs/loftr/bin/python runs/E3/code/error_nature.py analyse
    # 3) 自动挑出的示意对（要读影像）
    /opt/envs/loftr/bin/python runs/E3/code/error_nature.py examples

坐标全部是评测口径（原 512 网格、角点原点，与 A 和标注相同）。每个模型、每个有标注的 pair、每个标注点记：
- 仿射位置 A·o；
- 邻近匹配给出的局部对应：光学侧落在标注点 R px 内的原始点对（RANSAC 之前的全部点），
  取各点位移（SAR − 光学）的中位数加到标注点上；不足 MIN_NB 个点记缺失；
- RoMa 另记稠密对应：网络 warp 在标注点处双线性取样，及该处 certainty。
产物写到 runs/E3/extra/。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from workbench.dataset import Dataset   # noqa: E402

EXP = REPO / "runs" / "E3"
OUT = EXP / "extra"
DATA = OUT / "data"
YGC = Path("/remote-home/xufang/YGC")
MAIN = YGC / "moon-exp"         # 不进 git 的点对与 ckpt 只在 gpfs 主 checkout
SPLIT = "val"
R_NB, MIN_NB = 12.0, 3          # 邻近匹配：半径（px）与最少点数
SANE = 20.0                     # 整体偏移超过它视为粗错，不进相关与回归

# 6 个模型：显示名、预测、原始点对（baselines.match 格式：中心约定、全部点）、RoMa 的权重
MODELS = {
    "loftr_zs": dict(name="LoFTR zero-shot", base="loftr",
                     preds="runs/B0/preds/anymatch_loftr",
                     raw=YGC / "results/baselines/anymatch_loftr"),
    "loftr_q4": dict(name="LoFTR 无标注在线训练", base="loftr",
                     preds="runs/Q/preds/Q4", raw=MAIN / "runs/Q/sweep/Q4/match/step1500"),
    "loftr_e1": dict(name="LoFTR 标注过拟合", base="loftr",
                     preds="runs/E1/preds/loftr_lr5e5", raw=MAIN / "runs/E1/sweep/loftr_lr5e5/match/step14000"),
    "roma_zs": dict(name="RoMa zero-shot", base="roma",
                    preds="runs/B0m/preds/anymatch_roma__minmax",
                    raw=YGC / "results/baselines_ablation/anymatch_roma__minmax", ckpt=None),
    "roma_m4": dict(name="RoMa 无标注自训练", base="roma",
                    preds="runs/M/preds/M4", raw=MAIN / "runs/M/sweep/M4/match/step2000",
                    ckpt=MAIN / "runs/M/ckpt/M4/ckpt_2000.pt"),
    "roma_e1": dict(name="RoMa 标注过拟合", base="roma",
                    preds="runs/E1/preds/roma_vgg", raw=MAIN / "runs/E1/sweep/roma_vgg/match/step16000",
                    ckpt=MAIN / "runs/E1/ckpt/roma_vgg/ckpt_16000.pt"),
}
FIELDS = ["pair", "k", "ox", "oy", "sx", "sy", "ax", "ay", "lx", "ly", "ln", "dx", "dy", "dc"]


def load_preds(d) -> dict[str, np.ndarray | None]:
    out = {}
    for line in (REPO / d / f"{SPLIT}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["pair"]] = None if r.get("A") is None else np.asarray(r["A"], np.float64).reshape(2, 3)
    return out


def apply(A, p):
    return p @ A[:, :2].T + A[:, 2]


def local_nb(M: np.ndarray, o: np.ndarray):
    """邻近匹配的局部对应。M: N×5 中心约定 → 评测口径 +0.5。返回 (K×2，缺失为 nan；K 个邻居数)。"""
    out, cnt = np.full((len(o), 2), np.nan), np.zeros(len(o), int)
    if len(M) == 0:
        return out, cnt
    src = M[:, :2].astype(np.float64) + 0.5
    disp = M[:, 2:4].astype(np.float64) + 0.5 - src
    for i, p in enumerate(o):
        nb = np.linalg.norm(src - p, axis=1) <= R_NB
        cnt[i] = int(nb.sum())
        if cnt[i] >= MIN_NB:
            out[i] = p + np.median(disp[nb], axis=0)
    return out, cnt


class RomaDense:
    """RoMa 稠密对应：与评测同一套输入（推理配置的映射、适配器的 640 网格 uint8 PIL），warp 在标注点处取样。"""

    def __init__(self, ckpt, device="cuda"):
        os.environ.setdefault("TORCH_HOME", str(YGC / "weights/torch_home"))
        from baselines.adapters.romatch import RomatchAdapter
        from moonlib import inputs

        cfg = json.loads((REPO / "configs/baselines/anymatch_roma__minmax.json").read_text(encoding="utf-8"))
        w = ckpt or (YGC / "weights" / cfg["weights"])
        self.ad = RomatchAdapter(REPO / cfg["repo"], str(w), device=device, **cfg["params"])
        self.map_opt = inputs.get("optical", cfg["input"]["optical"])
        self.map_sar = inputs.get("sar", cfg["input"]["sar"])
        self.seed, self.device = int(cfg.get("seed", 0)), device

    def __call__(self, ds, pair, o):
        import torch
        import torch.nn.functional as F
        from baselines.match import seed_all

        opt, sar = self.map_opt(ds.optical(SPLIT, pair)), self.map_sar(ds.sar(SPLIT, pair))
        p0, g0 = self.ad._pil(opt)
        p1, g1 = self.ad._pil(sar)
        seed_all(self.seed)
        with torch.no_grad():
            warp, cert = self.ad.model.match(p0, p1, batched=False, device=self.device)
            W = warp.shape[1] // 2                                  # 对称：左半是光学网格上的 A→B
            h0, w0 = g0[0], g0[1]
            grid = torch.tensor(np.c_[o[:, 0] / (w0 / 2) - 1, o[:, 1] / (h0 / 2) - 1],
                                dtype=torch.float32, device=warp.device)[None, None]
            ab = warp[:, :W, 2:4].permute(2, 0, 1)[None].float()
            q = F.grid_sample(ab, grid, mode="bilinear", align_corners=False)[0, :, 0].T.cpu().numpy()
            c = F.grid_sample(cert[:, :W][None, None].float(), grid, mode="bilinear",
                              align_corners=False)[0, 0, 0].cpu().numpy()
            # 自检用：与评测同样的采样 + RANSAC
            m, conf = self.ad.model.sample(warp, cert)
            k0, k1 = self.ad.model.to_pixel_coordinates(m, g0[2], g0[3], g1[2], g1[3])
        h1, w1 = g1[0], g1[1]
        dense = np.c_[(q[:, 0] + 1) * w1 / 2, (q[:, 1] + 1) * h1 / 2]
        from baselines.adapters.base import to_original
        kp0 = to_original(k0.float().cpu().numpy() - 0.5, *g0)
        kp1 = to_original(k1.float().cpu().numpy() - 0.5, *g1)
        return dense, c, np.c_[kp0, kp1, conf.float().cpu().numpy()]


def collect(args):
    from moonlib.ransac import fit_affine

    ds = Dataset(args.data)
    pairs = ds.labelled(SPLIT)[: args.limit or None]
    DATA.mkdir(parents=True, exist_ok=True)
    check = {}
    for key in args.models:
        cfg = MODELS[key]
        P = load_preds(cfg["preds"])
        dense = RomaDense(cfg.get("ckpt")) if cfg["base"] == "roma" else None
        rows, dev_raw, dev_dense = [], [], []
        with np.load(cfg["raw"] / f"{SPLIT}.npz") as z:
            for j, pair in enumerate(pairs):
                o, s = ds.checkpoints(SPLIT, pair)
                A = P.get(pair)
                a = apply(A, o) if A is not None else np.full_like(o, np.nan)
                k = pair.replace("/", "__")
                M = z[k] if k in z.files else np.zeros((0, 5), np.float32)
                # 自检：原始点对重新 RANSAC 应复现已存的仿射
                Ar, _, _ = fit_affine(M, 3.0)
                if A is not None and Ar is not None:
                    dev_raw.append(float(np.abs(apply(Ar, o) - a).max()))
                loc, cnt = local_nb(M, o)
                dd, dc = np.full_like(o, np.nan), np.full(len(o), np.nan)
                if dense is not None:
                    dd, dc, Ms = dense(ds, pair, o)
                    if j < args.selfcheck:
                        Ad, _, _ = fit_affine(Ms, 3.0)
                        if A is not None and Ad is not None:
                            dev_dense.append(float(np.abs(apply(Ad, o) - a).max()))
                for i in range(len(o)):
                    rows.append([pair, i, *o[i], *s[i], *a[i], *loc[i], int(cnt[i]), *dd[i], dc[i]])
                if (j + 1) % 100 == 0:
                    print(f"[{key}] {j + 1}/{len(pairs)}", flush=True)
        with open(DATA / f"points_{key}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for r in rows:
                w.writerow([r[0], r[1]] + [f"{v:.4f}" if isinstance(v, float) else v for v in r[2:]])
        q = lambda v: {"n": len(v), "median": float(np.median(v)) if v else None,
                       "p95": float(np.percentile(v, 95)) if v else None, "max": max(v) if v else None}
        check[key] = {"refit_vs_stored_px": q(dev_raw), "dense_sample_refit_vs_stored_px": q(dev_dense)}
        print(key, json.dumps(check[key], ensure_ascii=False), flush=True)
    path = DATA / "selfcheck.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(check)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 统计

def read_points(key):
    with open(DATA / f"points_{key}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["pair"], []).append([float(r[c]) if r[c] not in ("", "nan") else np.nan for c in FIELDS[2:]])
    return {p: np.array(v) for p, v in by.items()}


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def ols(x, y):
    """y = a + b·x 的斜率与 95% 区间（pair 级 bootstrap 1000 次）。"""
    rng = np.random.default_rng(0)
    b = np.polyfit(x, y, 1)[0]
    bs = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
    return float(b), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def per_pair(pts, local="nb", cert_min=None):
    """每对：整体偏移 b、剩余部分大小、误差；局部对应的整体偏移 b_loc（只用有局部对应的点）。"""
    out = {}
    for p, v in pts.items():
        o, s, a = v[:, 0:2], v[:, 2:4], v[:, 4:6]
        loc = v[:, 6:8] if local == "nb" else v[:, 9:11]
        if not np.isfinite(a).all():
            out[p] = None
            continue
        d = a - s
        b = d.mean(0)
        ok = np.isfinite(loc).all(1)
        if cert_min is not None:
            ok &= v[:, 11] >= cert_min
        rec = {"b": b, "rem": float(np.linalg.norm(d - b, axis=1).mean()), "err": float(np.linalg.norm(d, axis=1).mean()),
               "msd": float((np.linalg.norm(d, axis=1) ** 2).mean()), "n": len(v), "n_loc": int(ok.sum())}
        if ok.sum() >= 2:
            e = loc[ok] - s[ok]
            rec.update(b_loc=e.mean(0), b_aff_sub=d[ok].mean(0),
                       e_loc=np.linalg.norm(e, axis=1), e_aff=np.linalg.norm(d[ok], axis=1))
        out[p] = rec
    return out


def source_test(pp):
    """局部对应的整体偏移跟着仿射的整体偏移走多少：b_loc 对 b（同一批点上的仿射整体偏移）回归。"""
    rows = [r for r in pp.values() if r is not None and "b_loc" in r and np.abs(r["b_aff_sub"]).max() < SANE]
    B = np.array([r["b_aff_sub"] for r in rows])
    L = np.array([r["b_loc"] for r in rows])
    ea = np.concatenate([r["e_aff"] for r in rows])
    el = np.concatenate([r["e_loc"] for r in rows])
    nb, nl = np.linalg.norm(B, axis=1), np.linalg.norm(L, axis=1)
    big = nb >= np.percentile(nb, 75)
    res = {"n_pairs": len(rows), "n_points": int(len(ea)),
           "median_point_err_affine": float(np.median(ea)), "median_point_err_local": float(np.median(el)),
           "median_offset_affine": float(np.median(nb)), "median_offset_local": float(np.median(nl)),
           "big_quarter": {"thr": float(np.percentile(nb, 75)),
                           "median_offset_affine": float(np.median(nb[big])),
                           "median_offset_local": float(np.median(nl[big])),
                           "frac_local_lt_half": float((nl[big] < nb[big] / 2).mean()),
                           "frac_same_dir": float(((B[big] * L[big]).sum(1) > 0).mean())}}
    for ax, c in (("x", 0), ("y", 1)):
        slope, ci = ols(B[:, c], L[:, c])
        res[f"follow_{ax}"] = {"slope": slope, "ci95": ci, "pearson": float(np.corrcoef(B[:, c], L[:, c])[0, 1]),
                               "std_affine": float(B[:, c].std()), "std_local": float(L[:, c].std())}
    return res, B, L


def analyse(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    keys = list(MODELS)
    pts = {k: read_points(k) for k in keys}
    pairs = sorted(set.intersection(*(set(v) for v in pts.values())))
    summary = {"n_pairs": len(pairs), "radius_px": R_NB, "min_neighbours": MIN_NB, "sane_px": SANE,
               "describe": {}, "agree": {}, "source": {}}

    # 1. 整体偏移 + 剩余部分
    PP = {k: per_pair(pts[k]) for k in keys}
    Bx = {}
    for k in keys:
        recs = [PP[k][p] for p in pairs]
        okr = [r for r in recs if r is not None]
        b = np.array([r["b"] for r in okr])
        sane = np.abs(b).max(1) < SANE
        summary["describe"][k] = {
            "n": len(recs), "n_fail": sum(r is None for r in recs), "n_gross": int((~sane).sum()),
            "median_err": float(np.median([r["err"] for r in okr])),
            "median_offset": float(np.median(np.linalg.norm(b, axis=1))),
            "median_rem": float(np.median([r["rem"] for r in okr])),
            "share_sq_offset": float((np.linalg.norm(b[sane], axis=1) ** 2).sum()
                                     / np.sum([r["msd"] for r, ok in zip(okr, sane) if ok])),
            "std_bx": float(b[sane, 0].std()), "std_by": float(b[sane, 1].std()),
            "median_abs_bx": float(np.median(np.abs(b[sane, 0]))), "median_abs_by": float(np.median(np.abs(b[sane, 1]))),
        }
        Bx[k] = np.array([PP[k][p]["b"] if PP[k][p] is not None else [np.nan, np.nan] for p in pairs])

    # 2. 6 个模型之间：x / y 整体偏移的相关与同号率
    allsane = np.all([np.isfinite(Bx[k]).all(1) & (np.abs(Bx[k]).max(1) < SANE) for k in keys], axis=0)
    summary["agree"]["n_pairs_all_sane"] = int(allsane.sum())
    for c, ax in ((0, "x"), (1, "y")):
        C = np.zeros((6, 6))
        S = np.full((6, 6), np.nan)
        N = np.zeros((6, 6), int)
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                xi, xj = Bx[ki][allsane, c], Bx[kj][allsane, c]
                C[i, j] = np.corrcoef(xi, xj)[0, 1]
                both = (np.abs(xi) > 1) & (np.abs(xj) > 1)
                N[i, j] = both.sum()
                if i != j and both.any():
                    S[i, j] = (np.sign(xi[both]) == np.sign(xj[both])).mean()
        summary["agree"][ax] = {"pearson": C.round(3).tolist(), "same_sign_gt1px": np.round(S, 3).tolist(),
                                "n_both_gt1px": N.tolist()}
    # 「整体偏移」能否被 6 个模型的均值解释：每个模型的 bx 与其余 5 个均值的相关
    mean_bx = np.mean([Bx[k][allsane, 0] for k in keys], 0)
    summary["agree"]["x_vs_mean_of_others"] = {
        k: float(np.corrcoef(Bx[k][allsane, 0], (mean_bx * 6 - Bx[k][allsane, 0]) / 5)[0, 1]) for k in keys}

    # 3. 来源检验
    srcB, srcL = {}, {}
    for k in keys:
        res, B, L = source_test(PP[k])
        summary["source"][k] = {"neighbour": res}
        srcB[k], srcL[k] = B, L
        if MODELS[k]["base"] == "roma":
            ppd = per_pair(pts[k], local="dense")
            res_d, Bd, Ld = source_test(ppd)
            cert = np.concatenate([v[:, 11] for v in pts[k].values()])
            cm = float(np.nanmedian(cert))
            res_c, _, _ = source_test(per_pair(pts[k], local="dense", cert_min=cm))
            res_c["cert_min"] = cm
            summary["source"][k].update(dense=res_d, dense_high_cert=res_c)
            srcB[k], srcL[k] = Bd, Ld          # RoMa 的图用稠密对应
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    names = [MODELS[k]["name"] for k in keys]
    # 图 1：各模型整体偏移 (bx, by)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for a, k in zip(axs.flat, keys):
        b = Bx[k][allsane]
        a.scatter(b[:, 0], b[:, 1], s=4, alpha=0.4)
        a.axhline(0, c="k", lw=0.5)
        a.axvline(0, c="k", lw=0.5)
        d = summary["describe"][k]
        a.set_title(f"{MODELS[k]['name']}\nstd x {d['std_bx']:.2f}, y {d['std_by']:.2f} px", fontsize=10)
        a.set_xlim(-10, 10)
        a.set_ylim(-10, 10)
        a.set_aspect("equal")
    for a in axs[1]:
        a.set_xlabel("整体偏移 x (px)")
    for a in axs[:, 0]:
        a.set_ylabel("整体偏移 y (px)")
    fig.tight_layout()
    fig.savefig(OUT / "offset_xy.png", dpi=130)
    plt.close(fig)

    # 图 2：误差中整体偏移与剩余部分
    fig, a = plt.subplots(figsize=(8, 4))
    x = np.arange(6)
    off = [summary["describe"][k]["median_offset"] for k in keys]
    rem = [summary["describe"][k]["median_rem"] for k in keys]
    a.bar(x - 0.2, off, 0.4, label="整体偏移大小（中位数）")
    a.bar(x + 0.2, rem, 0.4, label="剩余部分大小（中位数）")
    for i, k in enumerate(keys):
        a.text(i, max(off[i], rem[i]) + 0.05, f"{summary['describe'][k]['share_sq_offset']:.0%}", ha="center", fontsize=9)
    a.set_xticks(x, names, rotation=20, fontsize=9)
    a.set_ylabel("px")
    a.legend(fontsize=9)
    a.set_title("柱上数字：整体偏移占均方误差的比例", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "offset_vs_rest.png", dpi=130)
    plt.close(fig)

    # 图 3：模型之间 x 向整体偏移的相关（上三角）与同号率（下三角）
    fig, a = plt.subplots(figsize=(7, 6))
    C = np.array(summary["agree"]["x"]["pearson"])
    S = np.array(summary["agree"]["x"]["same_sign_gt1px"], dtype=float)
    M = np.where(np.triu(np.ones((6, 6)), 1) > 0, C, np.where(np.tril(np.ones((6, 6)), -1) > 0, S, np.nan))
    im = a.imshow(M, vmin=0, vmax=1, cmap="viridis")
    for i in range(6):
        for j in range(6):
            if i != j:
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.6 else "k", fontsize=9)
    a.set_xticks(range(6), names, rotation=30, ha="right", fontsize=8)
    a.set_yticks(range(6), names, fontsize=8)
    a.set_title("x 向整体偏移：右上为相关系数，左下为同号率（两者都超过 1 px 的对）", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "offset_agree_x.png", dpi=130)
    plt.close(fig)

    # 图 4：局部对应的整体偏移 vs 仿射的整体偏移（x）
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for a, k in zip(axs.flat, keys):
        B, L = srcB[k], srcL[k]
        a.scatter(B[:, 0], L[:, 0], s=4, alpha=0.4)
        lim = 8
        a.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, label="局部完全跟随仿射")
        a.axhline(0, c="g", ls="--", lw=0.8, label="局部与标注一致")
        src = summary["source"][k]["dense" if MODELS[k]["base"] == "roma" else "neighbour"]["follow_x"]
        a.set_title(f"{MODELS[k]['name']}（{'稠密对应' if MODELS[k]['base'] == 'roma' else '邻近匹配'}）\n"
                    f"斜率 {src['slope']:.2f} [{src['ci95'][0]:.2f}, {src['ci95'][1]:.2f}]", fontsize=10)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
    axs[0, 0].legend(fontsize=8, loc="upper left")
    for a in axs[1]:
        a.set_xlabel("仿射在标注点处的 x 向整体偏移 (px)")
    for a in axs[:, 0]:
        a.set_ylabel("局部对应的 x 向整体偏移 (px)")
    fig.tight_layout()
    fig.savefig(OUT / "local_vs_affine_x.png", dpi=130)
    plt.close(fig)

    # 图 5：逐点误差，按仿射整体偏移大小分四档
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for a, k in zip(axs.flat, keys):
        pp = per_pair(pts[k], local="dense" if MODELS[k]["base"] == "roma" else "nb")
        rows = [r for r in pp.values() if r is not None and "b_loc" in r and np.abs(r["b_aff_sub"]).max() < SANE]
        nb = np.array([np.linalg.norm(r["b_aff_sub"]) for r in rows])
        edges = np.percentile(nb, [0, 25, 50, 75, 100])
        ma, ml, lab = [], [], []
        for q in range(4):
            sel = [r for r, v in zip(rows, nb) if edges[q] <= v <= edges[q + 1]]
            ma.append(np.median(np.concatenate([r["e_aff"] for r in sel])))
            ml.append(np.median(np.concatenate([r["e_loc"] for r in sel])))
            lab.append(f"{edges[q]:.1f}–{edges[q + 1]:.1f}")
        x = np.arange(4)
        a.bar(x - 0.2, ma, 0.4, label="仿射")
        a.bar(x + 0.2, ml, 0.4, label="局部对应")
        a.set_xticks(x, lab, fontsize=8)
        a.set_title(MODELS[k]["name"], fontsize=10)
    axs[0, 0].legend(fontsize=8)
    for a in axs[1]:
        a.set_xlabel("该对仿射整体偏移大小分档 (px)")
    for a in axs[:, 0]:
        a.set_ylabel("标注点处误差中位数 (px)")
    fig.tight_layout()
    fig.savefig(OUT / "point_err_by_offset.png", dpi=130)
    plt.close(fig)

    # 示意对：按 6 个模型平均的「仿射整体偏移 − 局部整体偏移」挑最高、中位、最低
    score = {}
    for p in pairs:
        v = []
        for k in keys:
            pp = PP[k][p]
            if pp is None or "b_loc" not in pp:
                break
            v.append(np.linalg.norm(pp["b_aff_sub"]) - np.linalg.norm(pp["b_loc"]))
        if len(v) == 6:
            score[p] = float(np.mean(v))
    order = sorted(score, key=score.get)
    pick = {"最高": order[-1], "中位": order[len(order) // 2], "最低": order[0]}
    (DATA / "examples.json").write_text(json.dumps({t: [p, score[p]] for t, p in pick.items()}, ensure_ascii=False,
                                                   indent=1), encoding="utf-8")
    print(json.dumps(summary["describe"], ensure_ascii=False, indent=1))
    print(json.dumps({k: v["neighbour"]["follow_x"] for k, v in summary["source"].items()}, ensure_ascii=False))


def examples(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from moonlib import inputs
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]

    ds = Dataset(args.data)
    pick = json.loads((DATA / "examples.json").read_text(encoding="utf-8"))
    keys = list(MODELS)
    pts = {k: read_points(k) for k in keys}
    G = 10.0                                     # 误差向量放大倍数
    for tag, (pair, sc) in pick.items():
        sar = inputs.sar_p2p98(ds.sar(SPLIT, pair))
        fig, axs = plt.subplots(2, 3, figsize=(13, 9))
        for a, k in zip(axs.flat, keys):
            v = pts[k][pair]
            s, aff = v[:, 2:4], v[:, 4:6]
            loc = v[:, 9:11] if MODELS[k]["base"] == "roma" else v[:, 6:8]
            a.imshow(sar, cmap="gray", extent=(0, sar.shape[1], sar.shape[0], 0))
            a.scatter(s[:, 0], s[:, 1], s=14, c="lime", label="标注")
            for (p0, p1, c) in ((s, aff, "r"), (s, loc, "c")):
                for i in range(len(s)):
                    if np.isfinite(p1[i]).all():
                        a.annotate("", xy=p0[i] + G * (p1[i] - p0[i]), xytext=p0[i],
                                   arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
            a.set_title(MODELS[k]["name"], fontsize=10)
            a.axis("off")
        axs[0, 0].plot([], [], "r", label="仿射误差 ×10")
        axs[0, 0].plot([], [], "c", label="局部对应误差 ×10")
        axs[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle(f"{pair}（SAR）：仿射整体偏移比局部整体偏移大 {sc:.2f} px（6 个模型平均）", fontsize=11)
        fig.tight_layout()
        name = {"最高": "high", "中位": "median", "最低": "low"}[tag]
        fig.savefig(OUT / f"example_{name}.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "analyse", "examples"])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA", str(YGC / "dataset/Moon")))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="只跑前若干对（试跑）")
    ap.add_argument("--selfcheck", type=int, default=30, help="RoMa：前多少对用稠密采样 + RANSAC 复现已存仿射")
    args = ap.parse_args()
    if args.cmd == "collect" and os.environ.get("GPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
    {"collect": collect, "analyse": analyse, "examples": examples}[args.cmd](args)


if __name__ == "__main__":
    main()
