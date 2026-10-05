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
from baselines.data import Data          # noqa: E402
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

    def __call__(self, img, pair, o):
        import torch
        import torch.nn.functional as F
        from baselines.match import seed_all

        opt, sar = self.map_opt(img.optical(SPLIT, pair)), self.map_sar(img.sar(SPLIT, pair))
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

    ds, img = Dataset(args.data), Data(args.data)      # 影像用推理同一个读取器（不依赖 tifffile）
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
                    dd, dc, Ms = dense(img, pair, o)
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

LOCAL_COL = {"nb": slice(6, 8), "dense": slice(9, 11)}
MAIN_LOCAL = {k: "dense" if v["base"] == "roma" else "nb" for k, v in MODELS.items()}
LOCAL_NAME = {"nb": "邻近匹配", "dense": "稠密对应"}


def read_points(key):
    with open(DATA / f"points_{key}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["pair"], []).append([float(r[c]) if r[c] not in ("", "nan") else np.nan for c in FIELDS[2:]])
    return {p: np.array(v) for p, v in by.items()}


def ols(x, y):
    """y = a + b·x 的斜率与 95% 区间（pair 级 bootstrap 1000 次）。"""
    rng = np.random.default_rng(0)
    b = np.polyfit(x, y, 1)[0]
    bs = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
    return float(b), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def lsq_affine(o, s):
    return np.linalg.lstsq(np.c_[o, np.ones(len(o))], s, rcond=None)[0].T


def label_fit_residuals(o, s):
    """标注点自拟合仿射在各点的残差（标注 − 拟合）：样本内，以及留一（拟合不含该点）。点太少时为 nan。"""
    n = len(o)
    g_in = s - apply(lsq_affine(o, s), o) if n >= 4 else np.full_like(o, np.nan)
    g_loo = np.full_like(o, np.nan)
    if n >= 5:
        for i in range(n):
            keep = np.arange(n) != i
            g_loo[i] = s[i] - apply(lsq_affine(o[keep], s[keep]), o[i:i + 1])[0]
    return g_in, g_loo


def per_pair(pts, local="nb", cert_min=None):
    """每对：仿射误差的整体偏移 b、剩余部分、误差；局部对应在同一批点上的整体偏移 b_loc 与逐点误差。
    局部对应离标注超过 SANE 的点视为误匹配，不用。"""
    out = {}
    for p, v in pts.items():
        o, s, a, loc = v[:, 0:2], v[:, 2:4], v[:, 4:6], v[:, LOCAL_COL[local]]
        if not np.isfinite(a).all():
            out[p] = None
            continue
        d = a - s
        b = d.mean(0)
        rec = {"b": b, "rem": float(np.linalg.norm(d - b, axis=1).mean()), "err": float(np.linalg.norm(d, axis=1).mean()),
               "msd": float((np.linalg.norm(d, axis=1) ** 2).mean()), "n": len(v)}
        e = loc - s
        ok = np.isfinite(e).all(1)
        ok[ok] = np.linalg.norm(e[ok], axis=1) < SANE
        if cert_min is not None:
            ok &= v[:, 11] >= cert_min
        rec["n_loc"] = int(ok.sum())
        if ok.sum() >= 2:
            g_in, g_loo = label_fit_residuals(o, s)
            rec.update(b_loc=e[ok].mean(0), b_aff_sub=d[ok].mean(0), e=e[ok], d=d[ok], g_in=g_in[ok], g_loo=g_loo[ok])
        out[p] = rec
    return out


def usable(pp):
    return [r for r in pp.values() if r is not None and "b_loc" in r and np.abs(r["b_aff_sub"]).max() < SANE]


def source_test(pp):
    """来源检验。逐点：局部对应、仿射、标注自拟合仿射在标注点处的误差；逐对：局部对应的整体偏移
    跟着仿射的整体偏移走多少（b_loc 对 b 回归，b 只取有局部对应的点）。"""
    rows = usable(pp)
    B = np.array([r["b_aff_sub"] for r in rows])
    L = np.array([r["b_loc"] for r in rows])
    E, D = np.concatenate([r["e"] for r in rows]), np.concatenate([r["d"] for r in rows])
    Gi, Gl = np.concatenate([r["g_in"] for r in rows]), np.concatenate([r["g_loo"] for r in rows])
    nrm = lambda x: np.linalg.norm(x, axis=1)
    med = lambda x: float(np.nanmedian(x))
    n_all = sum(len(r["e"]) for r in rows)
    res = {"n_pairs": len(rows), "n_points": int(n_all),
           "point_median": {"local": med(nrm(E)), "affine": med(nrm(D)), "label_fit_in": med(nrm(Gi)),
                            "label_fit_loo": med(nrm(Gl))},
           "point_median_x": {"local": med(np.abs(E[:, 0])), "affine": med(np.abs(D[:, 0])),
                              "label_fit_in": med(np.abs(Gi[:, 0])), "label_fit_loo": med(np.abs(Gl[:, 0]))},
           "point_median_y": {"local": med(np.abs(E[:, 1])), "affine": med(np.abs(D[:, 1])),
                              "label_fit_in": med(np.abs(Gi[:, 1])), "label_fit_loo": med(np.abs(Gl[:, 1]))},
           "frac_local_closer": float((nrm(E) < nrm(D)).mean()),
           "offset_median": {"affine": med(nrm(B)), "local": med(nrm(L))}}
    nb, nl = nrm(B), nrm(L)
    big = nb >= np.percentile(nb, 75)
    res["big_quarter"] = {"thr": float(np.percentile(nb, 75)), "offset_affine": med(nb[big]), "offset_local": med(nl[big]),
                          "frac_local_lt_half": float((nl[big] < nb[big] / 2).mean())}
    for ax, c in (("x", 0), ("y", 1)):
        slope, ci = ols(B[:, c], L[:, c])
        res[f"follow_{ax}"] = {"slope": slope, "ci95": ci, "pearson": float(np.corrcoef(B[:, c], L[:, c])[0, 1]),
                               "std_affine": float(B[:, c].std()), "std_local": float(L[:, c].std())}
    return res, B, L


def corr_matrix(V, keys, both_gt=1.0):
    """V[k]: 每对的值（nan = 缺）。返回相关矩阵、同号率（两者绝对值都 > both_gt 的对）、对数。"""
    n = len(keys)
    C, S, N = np.eye(n), np.full((n, n), np.nan), np.zeros((n, n), int)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                continue
            ok = np.isfinite(V[ki]) & np.isfinite(V[kj])
            xi, xj = V[ki][ok], V[kj][ok]
            C[i, j] = np.corrcoef(xi, xj)[0, 1]
            both = (np.abs(xi) > both_gt) & (np.abs(xj) > both_gt)
            N[i, j] = both.sum()
            if both.any():
                S[i, j] = (np.sign(xi[both]) == np.sign(xj[both])).mean()
    return C, S, N


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def analyse(args):
    plt = setup_plt()
    keys = list(MODELS)
    names = [MODELS[k]["name"] for k in keys]
    pts = {k: read_points(k) for k in keys}
    pairs = sorted(set.intersection(*(set(v) for v in pts.values())))
    summary = {"n_pairs": len(pairs), "radius_px": R_NB, "min_neighbours": MIN_NB, "sane_px": SANE,
               "main_local": MAIN_LOCAL, "describe": {}, "agree": {}, "source": {}}

    # 1. 整体偏移 + 剩余部分
    PP = {k: per_pair(pts[k], MAIN_LOCAL[k]) for k in keys}
    Bv = {}
    for k in keys:
        okr = [PP[k][p] for p in pairs if PP[k][p] is not None]
        b = np.array([r["b"] for r in okr])
        sane = np.abs(b).max(1) < SANE
        msd = np.array([r["msd"] for r in okr])
        summary["describe"][k] = {
            "n": len(pairs), "n_fail": len(pairs) - len(okr), "n_gross": int((~sane).sum()),
            "median_err": float(np.median([r["err"] for r in okr])),
            "median_offset": float(np.median(np.linalg.norm(b[sane], axis=1))),
            "median_rem": float(np.median(np.array([r["rem"] for r in okr])[sane])),
            "share_sq_offset": float((np.linalg.norm(b[sane], axis=1) ** 2).sum() / msd[sane].sum()),
            "std_bx": float(b[sane, 0].std()), "std_by": float(b[sane, 1].std()),
            "median_abs_bx": float(np.median(np.abs(b[sane, 0]))), "median_abs_by": float(np.median(np.abs(b[sane, 1]))),
        }
        B = np.array([PP[k][p]["b"] if PP[k][p] is not None else [np.nan, np.nan] for p in pairs])
        B[~(np.abs(B).max(1) < SANE)] = np.nan
        Bv[k] = B

    # 2. 6 个模型之间：整体偏移的相关与同号率；局部对应整体偏移的相关
    for c, ax in ((0, "x"), (1, "y")):
        C, S, N = corr_matrix({k: Bv[k][:, c] for k in keys}, keys)
        summary["agree"][ax] = {"pearson": C.round(3).tolist(), "same_sign_gt1px": np.round(S, 3).tolist(),
                                "n_both_gt1px": N.tolist()}
    off = np.array([[np.nanmean(Bv[k][:, 0]) for k in keys]])
    summary["agree"]["mean_bx"] = dict(zip(keys, off[0].round(3).tolist()))
    Lx = {}
    for k in keys:
        Lx[k] = np.array([PP[k][p]["b_loc"][0] if PP[k][p] is not None and "b_loc" in PP[k][p]
                          and np.abs(PP[k][p]["b_aff_sub"]).max() < SANE else np.nan for p in pairs])
    C, S, N = corr_matrix(Lx, keys)
    summary["agree"]["local_x"] = {"pearson": C.round(3).tolist(), "same_sign_gt1px": np.round(S, 3).tolist(),
                                   "n_both_gt1px": N.tolist()}

    # 3. 来源检验
    srcB, srcL = {}, {}
    for k in keys:
        summary["source"][k] = {}
        for loc in (("nb", "dense") if MODELS[k]["base"] == "roma" else ("nb",)):
            pp = PP[k] if loc == MAIN_LOCAL[k] else per_pair(pts[k], loc)
            res, B, L = source_test(pp)
            summary["source"][k][loc] = res
            if loc == MAIN_LOCAL[k]:
                srcB[k], srcL[k] = B, L
        if MODELS[k]["base"] == "roma":
            cm = float(np.nanmedian(np.concatenate([v[:, 11] for v in pts[k].values()])))
            res_c, _, _ = source_test(per_pair(pts[k], "dense", cert_min=cm))
            res_c["cert_min"] = cm
            summary["source"][k]["dense_high_cert"] = res_c
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # 图：各模型整体偏移 (bx, by)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for a, k in zip(axs.flat, keys):
        b = Bv[k]
        a.scatter(b[:, 0], b[:, 1], s=4, alpha=0.4)
        a.axhline(0, c="k", lw=0.5)
        a.axvline(0, c="k", lw=0.5)
        d = summary["describe"][k]
        a.set_title(f"{MODELS[k]['name']}\n标准差 x {d['std_bx']:.2f} px，y {d['std_by']:.2f} px", fontsize=10)
        a.set_xlim(-12, 12)
        a.set_ylim(-12, 12)
        a.set_aspect("equal")
    for a in axs[1]:
        a.set_xlabel("整体偏移 x (px)")
    for a in axs[:, 0]:
        a.set_ylabel("整体偏移 y (px)")
    fig.tight_layout()
    fig.savefig(OUT / "offset_xy.png", dpi=120)
    plt.close(fig)

    # 图：误差中整体偏移与剩余部分
    fig, a = plt.subplots(figsize=(9, 4))
    x = np.arange(6)
    o_ = [summary["describe"][k]["median_offset"] for k in keys]
    r_ = [summary["describe"][k]["median_rem"] for k in keys]
    a.bar(x - 0.2, o_, 0.4, label="整体偏移的大小")
    a.bar(x + 0.2, r_, 0.4, label="剩余部分的大小（各点到整体偏移的平均距离）")
    for i, k in enumerate(keys):
        a.text(i - 0.2, o_[i] + 0.05, f"{summary['describe'][k]['share_sq_offset']:.0%}", ha="center", fontsize=9)
    a.set_xticks(x, names, rotation=15, fontsize=9)
    a.set_ylabel("各对的中位数 (px)")
    a.legend(fontsize=9, loc="upper right")
    a.set_ylim(0, max(r_) * 1.35)
    a.set_title("百分数：整体偏移占均方误差的比例（全部对合计）", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "offset_vs_rest.png", dpi=120)
    plt.close(fig)

    # 图：模型之间 x 向整体偏移的相关（右上）与同号率（左下）；仿射与局部对应各一张
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for a, key, title in ((axs[0], "x", "仿射"), (axs[1], "local_x", "局部对应")):
        C = np.array(summary["agree"][key]["pearson"])
        S = np.array(summary["agree"][key]["same_sign_gt1px"], dtype=float)
        up = np.triu(np.ones((6, 6)), 1) > 0
        lo = np.tril(np.ones((6, 6)), -1) > 0
        M = np.where(up, C, np.where(lo, S, np.nan))
        a.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(6):
            for j in range(6):
                if i != j:
                    a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                           color="w" if M[i, j] < 0.6 else "k", fontsize=9)
        a.set_xticks(range(6), names, rotation=30, ha="right", fontsize=8)
        a.set_yticks(range(6), names, fontsize=8)
        a.set_title(f"{title}的 x 向整体偏移\n右上：相关系数；左下：同号率（两者都超过 1 px 的对）", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "offset_agree_x.png", dpi=120)
    plt.close(fig)

    # 图：局部对应的整体偏移 vs 仿射的整体偏移（x）
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    lim = 10
    for a, k in zip(axs.flat, keys):
        B, L = srcB[k], srcL[k]
        a.scatter(B[:, 0], L[:, 0], s=4, alpha=0.4)
        a.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, label="局部对应完全跟随仿射（来源 2）")
        a.axhline(0, c="g", ls="--", lw=0.8, label="局部对应与标注一致（来源 1）")
        f = summary["source"][k][MAIN_LOCAL[k]]["follow_x"]
        a.set_title(f"{MODELS[k]['name']}（{LOCAL_NAME[MAIN_LOCAL[k]]}）\n"
                    f"斜率 {f['slope']:.2f}，95% 区间 [{f['ci95'][0]:.2f}, {f['ci95'][1]:.2f}]", fontsize=10)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    for a in axs[1]:
        a.set_xlabel("仿射的 x 向整体偏移 (px)")
    for a in axs[:, 0]:
        a.set_ylabel("局部对应的 x 向整体偏移 (px)")
    fig.tight_layout()
    fig.savefig(OUT / "local_vs_affine_x.png", dpi=120)
    plt.close(fig)

    # 图：标注点处 x 向误差（中位数），按该对仿射整体偏移大小分四档
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for a, k in zip(axs.flat, keys):
        rows = usable(PP[k])
        nb = np.array([abs(r["b_aff_sub"][0]) for r in rows])
        edges = np.percentile(nb, [0, 25, 50, 75, 100])
        bars = {"仿射": [], "局部对应": [], "标注自拟合仿射（样本内）": []}
        lab = []
        for q in range(4):
            sel = [r for r, v in zip(rows, nb) if edges[q] <= v <= edges[q + 1]]
            bars["仿射"].append(np.median(np.abs(np.concatenate([r["d"][:, 0] for r in sel]))))
            bars["局部对应"].append(np.median(np.abs(np.concatenate([r["e"][:, 0] for r in sel]))))
            bars["标注自拟合仿射（样本内）"].append(np.nanmedian(np.abs(np.concatenate([r["g_in"][:, 0] for r in sel]))))
            lab.append(f"{edges[q]:.1f}–{edges[q + 1]:.1f}")
        x = np.arange(4)
        for i, (t, v) in enumerate(bars.items()):
            a.bar(x + (i - 1) * 0.27, v, 0.27, label=t)
        a.set_xticks(x, lab, fontsize=8)
        a.set_title(MODELS[k]["name"], fontsize=10)
    axs[0, 0].legend(fontsize=8)
    for a in axs[1]:
        a.set_xlabel("该对仿射 x 向整体偏移的绝对值，按四分位分档 (px)")
    for a in axs[:, 0]:
        a.set_ylabel("标注点处 x 向误差绝对值的中位数 (px)")
    fig.tight_layout()
    fig.savefig(OUT / "point_err_x_by_offset.png", dpi=120)
    plt.close(fig)

    # 示意对：6 个模型中位数的「仿射整体偏移 − 局部整体偏移」（大小，px），取最高、中位、最低
    score = {}
    for p in pairs:
        v = []
        for k in keys:
            r = PP[k][p]
            if r is None or "b_loc" not in r or np.abs(r["b_aff_sub"]).max() >= SANE or r["n_loc"] < 4:
                break
            v.append(np.linalg.norm(r["b_aff_sub"]) - np.linalg.norm(r["b_loc"]))
        if len(v) == 6:
            score[p] = float(np.median(v))
    order = sorted(score, key=score.get)
    pick = {"high": order[-1], "median": order[len(order) // 2], "low": order[0]}
    summary["examples"] = {t: {"pair": p, "score": score[p]} for t, p in pick.items()}
    summary["examples_n_candidates"] = len(score)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {"describe": summary["describe"][k], "source": summary["source"][k][MAIN_LOCAL[k]]}
                      for k in keys}, ensure_ascii=False, indent=1))
    print(json.dumps(summary["examples"], ensure_ascii=False))


def sar_cache(args):
    """服务器上：把示意对的 SAR（主表映射 p2p98）存成 8 位 png，供本地画图。"""
    from PIL import Image
    from moonlib import inputs

    img = Data(args.data)
    ex = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["examples"]
    for t, e in ex.items():
        sar = inputs.sar_p2p98(img.sar(SPLIT, e["pair"]))
        Image.fromarray((sar * 255).astype(np.uint8)).save(DATA / f"sar_{e['pair'].replace('/', '__')}.png")


def examples(args):
    from PIL import Image
    plt = setup_plt()
    ex = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["examples"]
    keys = list(MODELS)
    pts = {k: read_points(k) for k in keys}
    for tag, e in ex.items():
        pair = e["pair"]
        sar = np.asarray(Image.open(DATA / f"sar_{pair.replace('/', '__')}.png"))
        # 误差向量放大倍数：最多 10 倍，最长的箭头不超过 150 px
        big = max(np.nanmax(np.linalg.norm(pts[k][pair][:, 4:6] - pts[k][pair][:, 2:4], axis=1)) for k in keys)
        G = float(max(1, min(10, int(150 / big))))
        fig, axs = plt.subplots(2, 3, figsize=(13, 9.2))
        for a, k in zip(axs.flat, keys):
            v = pts[k][pair]
            s, aff, loc = v[:, 2:4], v[:, 4:6], v[:, LOCAL_COL[MAIN_LOCAL[k]]]
            a.imshow(sar, cmap="gray", extent=(0, sar.shape[1], sar.shape[0], 0))
            for p1, c in ((aff, "r"), (loc, "c")):
                for i in range(len(s)):
                    if np.isfinite(p1[i]).all() and (c == "r" or np.linalg.norm(p1[i] - s[i]) < SANE):   # 误匹配只滤局部对应
                        a.annotate("", xy=s[i] + G * (p1[i] - s[i]), xytext=s[i],
                                   arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
            a.scatter(s[:, 0], s[:, 1], s=16, c="lime", zorder=3)
            a.set_xlim(0, sar.shape[1])
            a.set_ylim(sar.shape[0], 0)
            a.set_title(f"{MODELS[k]['name']}（局部：{LOCAL_NAME[MAIN_LOCAL[k]]}）", fontsize=10)
            a.axis("off")
        axs[0, 0].scatter([], [], s=16, c="lime", label="标注点")
        axs[0, 0].plot([], [], "r", label=f"仿射的误差（放大 {G:.0f} 倍）")
        axs[0, 0].plot([], [], "c", label=f"局部对应的误差（放大 {G:.0f} 倍）")
        axs[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle(f"SAR 影像；仿射整体偏移比局部对应整体偏移大（6 个模型的中位数） {e['score']:.2f} px", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"example_{tag}.png", dpi=100)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "analyse", "sar-cache", "examples"])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA", str(YGC / "dataset/Moon")))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="只跑前若干对（试跑）")
    ap.add_argument("--selfcheck", type=int, default=30, help="RoMa：前多少对用稠密采样 + RANSAC 复现已存仿射")
    args = ap.parse_args()
    if args.cmd == "collect" and os.environ.get("GPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
    {"collect": collect, "analyse": analyse, "sar-cache": sar_cache, "examples": examples}[args.cmd](args)


if __name__ == "__main__":
    main()
