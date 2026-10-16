"""E9：SAR 里残留的地形视差（#147）。

DEM（SLDEM2015，512 ppd ≈ 59 m）按各 patch 的地理参考重采样到 patch 的 7.5 m 网格，预测 SAR 里沿 x 的视差
P = (h − 基准) · cot(入射角) / 东西向像元的地面宽度（px，不带符号；符号由视方位角决定，留给数据判断），
与标注点处的仿射残差、局部对应残差，以及 LoFTR zero-shot 全部点对的残差对照。

  python runs/E9/code/terrain_parallax.py dem                 # 各 ROI 的 DEM 窗口 → raw/（只需一次，走网络）
  python runs/E9/code/terrain_parallax.py points [--main D:/Code/moon-exp]   # 逐点预测与实测 → extra/data/、raw/
  python runs/E9/code/terrain_parallax.py analyse             # 统计与图 → extra/

环境：GDAL + numpy + matplotlib（本地 conda 环境 moon-dataset）。数据集根目录取环境变量 MOON_DATA。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "10")
gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "5")

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "runs" / "E9"
OUT = EXP / "extra"
DATA_OUT = OUT / "data"
RAW = EXP / "raw"               # DEM 窗口与稠密点对的逐点数据（不进 git）
E3_DATA = REPO / "runs" / "E3" / "extra" / "data"
DATA = Path(os.environ.get("MOON_DATA", "G:/Lunar_Optical_SAR_Registration_Dataset")) / "Processed_Data"

SLDEM = ("/vsicurl/https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/LolaKaguya_Topo/"
         "Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif")   # Int16，×0.5 为米，nodata −32768
DEM_SCALE, DEM_NODATA = 0.5, -32768
R_MOON = 1737400.0
MARGIN = 3000.0                 # DEM 窗口在 ROI 外多取的米数
ORTHO_CELL = (560.0, 200.0)     # Mini-RF Ortho 处理的 DEM 网格（m），方向未知，两种朝向都算

# 入射角、视方位角取自 moon-dataset 的 data/metadata/minirf_metadata.csv（PDS 标签）
ROIS = {
    "ROI_035": ("val", 50.97, 88.37), "ROI_049": ("val", 52.47, 269.29), "ROI_070": ("val", 52.72, 269.73),
    "ROI_071": ("val", 52.57, 268.71), "ROI_075": ("val", 51.79, 87.74), "ROI_080": ("val", 53.71, 267.31),
    "ROI_007": ("test", 50.66, 92.54), "ROI_020": ("test", 49.39, 271.86),
}
SPLIT_DIRS = {"val": "Val", "test": "Test"}

# 与 E3 相同的 6 个模型（预测进 git；LoFTR zero-shot 的点对 npz 不进 git，在 --main 指向的 checkout 里）
MODELS = {
    "loftr_zs": ("LoFTR zero-shot", "runs/B0/preds/anymatch_loftr"),
    "loftr_q4": ("LoFTR 无标注训练", "runs/Q/preds/Q4"),
    "loftr_e1": ("LoFTR 标注过拟合", "runs/E1/preds/loftr_lr5e5"),
    "roma_zs": ("RoMa zero-shot", "runs/B0m/preds/anymatch_roma__minmax"),
    "roma_m4": ("RoMa 无标注训练", "runs/M/preds/M4"),
    "roma_e1": ("RoMa 标注过拟合", "runs/E1/preds/roma_vgg"),
}
DENSE_MODEL = "loftr_zs"
R_NB, MIN_NB = 12.0, 3          # 局部对应：与 E3 相同
SANE = 20.0                     # 与 E3 相同：整体偏移超过它的 pair 视为粗错
DENSE_KEEP = float(os.environ.get("E9_DENSE_KEEP", 15.0))  # 稠密点对：相对仿射的残差小于它才算（视差可达十几 px）

FIELDS = ("mean", "plane", "hpA", "hpB")   # 高程基准：patch 均值 / 去平面 / 去 Ortho 网格平滑（两种朝向）后再去平面


def trad(srs):
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def srs_of(wkt):
    s = osr.SpatialReference()
    s.ImportFromWkt(wkt)
    return trad(s)


def patches(split, roi):
    d = DATA / SPLIT_DIRS[split] / roi / "Optical"
    return sorted((p.stem for p in d.glob("patch_*.tif")), key=lambda s: int(s.split("_")[1]))


def opt_path(split, roi, stem):
    return DATA / SPLIT_DIRS[split] / roi / "Optical" / f"{stem}.tif"


def labels(split, roi, stem):
    p = DATA / SPLIT_DIRS[split] / roi / "Label" / f"{stem}.txt"
    if not p.exists():
        return None
    a = np.loadtxt(p, dtype=np.float64, ndmin=2)
    a[:, [1, 3]] *= -1
    return a[:, :2], a[:, 2:4]


# ---------------------------------------------------------------- dem

def cmd_dem(args):
    RAW.mkdir(parents=True, exist_ok=True)
    src = gdal.Open(SLDEM)
    dem_srs = trad(osr.SpatialReference(wkt=src.GetProjection()))
    for roi, (split, *_rest) in ROIS.items():
        if args.rois and roi not in args.rois:
            continue
        dst = RAW / f"dem_{roi}.tif"
        if dst.exists() and not args.force:
            print(roi, "已有", dst.name)
            continue
        xs, ys = [], []
        for stem in patches(split, roi):
            ds = gdal.Open(str(opt_path(split, roi, stem)))
            gt, w, h = ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize
            ct = osr.CoordinateTransformation(srs_of(ds.GetProjection()), dem_srs)
            for u, v in ((0, 0), (w, 0), (0, h), (w, h), (w / 2, 0), (w / 2, h), (0, h / 2), (w, h / 2)):
                x, y, _ = ct.TransformPoint(gt[0] + u * gt[1], gt[3] + v * gt[5])
                xs.append(x)
                ys.append(y)
        win = [min(xs) - MARGIN, max(ys) + MARGIN, max(xs) + MARGIN, min(ys) - MARGIN]
        tmp = dst.with_suffix(".part.tif")
        for attempt in range(5):        # 网络读到半截时整块重来
            try:
                gdal.Translate(str(tmp), src, projWin=win, creationOptions=["COMPRESS=DEFLATE"])
                break
            except RuntimeError as e:
                print(roi, "第", attempt + 1, "次失败：", str(e).splitlines()[0], flush=True)
        else:
            raise SystemExit(f"{roi} 下载失败")
        tmp.replace(dst)
        print(roi, "窗口 km", round((win[2] - win[0]) / 1e3, 1), "×", round((win[1] - win[3]) / 1e3, 1), flush=True)


# ---------------------------------------------------------------- points

def box(a, rx, ry):
    """可分离的方框平均，窗口 (2rx+1)×(2ry+1) px，边缘按镜像延拓。"""
    def run(a, r, axis):
        if r < 1:
            return a
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="reflect"), axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
        return (hi - lo) / (2 * r + 1)
    return run(run(a, rx, 1), ry, 0)


def deplane(f):
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w]
    X = np.c_[np.ones(f.size), xx.ravel(), yy.ravel()]
    c, *_ = np.linalg.lstsq(X, f.ravel(), rcond=None)
    return f - (X @ c).reshape(f.shape)


def sample(f, p):
    """双线性取值；p 为角点原点坐标（像元 i 覆盖 [i, i+1)）。"""
    h, w = f.shape
    x = np.clip(p[:, 0] - 0.5, 0, w - 1.001)
    y = np.clip(p[:, 1] - 0.5, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy)
            + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)


def patch_fields(dem, split, roi, stem, inc):
    """patch 网格上的 4 种预测视差场（px，不带符号）与东西 / 南北向像元的地面宽度（m）。"""
    ds = gdal.Open(str(opt_path(split, roi, stem)))
    gt, w, h, wkt = ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize, ds.GetProjection()
    bounds = (gt[0], gt[3] + h * gt[5], gt[0] + w * gt[1], gt[3])
    wd = gdal.Warp("", dem, format="MEM", dstSRS=wkt, outputBounds=bounds, width=w, height=h,
                   resampleAlg="cubicspline", srcNodata=DEM_NODATA, dstNodata=np.nan, outputType=gdal.GDT_Float32)
    z = wd.GetRasterBand(1).ReadAsArray().astype(np.float64) * DEM_SCALE
    if not np.isfinite(z).all():
        return None
    s = srs_of(wkt)
    ct = osr.CoordinateTransformation(s, trad(s.CloneGeogCS()))
    xc, yc = gt[0] + w / 2 * gt[1], gt[3] + h / 2 * gt[5]
    lon0, lat0, _ = ct.TransformPoint(xc, yc)
    lon1, _, _ = ct.TransformPoint(xc + gt[1], yc)
    _, lat2, _ = ct.TransformPoint(xc, yc + gt[5])
    gx = abs(math.radians(lon1 - lon0)) * R_MOON * math.cos(math.radians(lat0))
    gy = abs(math.radians(lat2 - lat0)) * R_MOON
    k = 1.0 / math.tan(math.radians(inc)) / gx          # 每米高程对应的 x 向视差（px）
    F = {"mean": (z - z.mean()) * k, "plane": deplane(z) * k}
    for name, (cx, cy) in (("hpA", ORTHO_CELL), ("hpB", ORTHO_CELL[::-1])):
        F[name] = deplane(z - box(z, int(round(cx / gx / 2)), int(round(cy / gy / 2)))) * k
    return F, gx, gy, float(z.std()), float(deplane(z).std())


def load_preds(rel, split):
    out = {}
    p = REPO / rel / f"{split}.jsonl"
    if not p.exists():          # RoMa zero-shot（B0m）只有 Val
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["pair"]] = None if r.get("A") is None else np.array(r["A"], dtype=np.float64)
    return out


def apply(A, p):
    return p @ A[:, :2].T + A[:, 2]


def local_nb(M, o):
    """与 E3 相同的局部对应：半径 R_NB 内点对位移的中位数。M 已是角点原点约定。"""
    out = np.full((len(o), 2), np.nan)
    if len(M) == 0:
        return out
    src, disp = M[:, :2], M[:, 2:4] - M[:, :2]
    for i, p in enumerate(o):
        nb = np.linalg.norm(src - p, axis=1) <= R_NB
        if nb.sum() >= MIN_NB:
            out[i] = p + np.median(disp[nb], axis=0)
    return out


def e3_local():
    """E3 的逐点局部对应（只有 Val）：(模型, pair, 序号) → (lx, ly)。"""
    out = {}
    for m in MODELS:
        with open(E3_DATA / f"points_{m}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[m, r["pair"], int(r["k"])] = (float(r["lx"]), float(r["ly"]))
    return out


def cmd_points(args):
    main = Path(args.main)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    loc3 = e3_local()
    preds = {(m, s): load_preds(rel, s) for m, (_, rel) in MODELS.items() for s in ("val", "test")}
    mats = {s: np.load(main / MODELS[DENSE_MODEL][1] / f"{s}_matches.npz") for s in ("val", "test")}
    rows, dense, meta = [], {k: [] for k in ("roi", "pair", "x", "y", "rx", "ry", *FIELDS)}, {}
    for roi, (split, inc, az) in ROIS.items():
        if args.rois and roi not in args.rois:
            continue
        dem = gdal.Open(str(RAW / f"dem_{roi}.tif"))
        n_pair = 0
        for stem in patches(split, roi):
            pair = f"{roi}/{stem}"
            A0 = preds[DENSE_MODEL, split].get(pair)
            lab = labels(split, roi, stem)
            if lab is None and A0 is None:
                continue
            got = patch_fields(dem, split, roi, stem, inc)
            if got is None:
                print("DEM 有空洞，跳过", pair)
                continue
            F, gx, gy, zsd, zsd_plane = got
            meta[pair] = {"gx": gx, "gy": gy, "z_std": zsd, "z_std_plane": zsd_plane}
            n_pair += 1
            key = pair.replace("/", "__")
            M = mats[split][key].astype(np.float64) if key in mats[split].files else np.zeros((0, 5))
            if A0 is not None and len(M):
                r = M[:, 2:4] - apply(A0, M[:, :2])
                keep = np.abs(r).max(1) < DENSE_KEEP
                for name, v in (("x", M[keep, 0]), ("y", M[keep, 1]), ("rx", r[keep, 0]), ("ry", r[keep, 1])):
                    dense[name].append(v)
                for f in FIELDS:
                    dense[f].append(sample(F[f], M[keep, :2]))
                dense["roi"].append(np.full(keep.sum(), list(ROIS).index(roi)))
                dense["pair"].append(np.full(keep.sum(), len(meta) - 1))
            if lab is None:
                continue
            o, s = lab
            P = {f: sample(F[f], o) for f in FIELDS}
            loc_zs = local_nb(M, o)
            for i in range(len(o)):
                row = {"split": split, "roi": roi, "pair": pair, "k": i, "inc": inc, "az": az,
                       "ox": o[i, 0], "oy": o[i, 1], "sx": s[i, 0], "sy": s[i, 1],
                       **{f"P_{f}": P[f][i] for f in FIELDS}}
                for m in MODELS:
                    A = preds[m, split].get(pair)
                    a = apply(A, o[i:i + 1])[0] if A is not None else (np.nan, np.nan)
                    row[f"ax_{m}"], row[f"ay_{m}"] = a
                    lx, ly = loc3.get((m, pair, i), (np.nan, np.nan))
                    if m == DENSE_MODEL and split == "test":
                        lx, ly = loc_zs[i]
                    row[f"lx_{m}"], row[f"ly_{m}"] = lx, ly
                row["lx_check"] = loc_zs[i, 0]   # 本脚本重算的 LoFTR zero-shot 局部对应，与 E3 核对
                rows.append(row)
        print(roi, split, "pair", n_pair, flush=True)
    with open(DATA_OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(RAW / "dense.npz", pairs=np.array(list(meta)),
                        **{k: np.concatenate(v) for k, v in dense.items()})
    (DATA_OUT / "patch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
    print("标注点", len(rows), "稠密点", sum(len(v) for v in dense["x"]))


# ---------------------------------------------------------------- analyse

GROUPS = {"E": "视方位角约 90°", "W": "视方位角约 270°"}
N_BOOT = 1000


def group_of(az):
    return "E" if az < 180 else "W"


def read_labels():
    with open(DATA_OUT / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def detrend(ids, X, *vals):
    """在每个 pair 内把各量投影到 [1, x, y] 的正交补上（去掉仿射能吸收的部分）。点数 < 4 的 pair 置 nan。"""
    out = [np.full(len(v), np.nan) for v in vals]
    for p in np.unique(ids):
        i = np.flatnonzero(ids == p)
        if len(i) < 4:
            continue
        D = np.c_[np.ones(len(i)), X[i]]
        Q, _ = np.linalg.qr(D)
        for o, v in zip(out, vals):
            o[i] = v[i] - Q @ (Q.T @ v[i])
    return out


def fit(ids, m, P, rng):
    """过原点回归 m ≈ s·P：斜率 s、r²（以 Σm² 为分母，即 s·P 去掉的均方比例）、按 pair 自助的 95% 区间。"""
    ok = np.isfinite(m) & np.isfinite(P)
    ids, m, P = ids[ok], m[ok], P[ok]
    if len(m) < 10:
        return None
    s = float((m * P).sum() / (P * P).sum())
    r2 = 1 - float(((m - s * P) ** 2).sum() / (m * m).sum())
    up, inv = np.unique(ids, return_inverse=True)
    mp = np.bincount(inv, m * P, len(up))
    pp = np.bincount(inv, P * P, len(up))
    bs = []
    for _ in range(N_BOOT):
        c = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))
        bs.append((c * mp).sum() / (c * pp).sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"slope": round(s, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "r2": round(r2, 3),
            "n_pts": int(len(m)), "n_pairs": int(len(up))}


def fit2(ids, m, P_lo, P_hi, rng):
    """m ≈ a·P_lo + b·P_hi：低频（Ortho 网格尺度以上）与高频视差各自的斜率，按 pair 自助的 95% 区间。
    若 Ortho 用真实（粗网格）DEM 校正过，低频部分已被去掉，a 应接近 0。"""
    ok = np.isfinite(m) & np.isfinite(P_lo) & np.isfinite(P_hi)
    ids, m, Z = ids[ok], m[ok], np.c_[P_lo[ok], P_hi[ok]]
    up, inv = np.unique(ids, return_inverse=True)
    ZZ = np.stack([np.bincount(inv, Z[:, i] * Z[:, j], len(up)) for i in range(2) for j in range(2)], 1)
    Zm = np.stack([np.bincount(inv, Z[:, i] * m, len(up)) for i in range(2)], 1)
    sol = lambda c: np.linalg.solve((c[:, None] * ZZ).sum(0).reshape(2, 2), (c[:, None] * Zm).sum(0))
    ab = sol(np.ones(len(up)))
    bs = np.array([sol(np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)).astype(float))
                   for _ in range(N_BOOT)])
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    return {"slope_low": round(float(ab[0]), 3), "ci95_low": [round(float(v), 3) for v in ci[:, 0]],
            "slope_high": round(float(ab[1]), 3), "ci95_high": [round(float(v), 3) for v in ci[:, 1]],
            "n_pts": int(len(m)), "n_pairs": int(len(up))}


def sane_pairs(L, m):
    """与 E3 相同：仿射在该 pair 标注点上的平均偏移超过 SANE 的视为粗错。"""
    ok = np.zeros(len(L["pair"]), bool)
    ex, ey = L["sx"] - L[f"ax_{m}"], L["sy"] - L[f"ay_{m}"]
    for p in np.unique(L["pair"]):
        i = L["pair"] == p
        b = np.array([np.nanmean(ex[i]), np.nanmean(ey[i])])
        ok[i] = np.isfinite(b).all() and np.abs(b).max() < SANE
    return ok


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def cmd_analyse(args):
    rng = np.random.default_rng(0)
    L = read_labels()
    L["grp"] = np.array([group_of(a) for a in L["az"]])
    X = np.c_[L["ox"], L["oy"]]
    meta = json.loads((DATA_OUT / "patch_meta.json").read_text(encoding="utf-8"))
    S = {"note": "斜率：实测残差（SAR 标注 − 仿射映射点，或 − 局部对应点）对预测视差 P（px，不带符号）的过原点回归；"
                 "r2 为 s·P 去掉的均方比例；detrended 为 pair 内去掉 [1,x,y] 平面后"}

    # 0. 视差有多大：各 ROI 标注点处预测视差（去平面）的均方根与 DEM 起伏
    S["magnitude"] = {}
    for roi in ROIS:
        i = L["roi"] == roi
        prs = [p for p in meta if p.startswith(roi)]
        if not i.any():
            continue
        S["magnitude"][roi] = {
            "group": group_of(ROIS[roi][2]), "n_pts": int(i.sum()),
            "rms_P_mean_px": round(float(np.sqrt(np.mean(L["P_mean"][i] ** 2))), 2),
            "rms_P_plane_px": round(float(np.sqrt(np.mean(L["P_plane"][i] ** 2))), 2),
            "p95_abs_P_plane_px": round(float(np.percentile(np.abs(L["P_plane"][i]), 95)), 2),
            "median_z_std_plane_m": round(float(np.median([meta[p]["z_std_plane"] for p in prs])), 1),
            "median_gx_m": round(float(np.median([meta[p]["gx"] for p in prs])), 2)}

    # 1. 标注点：各模型、各组、各口径
    S["labels"] = {}
    for m in MODELS:
        ok = sane_pairs(L, m)
        res = {}
        for split in ("val", "test"):
            for g in GROUPS:
                sel = ok & (L["split"] == split) & (L["grp"] == g)
                if not sel.any():
                    continue
                ids, Xs = L["pair"][sel], X[sel]
                d = {}
                for name, mx in (("aff_x", L["sx"] - L[f"ax_{m}"]), ("aff_y", L["sy"] - L[f"ay_{m}"]),
                                 ("loc_x", L["sx"] - L[f"lx_{m}"]), ("loc_y", L["sy"] - L[f"ly_{m}"])):
                    mx = mx[sel]
                    d[name + "_raw"] = fit(ids, mx - np.nanmean(mx), L["P_mean"][sel], rng)   # 票面口径
                    mt, Pt, Pa, Pb = detrend(ids, Xs, mx, L["P_mean"][sel], L["P_hpA"][sel], L["P_hpB"][sel])
                    d[name + "_detr"] = fit(ids, mt, Pt, rng)
                    if name == "aff_x":
                        for f, hi in (("hpA", Pa), ("hpB", Pb)):
                            d[f"aff_x_low_high_{f}"] = fit2(ids, mt, Pt - hi, hi, rng)
                # x 向均方误差的正交分解（只用能去趋势的点）：仿射残差 = 仿射能表达的部分 + 去平面部分，
                # 去平面部分与模型无关（= 标注自拟合仿射的残差），视差解释它的 r2；
                # 由此得视差 ŝ·P⊥ 占总 x 均方误差的比例，以及物理斜率 ±1 时的比例
                ex = (L["sx"] - L[f"ax_{m}"])[sel]
                et, Pt = detrend(ids, Xs, ex, L["P_mean"][sel])
                f_ = np.isfinite(et) & np.isfinite(Pt) & np.isfinite(ex)
                det = d["aff_x_detr"]
                if det and f_.any():
                    sh = det["slope"]
                    tot, nonaff = float(np.mean(ex[f_] ** 2)), float(np.mean(et[f_] ** 2))
                    d["mse_x"] = round(tot, 3)
                    d["mse_x_nonaffine"] = round(nonaff, 3)
                    d["share_parallax_fit"] = round(float(np.mean((sh * Pt[f_]) ** 2)) / tot, 3)
                    d["share_nonaffine_removed_phys"] = round(
                        1 - float(np.mean((et[f_] - np.sign(sh) * Pt[f_]) ** 2)) / nonaff, 3)
                res[f"{split}_{g}"] = d
        # 逐 ROI 的去趋势斜率（符号检验）
        res["per_roi"] = {}
        for roi in ROIS:
            sel = ok & (L["roi"] == roi)
            mt, Pt = detrend(L["pair"][sel], X[sel], (L["sx"] - L[f"ax_{m}"])[sel], L["P_mean"][sel])
            res["per_roi"][roi] = fit(L["pair"][sel], mt, Pt, rng)
        S["labels"][m] = res
    chk = np.isfinite(L["lx_check"]) & np.isfinite(L[f"lx_{DENSE_MODEL}"]) & (L["split"] == "val")
    S["local_check_max_abs_diff_px"] = round(float(np.abs(L["lx_check"] - L[f"lx_{DENSE_MODEL}"])[chk].max()), 4)

    # 2. 稠密点对（LoFTR zero-shot）：完整高程 vs 去 Ortho 网格平滑后的高程
    D = dict(np.load(RAW / "dense.npz"))
    roi_names = list(ROIS)
    D["grp"] = np.array([group_of(ROIS[roi_names[r]][2]) for r in D["roi"]])
    D["split"] = np.array([ROIS[roi_names[r]][0] for r in D["roi"]])
    XD = np.c_[D["x"], D["y"]]
    S["dense"] = {}
    det_cache = {}
    for split in ("val", "test"):
        for g in GROUPS:
            sel = (D["split"] == split) & (D["grp"] == g)
            if not sel.any():
                continue
            ids = D["pair"][sel]
            d = {}
            for f in ("plane", "hpA", "hpB"):
                rx, ry, Pf = detrend(ids, XD[sel], D["rx"][sel], D["ry"][sel], D[f][sel])
                d[f"x_{f}"] = fit(ids, rx, Pf, rng)
                d[f"y_{f}"] = fit(ids, ry, Pf, rng)
                det_cache[split, g, f] = (ids, rx, ry, Pf, D["roi"][sel])
            rx = det_cache[split, g, "plane"][1]
            for f in ("hpA", "hpB"):
                hi = det_cache[split, g, f][3]
                d[f"x_low_high_{f}"] = fit2(ids, rx, det_cache[split, g, "plane"][3] - hi, hi, rng)
            S["dense"][f"{split}_{g}"] = d
    S["dense_per_roi"] = {}
    for r, roi in enumerate(roi_names):
        sel = D["roi"] == r
        rx, Pf = detrend(D["pair"][sel], XD[sel], D["rx"][sel], D["plane"][sel])
        S["dense_per_roi"][roi] = fit(D["pair"][sel], rx, Pf, rng)
    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")

    plt = setup_plt()
    fig_dense(plt, det_cache)
    fig_labels(plt, S)
    fig_sign(plt, S)
    fig_examples(plt, D, meta, roi_names)
    print(json.dumps({k: S[k] for k in ("magnitude", "dense", "dense_per_roi")}, ensure_ascii=False, indent=1))


def fig_dense(plt, det_cache):
    bins = np.linspace(-4, 4, 17)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for a, g in zip(axs, GROUPS):
        for split, ls in (("val", "-"), ("test", "--")):
            if (split, g, "plane") not in det_cache:
                continue
            ids, rx, ry, Pf, rois = det_cache[split, g, "plane"]
            for r in np.unique(rois):
                i = (rois == r) & np.isfinite(rx)
                c = (bins[:-1] + bins[1:]) / 2
                k = np.digitize(Pf[i], bins) - 1
                mu = [np.mean(rx[i][k == b]) if (k == b).sum() >= 30 else np.nan for b in range(len(c))]
                a.plot(c, mu, ls, marker="o", ms=3, label=f"{list(ROIS)[r]}（{'Val' if split == 'val' else 'Test'}）")
            i = np.isfinite(ry)
            k = np.digitize(Pf[i], bins) - 1
            muy = [np.mean(ry[i][k == b]) if (k == b).sum() >= 30 else np.nan for b in range(len(c))]
            a.plot(c, muy, ls, c="0.6", lw=1, label=f"y 向残差（{'Val' if split == 'val' else 'Test'}）")
        a.plot(bins, bins, "k:", lw=0.8, label="斜率 +1")
        a.plot(bins, -bins, "k-.", lw=0.8, label="斜率 −1")
        a.axhline(0, c="k", lw=0.5)
        a.set_ylim(-4, 4)
        a.set_title(GROUPS[g])
        a.set_xlabel("预测视差 P（px，不带符号，pair 内去平面）")
        a.legend(fontsize=7, ncol=2)
    axs[0].set_ylabel("点对 x 向残差（px，pair 内去平面）分箱均值")
    fig.tight_layout()
    fig.savefig(OUT / "dense_binned.png", dpi=130)
    plt.close(fig)


LOOK = {"ROI_007": "右视", "ROI_020": "右视"}   # 其余 6 个 Val ROI 为左视（minirf_metadata.csv）


def fig_sign(plt, S):
    """逐 ROI 的斜率：标注点（去平面后与模型无关，取 LoFTR zero-shot 的样本）与点对。"""
    rois = sorted(ROIS, key=lambda r: (group_of(ROIS[r][2]), ROIS[r][0] == "test", r))
    fig, a = plt.subplots(figsize=(10, 4.2))
    for j, (vals, lab, c) in enumerate(((S["labels"][DENSE_MODEL]["per_roi"], "SAR 标注点（去平面）", "C0"),
                                        (S["dense_per_roi"], "LoFTR zero-shot 点对（去平面）", "C1"))):
        v = [vals[r]["slope"] for r in rois]
        e = [[vals[r]["slope"] - vals[r]["ci95"][0], vals[r]["ci95"][1] - vals[r]["slope"]] for r in rois]
        a.bar(np.arange(len(rois)) + (j - 0.5) * 0.35, v, 0.35, yerr=np.array(e).T, capsize=2, color=c, label=lab)
    a.axhline(0, c="k", lw=0.6)
    a.set_xticks(range(len(rois)))
    a.set_xticklabels([f"{r}\n{'Val' if ROIS[r][0] == 'val' else 'Test'}·{LOOK.get(r, '左视')}\n视方位角 {ROIS[r][2]:.0f}°"
                       for r in rois], fontsize=8)
    a.set_ylabel("x 残差对预测视差的斜率（95% 区间）")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sign_by_roi.png", dpi=130)
    plt.close(fig)


def fig_labels(plt, S):
    keys = list(MODELS)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for a, g in zip(axs, GROUPS):
        w = 0.2
        for j, (name, lab) in enumerate((("aff_x_detr", "仿射 x 残差"), ("loc_x_detr", "局部对应 x 残差"),
                                         ("aff_y_detr", "仿射 y 残差"))):
            v, e = [], []
            for m in keys:
                f = S["labels"][m].get(f"val_{g}", {}).get(name)
                v.append(f["slope"] if f else np.nan)
                e.append([f["slope"] - f["ci95"][0], f["ci95"][1] - f["slope"]] if f else [0, 0])
            a.bar(np.arange(len(keys)) + (j - 1) * w, v, w, yerr=np.array(e).T, capsize=2, label=lab)
        a.axhline(0, c="k", lw=0.6)
        a.set_xticks(range(len(keys)))
        a.set_xticklabels([MODELS[m][0] for m in keys], rotation=25, ha="right", fontsize=8)
        a.set_title(f"Val，{GROUPS[g]}")
    axs[0].set_ylabel("对预测视差的斜率（pair 内去平面，95% 区间）")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "labels_slope.png", dpi=130)
    plt.close(fig)


def fig_examples(plt, D, meta, roi_names, n=3):
    """起伏最大的几个 patch：光学、SAR、带符号的预测视差场、点对 x 残差（均去平面）。"""
    pairs = D["pairs"]
    cnt = np.bincount(D["pair"].astype(int), minlength=len(pairs))
    cand, seen = [], set()
    for p in sorted(meta, key=lambda p: -meta[p]["z_std_plane"]):   # 点对够多、起伏最大，每个 ROI 至多一个
        roi = p.split("/")[0]
        if roi not in seen and cnt[int(np.flatnonzero(pairs == p)[0])] >= 300:
            cand.append(p)
            seen.add(roi)
        if len(cand) == n:
            break
    fig, axs = plt.subplots(n, 4, figsize=(15, 3.9 * n))
    for row, pair in zip(axs, cand):
        roi, stem = pair.split("/")
        split, inc, az = ROIS[roi]
        g = group_of(az)
        sg = float(np.sign(json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["dense"]
                           .get(f"{split}_{g}", {}).get("x_plane", {}).get("slope", 1.0)))
        dem = gdal.Open(str(RAW / f"dem_{roi}.tif"))
        F = patch_fields(dem, split, roi, stem, inc)[0]
        opt = gdal.Open(str(opt_path(split, roi, stem))).ReadAsArray()
        sds = gdal.Open(str(DATA / SPLIT_DIRS[split] / roi / "SAR" / f"{stem}.tif"))
        sar = sds.GetRasterBand(2).ReadAsArray()
        pi = int(np.flatnonzero(pairs == pair)[0])
        i = D["pair"] == pi
        rx, Pf = detrend(D["pair"][i], np.c_[D["x"][i], D["y"][i]], D["rx"][i], D["plane"][i])
        lim = max(1.0, float(np.percentile(np.abs(sg * F["plane"]), 98)))
        row[0].imshow(opt, cmap="gray")
        row[1].imshow(sar, cmap="gray", vmin=np.percentile(sar, 1), vmax=np.percentile(sar, 99))
        im = row[2].imshow(sg * F["plane"], cmap="RdBu_r", vmin=-lim, vmax=lim)
        row[3].imshow(opt, cmap="gray", alpha=0.5)
        sc = row[3].scatter(D["x"][i], D["y"][i], c=rx, s=4, cmap="RdBu_r", vmin=-lim, vmax=lim)
        r = np.corrcoef(rx, Pf)[0, 1] * sg
        row[0].set_title(f"{pair} 光学（DEM 起伏 {meta[pair]['z_std_plane']:.0f} m）", fontsize=9)
        row[1].set_title("SAR", fontsize=9)
        row[2].set_title("预测 x 视差（px，去平面）", fontsize=9)
        row[3].set_title(f"LoFTR zero-shot 点对 x 残差（去平面），r = {r:.2f}", fontsize=9)
        for a in row:
            a.axis("off")
        fig.colorbar(im, ax=row[2], fraction=0.046)
        fig.colorbar(sc, ax=row[3], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "relief_examples.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dem")
    p.add_argument("--force", action="store_true")
    p.add_argument("--rois", nargs="*", help="只下载这些 ROI（可分几个进程并行）")
    p = sub.add_parser("points")
    p.add_argument("--main", default=str(REPO), help="有不进 git 的点对 npz 的 checkout")
    p.add_argument("--rois", nargs="*", help="只处理这些 ROI（调试用）")
    sub.add_parser("analyse")
    a = ap.parse_args()
    {"dem": cmd_dem, "points": cmd_points, "analyse": cmd_analyse}[a.cmd](a)
