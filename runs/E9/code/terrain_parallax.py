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
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

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
DENSE_KEEP = 6.0                # 稠密点对：相对仿射的残差小于它才算（视差最多几 px）

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
        gdal.Translate(str(dst), src, projWin=win, creationOptions=["COMPRESS=DEFLATE"])
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
    with open(REPO / rel / f"{split}.jsonl", encoding="utf-8") as f:
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dem")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("points")
    p.add_argument("--main", default=str(REPO), help="有不进 git 的点对 npz 的 checkout")
    sub.add_parser("analyse")
    a = ap.parse_args()
    {"dem": cmd_dem, "points": cmd_points}.get(a.cmd, lambda _: sys.exit("analyse 尚未实现"))(a)
