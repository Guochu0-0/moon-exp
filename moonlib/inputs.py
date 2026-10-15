"""原始影像 → 方法输入的映射。

主表口径（「重跑 baseline」#3 的决定）：所有方法同一套映射，送 float32 [0,1] 单通道，不经 uint8 量化。
- 光学：uint8 / 255。
- SAR：S1 = b1 + b2 → dB → 下限 −25 dB → 逐 patch p2–p98 线性拉伸并截断到 [0,1]。
其余映射只用于 Val 上的消融。映射按名字登记，run 配置里写名字。

SAR 的 4 个波段是 Mini-RF CDR 原始波段：b1 = ⟨|E_H|²⟩，b2 = ⟨|E_V|²⟩，b3 = Re⟨E_H E_V*⟩，b4 = Im⟨E_H E_V*⟩
（moon-dataset docs/pages/data/imaging.md）。极化派生量按同一文档：S1 = b1 + b2，S2 = b1 − b2，S3 = 2 b3，S4 = −2 b4，
SC = (S1 − S4)/2，OC = (S1 + S4)/2，CPR = SC/OC，m = √(S2² + S3² + S4²)/S1。m-χ 分解（Raney et al. 2012,
doi:10.1029/2011JE003986）在此约定下（单次散射增强 OC，即 S4 > 0）：奇次 = (m S1 + S4)/2，偶次 = (m S1 − S4)/2，
体散射 = S1 (1 − m)。（SAR 信号差异的零样本检验 #150）
"""
from __future__ import annotations

import numpy as np

DB_FLOOR = -25.0


def sar_db(sar: np.ndarray) -> np.ndarray:
    """H×W×4 原始波段 → S1 的 dB 图，下限 DB_FLOOR；非有限值当作下限。"""
    s1 = sar[..., 0].astype(np.float64) + sar[..., 1].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(s1)
    db = np.where(np.isfinite(db), db, DB_FLOOR)
    return np.maximum(db, DB_FLOOR)


def _stretch(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-12), 0.0, 1.0)


def optical_div255(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def sar_p2p98(sar: np.ndarray) -> np.ndarray:
    return _stretch(sar_db(sar), 2.0, 98.0).astype(np.float32)


def sar_zscore_2p5(sar: np.ndarray) -> np.ndarray:
    """消融：逐 patch z-score，±2.5σ 线性映射到 [0,1] 并截断。"""
    db = sar_db(sar)
    z = (db - db.mean()) / max(db.std(), 1e-12)
    return np.clip((z + 2.5) / 5.0, 0.0, 1.0).astype(np.float32)


def sar_minmax(sar: np.ndarray) -> np.ndarray:
    """消融：不做分位截断，逐 patch min–max。"""
    return _stretch(sar_db(sar), 0.0, 100.0).astype(np.float32)


# ---- 极化派生量（#150）：功率量取 dB 后逐 patch p2–p98 拉伸，极化度 m 直接拉伸 ----

POW_FLOOR = 1e-4   # 功率下限（−40 dB），非正值都当作下限


def stokes(sar: np.ndarray):
    b = [np.nan_to_num(sar[..., i].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0) for i in range(4)]
    return b[0] + b[1], b[0] - b[1], 2.0 * b[2], -2.0 * b[3]


def _box(x: np.ndarray, k: int) -> np.ndarray:
    """k×k 均值（边缘反射延拓），沿前两维。"""
    r = k // 2
    x = np.pad(x, [(r, r), (r, r)] + [(0, 0)] * (x.ndim - 2), mode="reflect")
    for ax in (0, 1):
        c = np.cumsum(x, axis=ax, dtype=np.float64)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=ax)), c], axis=ax)
        n = x.shape[ax] - 2 * r
        x = (np.take(c, np.arange(k, k + n), axis=ax) - np.take(c, np.arange(n), axis=ax)) / k
    return x


def polar(sar: np.ndarray, looks: int = 1) -> dict:
    """H×W×4 原始波段 → 极化派生量（功率为线性值）。looks > 1 时先对 4 个波段做 looks×looks 均值（多视），再求派生量。"""
    if looks > 1:
        sar = _box(np.nan_to_num(sar.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0), looks)
    s1, s2, s3, s4 = stokes(sar)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.sqrt(s2 ** 2 + s3 ** 2 + s4 ** 2) / s1
    m = np.clip(np.where(np.isfinite(m), m, 0.0), 0.0, 1.0)
    sc, oc = 0.5 * (s1 - s4), 0.5 * (s1 + s4)
    return {"s1": s1, "sc": sc, "oc": oc, "cpr": np.maximum(sc, POW_FLOOR) / np.maximum(oc, POW_FLOOR), "m": m,
            "odd": 0.5 * (m * s1 + s4), "even": 0.5 * (m * s1 - s4), "vol": s1 * (1.0 - m)}


def _db(p: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, POW_FLOOR))


def _pol_db(key: str, looks: int = 1):
    def f(sar):
        return _stretch(_db(polar(sar, looks)[key]), 2.0, 98.0).astype(np.float32)
    f.__doc__ = f"消融（#150）：{key}（{looks}×{looks} 多视）取 dB，逐 patch p2–p98 拉伸。"
    return f


def _pol_m(looks: int = 1):
    def f(sar):
        return _stretch(polar(sar, looks)["m"], 2.0, 98.0).astype(np.float32)
    f.__doc__ = f"消融（#150）：极化度 m（{looks}×{looks} 多视），逐 patch p2–p98 拉伸。"
    return f


def sar_mchi_rgb(sar: np.ndarray) -> np.ndarray:
    """消融（#150）：m-χ 三通道 H×W×3（R 偶次、G 体散射、B 奇次），各自取 dB 后 p2–p98 拉伸。只有 RoMa 适配器吃三通道。"""
    p = polar(sar)
    return np.stack([_stretch(_db(p[k]), 2.0, 98.0) for k in ("even", "vol", "odd")], -1).astype(np.float32)


# ---- 光学模糊（#150）：模拟 SAR 较粗的分辨率。patch 像元 7.5 m，高斯核按 FWHM（px）给出 ----

def _gauss1d(img: np.ndarray, fwhm: float, axis: int) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    r = int(np.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    x = np.pad(img, pad, mode="reflect")
    n = img.shape[axis]
    return sum(w * np.take(x, np.arange(i, i + n), axis=axis) for i, w in enumerate(k))


def _optical_blur(fx: float, fy: float):
    def f(img):
        x = img.astype(np.float64) / 255.0
        if fx:
            x = _gauss1d(x, fx, 1)
        if fy:
            x = _gauss1d(x, fy, 0)
        return x.astype(np.float32)
    f.__doc__ = f"消融（#150）：/255 后 x 向 FWHM {fx} px、y 向 FWHM {fy} px 的高斯模糊。"
    return f


OPTICAL = {"div255": optical_div255,
           "xblur4": _optical_blur(4.0, 0.0),        # x（约为距离向）≈ 30 m
           "blur4x2": _optical_blur(4.0, 2.0)}       # x ≈ 30 m，y（约为方位向）≈ 15 m
SAR = {"p2p98": sar_p2p98, "zscore_2p5": sar_zscore_2p5, "minmax": sar_minmax,
       **{k: _pol_db(k) for k in ("sc", "oc", "cpr")},
       **{f"mchi_{k}": _pol_db(k) for k in ("odd", "even", "vol")},
       "m": _pol_m(), "mchi_rgb": sar_mchi_rgb,
       **{f"cpr_ml{k}": _pol_db("cpr", k) for k in (5, 9)},     # 逐像素的 CPR、m 几乎全是斑点，先多视再求比值
       **{f"m_ml{k}": _pol_m(k) for k in (5, 9)}}


def get(kind: str, name: str):
    table = {"optical": OPTICAL, "sar": SAR}[kind]
    if name not in table:
        raise KeyError(f"未知的 {kind} 输入映射 {name!r}，可选 {sorted(table)}")
    return table[name]
