"""原始影像 → 方法输入的映射。

主表口径（「重跑 baseline」#3 的决定）：所有方法同一套映射，送 float32 [0,1] 单通道，不经 uint8 量化。
- 光学：uint8 / 255。
- SAR：S1 = b1 + b2 → dB → 下限 −25 dB → 逐 patch p2–p98 线性拉伸并截断到 [0,1]。
其余映射只用于 Val 上的消融。映射按名字登记，run 配置里写名字。
"""
from __future__ import annotations

import numpy as np

DB_FLOOR = -25.0


def sar_db(sar: np.ndarray) -> np.ndarray:
    """H×W×4 Stokes → S1 的 dB 图，下限 DB_FLOOR；非有限值当作下限。"""
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


OPTICAL = {"div255": optical_div255}
SAR = {"p2p98": sar_p2p98, "zscore_2p5": sar_zscore_2p5, "minmax": sar_minmax}


def get(kind: str, name: str):
    table = {"optical": OPTICAL, "sar": SAR}[kind]
    if name not in table:
        raise KeyError(f"未知的 {kind} 输入映射 {name!r}，可选 {sorted(table)}")
    return table[name]
