"""中间结果的显示素材：标量图 → 热力图 PNG，位移场 → 另一模态按位移摆回 frame 后的 PNG。只用于工作台显示。"""
from __future__ import annotations

import io

import numpy as np

# viridis 的 9 个采样点（matplotlib 同名色图），中间线性插值；不依赖 matplotlib
VIRIDIS = np.array([[68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
                    [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]], dtype=np.float64)


def colormap(t: np.ndarray) -> np.ndarray:
    """t ∈ [0, 1]（NaN 透明）→ RGBA uint8。"""
    ok = np.isfinite(t)
    x = np.clip(np.where(ok, t, 0), 0, 1) * (len(VIRIDIS) - 1)
    i = np.minimum(x.astype(int), len(VIRIDIS) - 2)
    f = (x - i)[..., None]
    rgb = VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f
    return np.dstack([np.round(rgb).astype(np.uint8), np.where(ok, 255, 0).astype(np.uint8)])


def value_range(a: np.ndarray) -> tuple[float | None, float | None]:
    v = a[np.isfinite(a)]
    return (float(v.min()), float(v.max())) if v.size else (None, None)


def magnitude(kind: str, a: np.ndarray) -> np.ndarray:
    """色标用的量：scalar 为取值，points 为 v 列，flow 为位移大小。"""
    return a[:, 2] if kind == "points" else np.linalg.norm(a, axis=-1) if kind == "flow" else a


def heatmap(a: np.ndarray) -> np.ndarray:
    """标量图按自身的最小、最大值映射到色图，分辨率不变（页面按 frame 拉伸）。"""
    lo, hi = value_range(a)
    if lo is None:
        return colormap(np.full(a.shape, np.nan))
    return colormap((a.astype(np.float64) - lo) / (hi - lo) if hi > lo else np.where(np.isfinite(a), 0.5, np.nan))


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """按下标坐标（像素中心为整数）双线性采样，越界处钳到边缘。img 为 H×W 或 H×W×C。"""
    h, w = img.shape[:2]
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(int), max(w - 2, 0))
    y0 = np.minimum(np.floor(y).astype(int), max(h - 2, 0))
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    img = img.astype(np.float64)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def warp(flow: np.ndarray, size: tuple[int, int], other: np.ndarray) -> np.ndarray:
    """位移场画成 warp：frame 影像（宽、高为 size）的每个像素按位移到另一模态 other（uint8 灰度）里取值，RGBA uint8。

    flow 是 h×w×2，按格点中心拉伸到 frame 大小；位移单位是原始 patch 的 px。落到 other 外的像素透明。
    """
    W, H = size
    h, w = flow.shape[:2]
    cy, cx = np.mgrid[0:H, 0:W] + 0.5                        # frame 像素中心（角点原点坐标）
    f = _bilinear(flow, cx * w / W - 0.5, cy * h / H - 0.5)
    tx, ty = cx + f[..., 0], cy + f[..., 1]                    # 另一模态里的对应点（角点原点坐标）
    oh, ow = other.shape[:2]
    inside = np.isfinite(tx) & np.isfinite(ty) & (tx >= 0) & (tx <= ow) & (ty >= 0) & (ty <= oh)
    v = np.round(_bilinear(other, np.nan_to_num(tx) - 0.5, np.nan_to_num(ty) - 0.5)).astype(np.uint8)
    return np.dstack([v, v, v, np.where(inside, 255, 0).astype(np.uint8)])


def png(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()
