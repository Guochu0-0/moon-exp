"""适配器共用的几何小工具。"""
from __future__ import annotations

import numpy as np


def load_ckpt(path):
    """torch.load 到 CPU。权重是官方/论文发布的 PL checkpoint，含非张量对象；torch ≥ 2.6 默认 weights_only=True 会拒读。"""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:   # torch < 1.13 没有 weights_only
        return torch.load(path, map_location="cpu")


def long_side_size(h: int, w: int, long_side, df: int = 1) -> tuple:
    """长边缩放到 long_side（None 不缩放），再向下取整到 df 的倍数。返回 (h_new, w_new)。"""
    if long_side:
        s = long_side / max(h, w)
        h, w = int(round(h * s)), int(round(w * s))
    return h // df * df, w // df * df


def to_original(kp: np.ndarray, h: int, w: int, h_new: int, w_new: int) -> np.ndarray:
    """resize 网格上的像素中心坐标 → 原网格。cv2.resize 按像素中心对齐：x = (x' + 0.5)·s − 0.5。
    官方代码多用 x = x'·s，差 0.5·(s − 1)，这里不照搬。"""
    kp = np.asarray(kp, dtype=np.float64).reshape(-1, 2)
    s = np.array([w / w_new, h / h_new])
    return (kp + 0.5) * s - 0.5
