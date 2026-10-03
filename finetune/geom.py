"""坐标约定与仿射小工具，各训练成分共用。

- 原网格：patch 原尺寸 512。网络输入网格：模型自己的分辨率（LoFTR 长边 640）。两者都是「整数 = 像素中心」。
- 仿射 A：2×3，光学 → SAR，原网格**角点约定**（baselines.fit 两侧 +0.5 后估计，与 workbench 同）。
"""
from __future__ import annotations

import numpy as np

HW = 512
CORNERS = np.array([[0, 0], [HW, 0], [0, HW], [HW, HW]], np.float64)   # 角点约定


def in_to_orig(p, s):
    """输入网格 → 原网格（中心约定），s = 原尺寸 / 输入尺寸。与 baselines.adapters.base.to_original 相同。"""
    return (p + 0.5) * s - 0.5


def orig_to_in(p, s):
    return (p + 0.5) / s - 0.5


def apply_affine(A, p):
    """A: 2×3（角点约定）；p: N×2 中心约定 → N×2 中心约定。"""
    q = p + 0.5
    return q @ A[:, :2].T + A[:, 2] - 0.5


def compose(T, A):
    """T∘A，两者 2×3。"""
    H = lambda M: np.r_[np.asarray(M, np.float64).reshape(2, 3), [[0, 0, 1]]]
    return (H(T) @ H(A))[:2]


def inv_affine(A):
    A = np.asarray(A, np.float64).reshape(2, 3)
    Li = np.linalg.inv(A[:, :2])
    return np.c_[Li, -Li @ A[:, 2]]


def identity_dist(A):
    """仿射 A 相对 [I|0] 的 patch 四角平均位移，px。A 为 None 返回 None。"""
    if A is None:
        return None
    A = np.asarray(A, np.float64).reshape(2, 3)
    return float(np.linalg.norm(CORNERS @ A[:, :2].T + A[:, 2] - CORNERS, axis=1).mean())
