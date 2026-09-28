"""学生侧扰动（「【伪标签】首轮实验设计」#53，调研 C2：Noisy-Student 式「学生看扰动、老师看干净」）。

- 几何：对 SAR 施加已知随机仿射 T（绕 patch 中心的旋转、各向同性缩放 + 平移），在原网格（512）上 warp，图外补 0。
  离线伪仿射 A（光学 → SAR，角点约定）随之精确变为 T∘A（compose），不引入新的标签噪声。
  平移 ≥ 1–2 个粗格（原网格 6.4 px / 格），让学生不能靠「按位置配」拿到目标。
- 光度：两侧各自随机 gamma、对比度；SAR 另加乘性 gamma 散斑，光学另加高斯噪声；可选轻微高斯模糊。

坐标约定同 finetune/pseudo.py：T、A 都是原网格角点约定的 2×3。
"""
from __future__ import annotations

import cv2
import numpy as np

HW = 512


def sample_T(rng, shift=12.0, rot=3.0, scale=0.03, hw=HW):
    """随机仿射（角点约定）：q' = R·S·(q − c) + c + t，c = patch 中心。shift 单位原网格 px，rot 单位度。"""
    th = np.deg2rad(rng.uniform(-rot, rot))
    sc = 1.0 + rng.uniform(-scale, scale)
    L = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.array([hw / 2, hw / 2])
    t = rng.uniform(-shift, shift, 2)
    return np.c_[L, c - L @ c + t]


def compose(T, A):
    """T∘A，两者 2×3。"""
    H = lambda M: np.r_[np.asarray(M, np.float64).reshape(2, 3), [[0, 0, 1]]]
    return (H(T) @ H(A))[:2]


def warp(img, T):
    """按 T（角点约定，原 → 新）warp 原网格影像；cv2 用中心约定：p' = L p + (t + L·0.5 − 0.5)。"""
    L, t = T[:, :2], T[:, 2]
    M = np.c_[L, t + L @ [0.5, 0.5] - 0.5].astype(np.float32)
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def photometric(img, rng, sar):
    """img: float32 [0,1]。"""
    x = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.4)
    m = x.mean()
    x = (x - m) * rng.uniform(0.8, 1.2) + m
    if sar and rng.random() < 0.5:   # 乘性散斑：均值 1 的 gamma 分布，视数 4–16
        looks = rng.uniform(4, 16)
        x = x * rng.gamma(looks, 1 / looks, x.shape).astype(np.float32)
    elif not sar and rng.random() < 0.5:
        x = x + rng.normal(0, rng.uniform(0.005, 0.02), x.shape).astype(np.float32)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.0))
    return np.clip(x, 0, 1).astype(np.float32)
