"""统一的仿射 RANSAC。只依赖 numpy + cv2，matcher 环境（py3.8+）和工作台环境都能 import。

口径（「定义评价协议与指标」#2）：cv2.estimateAffine2D，confidence 0.99999，每次调用前重置种子；
点对少于 3 或内点少于 3 记失败。坐标约定见 baselines/fit.py。
"""
from __future__ import annotations

import cv2
import numpy as np

CONFIDENCE = 0.99999
MAX_ITERS = 10000
SEED = 0


def fit_affine(M: np.ndarray, thr: float):
    """M: N×5 (x0, y0, x1, y1, conf)，中心约定。返回 (A 或 None, 内点掩码 或 None, 失败原因 或 None)。"""
    if len(M) < 3:
        return None, None, "few_matches"
    src = M[:, :2].astype(np.float64) + 0.5
    dst = M[:, 2:4].astype(np.float64) + 0.5
    cv2.setRNGSeed(SEED)
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                  maxIters=MAX_ITERS, confidence=CONFIDENCE)
    if A is None or inl is None or int(inl.sum()) < 3:
        return None, None, "few_inliers"
    return A, inl.ravel().astype(bool), None
