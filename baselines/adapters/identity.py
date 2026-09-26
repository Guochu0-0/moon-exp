"""恒等「匹配器」：规则网格上的点原样配对。只用来打通 match → fit → eval 整条链路，
其指标应与 `workbench` 的「未配准」完全一致，以此验证坐标约定。"""
from __future__ import annotations

import numpy as np


class Identity:
    notes = "grid identity, pipeline check only"

    def __init__(self, repo=None, weights=None, device=None, step: int = 32):
        self.step = step

    def match(self, opt, sar):
        h, w = opt.shape[:2]
        ys, xs = np.mgrid[self.step // 2:h:self.step, self.step // 2:w:self.step]
        kp = np.c_[xs.ravel(), ys.ravel()].astype(np.float64)
        return kp, kp.copy(), np.ones(len(kp))
