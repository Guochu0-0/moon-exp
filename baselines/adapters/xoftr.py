"""XoFTR（OnderT/XoFTR，third_party/XoFTR）；MINIMA-XoFTR 用同一份代码（MINIMA 的子模块是同一上游），只换权重。

对照官方（docs/research/baselines/interfaces-a.md §2，读的是 e9635d8；这里钉 e0fbea4，只多修了 data_io.py:67 的 np.float）：
- 配置：`get_cfg_defaults(inference=True)`（必须 True，否则走训练分支，default.py:195-203），小写化后取 `xoftr`。
  不 import `src.utils.misc/data_io` 的 lower_config（前者要 pytorch_lightning），这里自带一份同样的小写化。
- 模型：裸模型路径（notebook cell 8/10），strict 加载，`load_state_dict` 去 `matcher.` 前缀（xoftr.py:90-94）。
- 输入：`(1,1,H,W)` float。forward 内逐图 z-score（xoftr.py:39-47），所以 [0,1] 上的线性尺度无关紧要。
  跳过包装器 `DataIOWrapper.from_cv_imgs`（其 /255 与 .cuda() 硬编码，data_io.py:57-76）。
- 分辨率：TEST.IMG*_RESIZE=640 长边，取整到 8 的倍数，cv2.resize 双线性（data_io.py:57-66）。
- 输出：`mkpts0_f/mkpts1_f/mconf_f`，输入网格、整数 = 像素中心，按中心对齐公式回映。
- 细层全部低于阈值时，官方强行吐一条 conf=1 的假匹配（fine_matching.py:87-89）。此时匹配数恰为 1，
  这里直接返回空（反正 < 3 对也会判失败，只是日志里的 n 不再误导）。
"""
from __future__ import annotations

import sys

import numpy as np

from .base import load_ckpt, long_side_size, to_original


def _lower(cfg):
    if not hasattr(cfg, "items"):
        return cfg
    return {k.lower(): _lower(v) for k, v in cfg.items()}


class XoFTRAdapter:
    def __init__(self, repo, weights, device="cuda", long_side=640):
        sys.path.insert(0, str(repo))
        from src.config.default import get_cfg_defaults
        from src.xoftr import XoFTR

        cfg = _lower(get_cfg_defaults(inference=True))
        self.model = XoFTR(config=cfg["xoftr"])
        self.model.load_state_dict(load_ckpt(weights)["state_dict"], strict=True)
        self.model = self.model.eval().to(device)
        self.device, self.long_side = device, long_side
        self.notes = f"long_side={long_side}, inference cfg, float input; single fake match -> empty"

    def _tensor(self, img):
        import cv2
        import torch

        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side, df=8)
        if (hn, wn) != (h, w):
            img = cv2.resize(img, (wn, hn))
        return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None].to(self.device), (h, w, hn, wn)

    def match(self, opt, sar):
        import torch

        t0, g0 = self._tensor(opt)
        t1, g1 = self._tensor(sar)
        batch = {"image0": t0, "image1": t1}
        with torch.no_grad():
            self.model(batch)
        conf = batch["mconf_f"].cpu().numpy()
        if len(conf) <= 1:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        kp0 = to_original(batch["mkpts0_f"].cpu().numpy(), *g0)
        kp1 = to_original(batch["mkpts1_f"].cpu().numpy(), *g1)
        return kp0, kp1, conf
