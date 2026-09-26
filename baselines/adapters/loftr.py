"""LoFTR（zju3dv/LoFTR，third_party/LoFTR）。

对照官方（行号见 docs/research/baselines/interfaces-a.md §1，commit df7ca80）：
- 模型：`src.loftr.LoFTR(config=default_cfg)`；`outdoor_ds.ckpt` 是旧权重，必须 `coarse.temp_bug_fix=False`
  （`default_cfg` 本来就是 False，这里显式写出）。`load_state_dict` 会去掉 `matcher.` 前缀，strict 保持 True。
- 输入：`(1,1,H,W)` float [0,1]。官方 loader `read_megadepth_gray` 读 uint8 再 ÷255（`src/utils/dataset.py:94-125`），
  这里跳过它，直接喂 float；forward 内部无归一化。
- 分辨率：室外官方测试为长边 840、H/W 取整到 8 的倍数（`configs/data/megadepth_test_1500.py:10`）；
  resize 用 cv2.resize 默认双线性，与官方 loader 相同。512² 无需 pad，不传 mask。
- 输出：`mkpts0_f/mkpts1_f/mconf`，在输入张量网格上，整数 = 像素中心。不往 batch 里放 scale0/1，
  由 base.to_original 用中心对齐公式映射回 512 网格（官方的 x·s 有 0.5·(s−1) 偏差）。
"""
from __future__ import annotations

import copy
import sys

import numpy as np

from .base import load_ckpt, long_side_size, to_original


class LoFTRAdapter:
    def __init__(self, repo, weights, device="cuda", long_side=840, temp_bug_fix=False):
        import torch

        sys.path.insert(0, str(repo))
        from src.loftr import LoFTR, default_cfg

        cfg = copy.deepcopy(default_cfg)
        cfg["coarse"]["temp_bug_fix"] = temp_bug_fix
        self.model = LoFTR(config=cfg)
        sd = load_ckpt(weights)["state_dict"]
        self.model.load_state_dict(sd, strict=True)
        self.model = self.model.eval().to(device)
        self.device, self.long_side = device, long_side
        self.notes = (f"long_side={long_side}, temp_bug_fix={temp_bug_fix}, coarse_thr={cfg['match_coarse']['thr']}, "
                      f"float input (no uint8)")

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
        kp0 = to_original(batch["mkpts0_f"].cpu().numpy(), *g0)
        kp1 = to_original(batch["mkpts1_f"].cpu().numpy(), *g1)
        return kp0, kp1, batch["mconf"].cpu().numpy()
