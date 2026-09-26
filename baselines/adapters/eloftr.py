"""EfficientLoFTR（zju3dv/EfficientLoFTR，third_party/EfficientLoFTR，07e9c14）；用于 MINIMA-ELoFTR。

MINIMA 只发布了 minima_eloftr.ckpt，没有推理代码（docs/research/baselines/interfaces-b.md §9），所以用上游官方路径（README
"Inference"）：`LoFTR(config=deepcopy(full_default_cfg))` → `load_state_dict(ckpt['state_dict'])`（剥 `matcher.`，strict）
→ `reparameter(matcher)`（README 注明 "Essential for good performance"）→ eval。精度 fp32（官方默认，MP/HALF 关）。
- 配置：full 模型，coarse thr 0.2（full_config.py:37 推荐值）；NPE 按官方注释设为 [832, 832, long_side, long_side]（:33）。
- 输入：沿用 MINIMA 的 LoFTR 系惯例长边 640（其 TEST.IMG*_RESIZE），cv2 双线性，边长须被 32 整除（640 满足）；
  官方从 uint8 /255 得到 [0,1]，这里直接喂 float。forward 内无归一化。
- 输出：`mkpts0_f/mkpts1_f/mconf`，输入网格、整数 = 像素中心，按中心对齐公式回映。
- 自检（光学对自身平移 (7,−5)）：零平移精确，非整格平移有 0.5–1 px 误差，512/640/832 都如此；其他方法 ≤0.2 px。
  权重与配置的搭配没有官方依据，结果解读时注意。
- 依赖：src/utils/misc.py 顶层 import pytorch_lightning 只为 rank_zero_only，没装时打桩（base.stub_rank_zero_only）。
"""
from __future__ import annotations

import copy
import sys

import numpy as np

from .base import load_ckpt, long_side_size, stub_rank_zero_only, to_original


class ELoFTRAdapter:
    def __init__(self, repo, weights, device="cuda", long_side=640, model_type="full", coarse_thr=0.2):
        sys.path.insert(0, str(repo))
        stubbed = stub_rank_zero_only("pytorch_lightning")
        from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

        cfg = copy.deepcopy(full_default_cfg if model_type == "full" else opt_default_cfg)
        hn, wn = long_side_size(512, 512, long_side, df=32)
        cfg["coarse"]["npe"] = [832, 832, hn, wn]
        cfg["match_coarse"]["thr"] = coarse_thr
        m = LoFTR(config=cfg)
        m.load_state_dict(load_ckpt(weights)["state_dict"], strict=True)
        self.model = reparameter(m).eval().to(device)
        self.device, self.long_side = device, long_side
        self.notes = (f"{model_type} cfg, npe={cfg['coarse']['npe']}, coarse_thr={coarse_thr}, reparameter, fp32, "
                      f"long_side={long_side}, pl stub={stubbed}")

    def _tensor(self, img):
        import cv2
        import torch

        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side, df=32)
        if (hn, wn) != (h, w):
            img = cv2.resize(np.ascontiguousarray(img, np.float32), (wn, hn))
        return torch.from_numpy(np.ascontiguousarray(img, np.float32))[None, None].to(self.device), (h, w, hn, wn)

    def match(self, opt, sar):
        import torch

        t0, g0 = self._tensor(opt)
        t1, g1 = self._tensor(sar)
        batch = {"image0": t0, "image1": t1}
        with torch.no_grad():
            self.model(batch)
        kp0 = to_original(batch["mkpts0_f"].float().cpu().numpy(), *g0)
        kp1 = to_original(batch["mkpts1_f"].float().cpu().numpy(), *g1)
        return kp0, kp1, batch["mconf"].float().cpu().numpy()
