"""SuperPoint + LightGlue（cvg/LightGlue v0.2，third_party/LightGlue，edb2b83 = MINIMA 的子模块版本）；用于 MINIMA-SP+LG。

对照 MINIMA `load_sp_lg`（load_model.py:64-140；docs/research/baselines/interfaces-b.md §8）：
- SuperPoint：官方 v1 权重（MINIMA 不微调），`max_num_keypoints=2048, detection_threshold=0.0005, nms_radius=4,
  remove_borders=4`。
- LightGlue：`LightGlue(features='superpoint', n_layers=9, flash=True, mp=False, depth_confidence=0.95,
  width_confidence=0.99, filter_threshold=0.1)`，构造时先加载官方 superpoint_lightglue 权重（hub 缓存），
  再把 MINIMA 权重的 `self_attn.{i}`/`cross_attn.{i}` 改名成 `transformers.{i}.*` 后 strict=False 覆盖。
  官方不核对；这里核对覆盖后没有缺键（否则会静默沿用官方 LG 权重），有缺键就报错。
- 输入：MINIMA wrapper 先 cv2 双线性长边 640（df 8）、/255 得到 (1,1,640,640)，再 `extractor.extract(img)`
  内部 kornia antialias 长边缩放到 1024（SuperPoint.preprocess_conf），关键点换回 640 网格
  `(kp+0.5)/scale-0.5`（lightglue/utils.py:137-147）。这里从 float [0,1] 起步，其余相同。
- 输出：`matches` 索引取点、`matching_scores0` 作置信度；640 网格、整数 = 像素中心，按中心对齐公式回映 512
  （官方 wrapper 直接乘 scale）。
"""
from __future__ import annotations

import sys

import numpy as np

from .base import load_ckpt, long_side_size, to_original

SP_CONF = dict(descriptor_dim=256, nms_radius=4, max_num_keypoints=2048, detection_threshold=0.0005, remove_borders=4)
LG_CONF = dict(name="lightglue", input_dim=256, descriptor_dim=256, add_scale_ori=False, n_layers=9, num_heads=4,
               flash=True, mp=False, depth_confidence=0.95, width_confidence=0.99, filter_threshold=0.1, weights=None)


class SPLGAdapter:
    def __init__(self, repo, weights, device="cuda", long_side=640):
        sys.path.insert(0, str(repo))
        from lightglue import LightGlue, SuperPoint

        self.extractor = SuperPoint(**SP_CONF).eval().to(device)
        self.matcher = LightGlue(features="superpoint", **LG_CONF).eval().to(device)
        sd = load_ckpt(weights)
        for i in range(LG_CONF["n_layers"]):
            sd = {k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn"): v for k, v in sd.items()}
            sd = {k.replace(f"cross_attn.{i}", f"transformers.{i}.cross_attn"): v for k, v in sd.items()}
        res = self.matcher.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"MINIMA LightGlue 权重缺键 {len(res.missing_keys)} 个，例如 {res.missing_keys[:5]}")
        self.device, self.long_side = device, long_side
        self.notes = (f"sp={SP_CONF}, lg filter 0.1 depth 0.95 width 0.99 flash, long_side={long_side} then extract "
                      f"resize 1024; unexpected_keys={len(res.unexpected_keys)}")

    def _tensor(self, img):
        import cv2
        import torch

        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side, df=8)
        if (hn, wn) != (h, w):
            img = cv2.resize(np.ascontiguousarray(img, np.float32), (wn, hn))
        return torch.from_numpy(np.ascontiguousarray(img, np.float32))[None, None].to(self.device), (h, w, hn, wn)

    def match(self, opt, sar):
        import torch
        from lightglue.utils import rbd

        t0, g0 = self._tensor(opt)
        t1, g1 = self._tensor(sar)
        with torch.no_grad():
            f0 = self.extractor.extract(t0)
            f1 = self.extractor.extract(t1)
            m01 = self.matcher({"image0": f0, "image1": f1})
        f0, f1, m01 = [rbd(x) for x in (f0, f1, m01)]
        idx = m01["matches"]
        k0 = f0["keypoints"][idx[:, 0]].cpu().numpy()
        k1 = f1["keypoints"][idx[:, 1]].cpu().numpy()
        conf = m01["matching_scores0"][idx[:, 0]].cpu().numpy()
        return to_original(k0, *g0), to_original(k1, *g1), conf
