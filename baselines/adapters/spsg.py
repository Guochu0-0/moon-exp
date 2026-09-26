"""SuperPoint + SuperGlue（magicleap/SuperGluePretrainedNetwork，third_party/SuperGlue，ddcf11f）。

对照官方（docs/research/baselines/interfaces-a.md §3）：
- 模型：`models.matching.Matching(config)`（matching.py:49-84）；权重在仓库里，构造时自动加载
  （superpoint.py:136-137，superglue.py:223-226）。SuperGlue 类默认是 indoor，这里显式用 outdoor。
- 配置：README 的室外推荐（README.md:217-220）：outdoor、长边 1600、max_keypoints 2048、nms_radius 3、
  sinkhorn 20、match_threshold 0.2、keypoint_threshold 0.005、resize_float。
- 输入：`(1,1,H,W)` float [0,1]。官方 `read_image` 读 uint8，resize_float 时先转 float 再 cv2.resize，再 /255
  （utils.py:263-282）；这里直接在 [0,1] float 上 resize，等价且不量化。检测用绝对阈值 0.005，对灰度尺度敏感。
- 输出：`keypoints0/1`（resize 网格上的整数像素，整数 = 像素中心）、`matches0`、`matching_scores0`；
  官方脚本不回映原图，这里按中心对齐公式回映。
"""
from __future__ import annotations

import sys

import numpy as np

from .base import long_side_size, to_original


class SPSGAdapter:
    def __init__(self, repo, weights=None, device="cuda", long_side=1600, superglue="outdoor", max_keypoints=2048,
                 nms_radius=3, keypoint_threshold=0.005, sinkhorn_iterations=20, match_threshold=0.2):
        sys.path.insert(0, str(repo))
        from models.matching import Matching

        cfg = {"superpoint": {"nms_radius": nms_radius, "keypoint_threshold": keypoint_threshold,
                              "max_keypoints": max_keypoints},
               "superglue": {"weights": superglue, "sinkhorn_iterations": sinkhorn_iterations,
                             "match_threshold": match_threshold}}
        self.model = Matching(cfg).eval().to(device)
        self.device, self.long_side = device, long_side
        self.notes = f"{cfg}, long_side={long_side}, float resize (resize_float)"

    def _tensor(self, img):
        import cv2
        import torch

        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side)
        if (hn, wn) != (h, w):
            img = cv2.resize(img, (wn, hn))
        return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None].to(self.device), (h, w, hn, wn)

    def match(self, opt, sar):
        import torch

        t0, g0 = self._tensor(opt)
        t1, g1 = self._tensor(sar)
        with torch.no_grad():
            p = self.model({"image0": t0, "image1": t1})
        k0 = p["keypoints0"][0].cpu().numpy()
        k1 = p["keypoints1"][0].cpu().numpy()
        m = p["matches0"][0].cpu().numpy()
        c = p["matching_scores0"][0].cpu().numpy()
        v = m > -1
        return to_original(k0[v], *g0), to_original(k1[m[v]], *g1), c[v]
