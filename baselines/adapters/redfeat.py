"""ReDFeat official VIS-SAR inference, without the demo's essential-matrix fit.

Reference: ACuOoOoO/ReDFeat@66df80027dcb649118259a91cadbff4b50adc880:
- match.py:24-29 constructs MMNet and loads checkpoint['model']; :114-125
  supplies the feature-count, scale, NMS threshold and border defaults.
- match.py:138-142,158-162 converts RGB to float and applies channel z-score.
- match.py:32-50 implements repeatability NMS/10px border; :53-109 runs the
  scale schedule and forward1/forward2, and collects coordinates/descriptors.
- match.py:151-155,171-175 selects top responses; :180-188 does BF ratio 0.9.
  Return those raw correspondences, before findEssentialMat at :190-193.

Use the official MMNet,
multiscale schedule, repeatability NMS and ratio matching. Keep optical forward1
and SAR forward2. Resize coordinates use this project's pixel-centre convention.
Importing the demo itself would overwrite CUDA_VISIBLE_DEVICES, so it is avoided.
"""
from __future__ import annotations

import sys

import numpy as np

from .base import load_ckpt, to_original


class ReDFeatAdapter:
    def __init__(self, repo, weights, device="cuda", num_features=4096,
                 min_size=256, max_size=1000, scale_factor=2**0.25,
                 repeatability_threshold=0.4, border=5, ratio=0.9):
        sys.path.insert(0, str(repo))
        from lib.model import MMNet

        self.model = MMNet()
        state = load_ckpt(weights)["model"]
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
        self.model = self.model.eval().to(device)
        self.device = device
        self.num_features = int(num_features)
        self.min_size, self.max_size = int(min_size), int(max_size)
        self.scale_factor = float(scale_factor)
        self.threshold, self.border, self.ratio = float(repeatability_threshold), int(border), float(ratio)
        if self.scale_factor <= 1 or self.min_size <= 0:
            raise ValueError("ReDFeat requires scale_factor > 1 and min_size > 0")
        self.notes = ("official VIS_SAR model; optical forward1/SAR forward2; channel z-score; "
                      "official multiscale NMS (10px border) and ratio=%.3g; pixel-centre resize; "
                      "no essential-matrix filtering" % self.ratio)

    def _features(self, image, branch):
        import torch
        import torch.nn.functional as F

        h, w = image.shape
        tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
        tensor = tensor.repeat(1, 3, 1, 1).to(self.device)
        tensor = (tensor - tensor.mean(dim=(-1, -2), keepdim=True)) / tensor.std(
            dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        coordinates, descriptors, scores = [], [], []
        scale = 1.0
        while scale + 0.001 >= self.min_size / max(h, w):
            if scale - 0.001 <= min(1.0, self.max_size / max(h, w)):
                with torch.no_grad():
                    desc, rep = getattr(self.model, branch)(tensor)
                # Official detector applies both args.border and a 10px NMS border.
                keep = (rep == F.max_pool2d(rep, 3, stride=1, padding=1)) & (rep >= self.threshold)
                border = max(self.border, 10)
                mask = torch.zeros_like(keep)
                mask[:, :, border:-border, border:-border] = True
                y, x = (keep & mask).nonzero(as_tuple=True)[2:4]
                kp = torch.stack((x, y), dim=-1).float().cpu().numpy()
                coordinates.append(to_original(kp, h, w, tensor.shape[-2], tensor.shape[-1]))
                descriptors.append(desc[0, :, y, x].t().float().cpu().numpy())
                scores.append(rep[0, 0, y, x].float().cpu().numpy())
            scale /= self.scale_factor
            hn, wn = round(h * scale), round(w * scale)
            tensor = F.interpolate(tensor, (hn, wn), mode="bilinear", align_corners=False)
        if not descriptors:
            return np.empty((0, 2)), np.empty((0, 0), dtype=np.float32)
        score = np.concatenate(scores)
        indices = np.argsort(-score, kind="stable")[:self.num_features]
        return np.concatenate(coordinates)[indices], np.ascontiguousarray(np.concatenate(descriptors)[indices], dtype=np.float32)

    def match(self, opt, sar):
        import cv2

        kp0, desc0 = self._features(opt, "forward1")
        kp1, desc1 = self._features(sar, "forward2")
        if len(desc0) == 0 or len(desc1) < 2:
            return np.empty((0, 2)), np.empty((0, 2)), None
        pairs = cv2.BFMatcher().knnMatch(desc0, desc1, k=2)
        good = [row[0] for row in pairs if len(row) == 2 and row[0].distance < self.ratio * row[1].distance]
        return (np.asarray([kp0[m.queryIdx] for m in good]).reshape(-1, 2),
                np.asarray([kp1[m.trainIdx] for m in good]).reshape(-1, 2), None)
