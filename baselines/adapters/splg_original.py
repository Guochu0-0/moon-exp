"""Original cvg/LightGlue + SuperPoint, without the MINIMA wrapper.

Use the pinned LightGlue constructors and their defaults. SuperPoint.extract
does its own 1024-long-side resize and returns pixel centers in the input
grid (utils.py Extractor.extract). No outer 640 resize, quantization, top-k
override, or MINIMA checkpoint is applied. Official weights come from the
torch hub cache; the runner records their paths/hashes separately.
"""
from __future__ import annotations

import sys

import numpy as np


class OriginalSPLGAdapter:
    def __init__(self, repo, weights=None, device="cuda"):
        if weights is not None:
            raise ValueError("Original SP+LG uses the official constructor weights, not a replacement checkpoint")
        sys.path.insert(0, str(repo))
        from lightglue import LightGlue, SuperPoint

        self.extractor = SuperPoint().eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        self.device = device
        self.notes = ("official SuperPoint/LightGlue defaults; SuperPoint max_num_keypoints=None; "
                      "extract resize=1024; float input; no outer resize; original-grid pixel centers")

    def _tensor(self, image):
        import torch

        return torch.from_numpy(np.ascontiguousarray(image, np.float32))[None, None].to(self.device)

    def match(self, opt, sar):
        import torch
        from lightglue.utils import rbd

        with torch.inference_mode():
            f0 = self.extractor.extract(self._tensor(opt))
            f1 = self.extractor.extract(self._tensor(sar))
            matches = self.matcher({"image0": f0, "image1": f1})
        f0, f1, matches = [rbd(x) for x in (f0, f1, matches)]
        indices = matches["matches"]
        return (f0["keypoints"][indices[:, 0]].cpu().numpy(),
                f1["keypoints"][indices[:, 1]].cpu().numpy(),
                matches["matching_scores0"][indices[:, 0]].cpu().numpy())
