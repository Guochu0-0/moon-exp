"""Official RoMa v1 with float tensor inputs at its two native resolutions.

Parskatt/RoMa@77f8d68 matcher.py:795-902 accepts already normalized tensors
and explicit high-resolution tensors. utils.py get_tuple_transform_ops uses
bicubic resizing and ImageNet normalization; reproduce those operations on
float inputs without routing through the optional uint8 PIL loader.
"""
from __future__ import annotations

import sys

import numpy as np

from .base import load_ckpt


class OriginalRoMaAdapter:
    def __init__(self, repo, weights, device="cuda", use_custom_corr=True):
        import torch

        torch.set_float32_matmul_precision("highest")
        sys.path.insert(0, str(repo))
        from romatch import roma_outdoor

        self.model = roma_outdoor(device=device, weights=load_ckpt(weights),
                                  use_custom_corr=use_custom_corr).eval()
        self.device = device
        self.notes = ("official RoMa v1 outdoor; coarse=560, upsample=864; float bicubic resize/ImageNet "
                      "normalization; no outer resize or uint8; symmetric; sample=10000; "
                      "float32_matmul_precision=highest; use_custom_corr=" + str(use_custom_corr))

    def _tensor(self, image, size):
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import normalize, resize

        rgb = torch.from_numpy(np.ascontiguousarray(image, np.float32))[None].expand(3, -1, -1)
        rgb = resize(rgb, list(size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        rgb = normalize(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return rgb[None].to(self.device)

    def match(self, opt, sar):
        import torch

        coarse = (self.model.h_resized, self.model.w_resized)
        high = self.model.upsample_res
        with torch.inference_mode():
            warp, certainty = self.model.match(self._tensor(opt, coarse), self._tensor(sar, coarse),
                                               im_A_high_res=self._tensor(opt, high),
                                               im_B_high_res=self._tensor(sar, high), batched=True,
                                               device=self.device)
            matches, confidence = self.model.sample(warp, certainty)
            first, second = self.model.to_pixel_coordinates(matches, *opt.shape, *sar.shape)
        # RoMa coordinates use half-integer pixel centers, the baseline adapter uses integers.
        return (first.float().cpu().numpy() - 0.5, second.float().cpu().numpy() - 0.5,
                confidence.float().cpu().numpy())
