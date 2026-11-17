"""VMGGA optical-SAR raw matches, before demo homography fitting.

Reference: yeyuanxin110/VMGGA@5df54e852ba24135be45f782623db30032bf7602:
- demo_vmgga.py:58-85 constructs the configured model and loads the state.
- configs/demo/optical_sar.yaml:5-7 sets threshold 2e-5 and fine window 5.
- demo_vmgga.py:318-327 converts gray, pads to 16 and builds float [0,1].
- demo_vmgga.py:328-340 forwards and extracts in-bounds raw mkpts*_f;
  return these before ransac_filter at :342-344.

The 512px inputs already satisfy the 16px padding multiple; no resize is needed.
"""
from __future__ import annotations

import sys

import numpy as np

from .base import load_ckpt


class VMGGAAdapter:
    def __init__(self, repo, weights, device="cuda", threshold=2e-5, fine_window_size=5):
        sys.path.insert(0, str(repo))
        from src.config.model import get_vmgga_model_config
        from src.vmgga import VMGGA

        self.model = VMGGA(config=get_vmgga_model_config(threshold, fine_window_size))
        state = load_ckpt(weights)
        state = state.get("state_dict", state.get("model", state))
        self.model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state.items()}, strict=True)
        self.model = self.model.eval().to(device)
        self.device = device
        self.notes = "official optical_sar checkpoint; float gray; bottom/right pad16; raw matches before homography"

    def _tensor(self, image):
        import torch

        h, w = image.shape
        image = np.pad(image, ((0, (-h) % 16), (0, (-w) % 16)), mode="constant")
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(self.device)

    def match(self, opt, sar):
        import torch

        batch = {"image0": self._tensor(opt), "image1": self._tensor(sar)}
        with torch.no_grad():
            self.model(batch)
        kp0 = batch["mkpts0_f"].float().cpu().numpy().reshape(-1, 2)
        kp1 = batch["mkpts1_f"].float().cpu().numpy().reshape(-1, 2)
        valid = ((kp0[:, 0] >= 0) & (kp0[:, 0] < opt.shape[1]) &
                 (kp0[:, 1] >= 0) & (kp0[:, 1] < opt.shape[0]) &
                 (kp1[:, 0] >= 0) & (kp1[:, 0] < sar.shape[1]) &
                 (kp1[:, 1] >= 0) & (kp1[:, 1] < sar.shape[0]))
        conf = batch.get("mconf")
        conf = None if conf is None else conf.float().cpu().numpy().reshape(-1)[valid]
        return kp0[valid], kp1[valid], conf
