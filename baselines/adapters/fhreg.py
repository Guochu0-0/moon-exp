"""FHReg/GUSO official cross-domain inference, before FSC/KDE.

Reference: vision-heng/GUSO@f3e5420d59902edd025f06159b9a2b2320ea51d4:
- zero_shot_msaw_os.py:36-56 configures/constructs FHReg and strictly loads
  the checkpoint; :114-117 supplies float RGB [0,1].
- zero_shot_msaw_os.py:64-74 calls image_opt/image_sar and reads mkpts*_f;
  return those points before the FSC call at :163-164.
- fhreg/matching_module.py:337-342 stores the final, filtered fine-point
  confidence in mconf_f; coarse mconf is a different array (:164-192).
  fhreg/fhreg.py:121-124 explicitly skips refinement for matching_type='usm'.

Only the model is imported: the demo would overwrite CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import copy
import sys

import numpy as np

from .base import load_ckpt, to_original


class FHRegAdapter:
    def __init__(self, repo, weights, device="cuda", image_size=512, threshold=0.1):
        sys.path.insert(0, str(repo))
        from configs.test_config import test_cfg
        from utils.cvpr_ds_config import default_cfg
        from fhreg.fhreg import FHReg

        cfg = copy.deepcopy(dict(default_cfg))
        cfg["match_coarse"]["thr"] = threshold
        test_cfg["coarse_thr"] = threshold
        test_cfg["fine_thr"] = threshold
        self.model = FHReg(cfg, model_training=False)
        state = load_ckpt(weights)
        self.model.load_state_dict(state.get("state_dict", state), strict=True)
        self.model = self.model.eval().to(device)
        self.device, self.image_size = device, int(image_size)
        self.notes = ("official GUSO checkpoint/cross-domain float [0,1] path; 3 replicated channels; "
                      "image_size=%s, threshold=%s; raw mkpts before FSC/KDE" % (image_size, threshold))

    def _tensor(self, image):
        import cv2
        import torch

        h, w = image.shape
        hn = wn = self.image_size
        resized = cv2.resize(image, (wn, hn), interpolation=cv2.INTER_LINEAR) if (h, w) != (hn, wn) else image
        tensor = torch.from_numpy(np.ascontiguousarray(resized, dtype=np.float32))[None, None]
        return tensor.repeat(1, 3, 1, 1).to(self.device), (h, w, hn, wn)

    def match(self, opt, sar):
        import torch

        t0, g0 = self._tensor(opt)
        t1, g1 = self._tensor(sar)
        batch = {"image_opt": t0, "image_sar": t1}
        with torch.no_grad():
            result = self.model(batch)
        if result is not None:
            batch = result
        kp0 = to_original(batch["mkpts0_f"].float().cpu().numpy(), *g0)
        kp1 = to_original(batch["mkpts1_f"].float().cpu().numpy(), *g1)
        # FineSubMatching filters/reorders coarse candidates. Its mconf_f is
        # aligned with mkpts*_f; coarse mconf is valid only in explicit USM mode.
        if getattr(self.model, "config", {}).get("matching_type") == "usm":
            conf = batch.get("mconf")
        else:
            conf = batch.get("mconf_f")
        return kp0, kp1, None if conf is None else conf.float().cpu().numpy().reshape(-1)
