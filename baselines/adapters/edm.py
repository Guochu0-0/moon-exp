"""EDM（AnyMatch 的 third_party/EDM_AnyMatch，AnyMatch@259ad34）；用于 AnyMatch-EDM。

AnyMatch 仓库缺 load_model.py，没有给出 EDM 的推理配置。依据（tmp/interfaces/anymatch.md §2，均带 file:line）：
- 模型：`EDM(lower_config(cfg)['edm'])`（src/edm/edm.py:13），DEPLOY=False，无 reparam；`load_state_dict` 剥 `matcher.`
  （edm.py:200-204），上游 lightning 用 strict=True。
- 配置（**推断**）：default + outdoor/edm_base.py 的值：MCONF_THR 0.05、SIGMA_THR 1e-6、BORDER_RM 0；
  NPE = [832, 832, 640, 640]（训练 832，测试输入 640）；TOPK = int(640/8 · 640/8 · 0.35) = 2240（test.py:92-95 的公式）。
- 输入：AnyMatch wrapper（src/utils/data_io_edm.py）与 LoFTR 系相同：灰度、长边 640、df 8、/255；512² → 640²，
  无需 pad；边长须被 32 整除（640 满足）。forward 内无归一化。这里直接喂 float [0,1]。
- 输出：`mkpts0_f/mkpts1_f/mconf`，输入网格、无半像素偏移（整数 = 像素中心），按中心对齐公式回映。
- 依赖：src/utils/misc.py 顶层 `from lightning.pytorch.utilities import rank_zero_only`，推理只用到其中的
  detect_NaN。环境里没有 lightning 时注入一个只含 rank_zero_only 的桩模块，避免为此装 lightning（会牵动 torch 版本）。
"""
from __future__ import annotations

import sys

import numpy as np

from .base import load_ckpt, long_side_size, stub_rank_zero_only, to_original


class EDMAdapter:
    def __init__(self, repo, weights, device="cuda", long_side=640, train_res=832, mconf_thr=0.05, sigma_thr=1e-6,
                 border_rm=0, topk_ratio=0.35):
        sys.path.insert(0, str(repo))
        stubbed = stub_rank_zero_only("lightning.pytorch")
        from src.config.default import get_cfg_defaults
        from src.edm import EDM
        from src.utils.misc import lower_config

        hn, wn = long_side_size(512, 512, long_side, df=8)
        cfg = get_cfg_defaults()
        cfg.EDM.COARSE.MCONF_THR = mconf_thr
        cfg.EDM.FINE.SIGMA_THR = sigma_thr
        cfg.EDM.COARSE.BORDER_RM = border_rm
        cfg.EDM.NECK.NPE = [train_res, train_res, hn, wn]
        cfg.EDM.COARSE.TOPK = int(hn / 8 * wn / 8 * topk_ratio)
        cfg.EDM.TEST_RES_H, cfg.EDM.TEST_RES_W = hn, wn
        self.model = EDM(lower_config(cfg)["edm"])
        self.model.load_state_dict(load_ckpt(weights)["state_dict"], strict=True)
        self.model = self.model.eval().to(device)
        self.device, self.long_side = device, long_side
        self.notes = (f"INFERRED cfg: mconf {mconf_thr}, sigma {sigma_thr}, border_rm {border_rm}, "
                      f"npe {cfg.EDM.NECK.NPE}, topk {cfg.EDM.COARSE.TOPK}; long_side={long_side}; "
                      f"lightning stub={stubbed}")

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

        t0, g0 = self._tensor(opt)
        t1, g1 = self._tensor(sar)
        batch = {"image0": t0, "image1": t1}
        with torch.no_grad():
            self.model(batch)
        kp0 = to_original(batch["mkpts0_f"].float().cpu().numpy(), *g0)
        kp1 = to_original(batch["mkpts1_f"].float().cpu().numpy(), *g1)
        return kp0, kp1, batch["mconf"].float().cpu().numpy()
