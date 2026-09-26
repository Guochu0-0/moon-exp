"""MatchAnything（HF Space LittleFrog/MatchAnything，third_party/MatchAnything，6a7bcb5）：ELoFTR 与 RoMa 两个模型。

代码在 `imcui/third_party/MatchAnything/`（下称 MA/）。对照官方 SAR 评测（docs/research/baselines/interfaces-b.md §1–3，
MA/scripts/evaluate/eval_visible_sar.sh:14,17 → MA/tools/evaluate_datasets.py）：
- 构造：官方用 `PL_LoFTR(cfg, ckpt, test_mode=True).matcher`（lightning_loftr.py:48-76），但它顶层 import
  pytorch_lightning、matplotlib、pynvml 等。这里只复刻它对推理有效的两步：按 METHOD 构造
  `LoFTR(config=lower['loftr'])` 或 `MatchAnything_Model(config=lower['roma'], test_mode=True)`，
  再 `load_state_dict(ckpt['state_dict'], strict=False)`。官方 strict=False 只打日志，这里核对 missing/unexpected，
  有 missing 就报错（RoMa 的 DINOv2 故意不在 state_dict 里，不算）。
- 配置：`get_cfg_defaults()` + `merge_from_file(configs/models/<m>_model.py)`。这两个配置文件直接改全局 `_CN`，
  所以一个进程只构造一个模型（runner 本来就是一方法一进程）。
- ELoFTR：SAR 集强制拉伸到 832×832（evaluate_datasets.py:117-118，dataset.py:229-241，cv2 双线性），
  `NPE=[832,832,832,832]`（--npe，:114-115），coarse thr 0.05（脚本 --thr 0.05，:120-121），FP16 关。
  输入 `(1,1,832,832)` float [0,1]，forward 内无归一化；官方 loader 的 uint8 读取与 /255 跳过。
  不传 scale0/1，输出在 832 网格，按中心对齐公式回映。
- RoMa：`forward_inference` 只读 `image*_rgb_origin[0]`（matchanything_roma_model.py:63-71）：
  原尺寸 3 通道 float [0,1]（官方由 PIL RGB /255 得到，灰度复制成 3 通道），模型内部拉伸到 560 / 864，
  不做 ImageNet 归一化（NORMALIZE_IMG=False）。评测外层 autocast(enabled=LOFTR.FP16=True)（roma_model.py:22）。
  输出已在原图坐标但像素中心在 +0.5（matcher.py:741-746），这里 −0.5。采样随机，runner 每个 pair 前重置种子。
  构造时要 DINOv2（hub 缓存）和 torchvision VGG19_bn（pretrained=True，随后被 ckpt 覆盖），离线时由 TORCH_HOME 提供缓存。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base import load_ckpt, to_original


def _lower(cfg):
    if not hasattr(cfg, "items"):
        return cfg
    return {k.lower(): _lower(v) for k, v in cfg.items()}


class MatchAnythingAdapter:
    def __init__(self, repo, weights, device="cuda", model="eloftr", size=832, coarse_thr=0.05):
        import torch

        root = Path(repo) / "imcui" / "third_party" / "MatchAnything"
        sys.path.insert(0, str(root))
        from src.config.default import get_cfg_defaults

        cfg = get_cfg_defaults()
        cfg.merge_from_file(str(root / "configs" / "models" / f"{model}_model.py"))
        cfg.METHOD = f"matchanything_{model}"
        if model == "eloftr":
            cfg.LOFTR.COARSE.NPE = [832, 832, size, size]
            cfg.LOFTR.MATCH_COARSE.THR = coarse_thr
            from src.loftr import LoFTR

            net = LoFTR(config=_lower(cfg)["loftr"])
        elif model == "roma":
            from third_party.ROMA.roma.matchanything_roma_model import MatchAnything_Model

            net = MatchAnything_Model(config=_lower(cfg)["roma"], test_mode=True)
        else:
            raise ValueError(model)
        res = net.load_state_dict(load_ckpt(weights)["state_dict"], strict=False)
        if res.missing_keys:
            raise RuntimeError(f"权重缺键 {len(res.missing_keys)} 个，例如 {res.missing_keys[:5]}")
        self.net = net.eval().to(device)
        self.model, self.size, self.device = model, size, device
        self.fp16 = bool(cfg.LOFTR.FP16)
        self.notes = (f"model={model}, unexpected_keys={len(res.unexpected_keys)}, fp16={self.fp16}, "
                      + (f"stretch {size}, npe=[832,832,{size},{size}], coarse_thr={coarse_thr}" if model == "eloftr"
                         else f"rgb_origin float, coarse {cfg.ROMA.TEST_TIME.COARSE_RES}, upsample "
                              f"{cfg.ROMA.TEST_TIME.UPSAMPLE_RES}, n_sample={cfg.ROMA.SAMPLE.N_SAMPLE}"))

    def match(self, opt, sar):
        import torch

        if self.model == "eloftr":
            import cv2

            h, w = opt.shape
            t = lambda a: torch.from_numpy(cv2.resize(np.ascontiguousarray(a, np.float32), (self.size, self.size)))[None, None].to(self.device)
            batch = {"image0": t(opt), "image1": t(sar)}
            with torch.no_grad(), torch.autocast("cuda", enabled=self.fp16):
                self.net(batch)
            g0 = (h, w, self.size, self.size)
            g1 = (*sar.shape, self.size, self.size)
            kp0 = to_original(batch["mkpts0_f"].float().cpu().numpy(), *g0)
            kp1 = to_original(batch["mkpts1_f"].float().cpu().numpy(), *g1)
        else:
            t = lambda a: torch.from_numpy(np.repeat(np.ascontiguousarray(a, np.float32)[None], 3, 0))[None].to(self.device)
            batch = {"image0_rgb_origin": t(opt), "image1_rgb_origin": t(sar)}
            with torch.no_grad(), torch.autocast("cuda", enabled=self.fp16):
                self.net(batch)
            kp0 = batch["mkpts0_f"].float().cpu().numpy() - 0.5
            kp1 = batch["mkpts1_f"].float().cpu().numpy() - 0.5
        return kp0, kp1, batch["mconf"].float().cpu().numpy()
