"""RoMa 上游的双图共同增强：归一化后整数平移补零，再共同水平翻转（#267）。"""
from __future__ import annotations

import numpy as np
import torch

from ..config import ConfigError
from ..geom import HW, compose

PARAMS = {"shift": 32, "flip": 0.5}
MODELS = {"roma": {}}
Term = None


def fill(p):
    return p


def check(cfg):
    p = cfg["common"]
    if not 0 <= p["shift"] < 560 or not 0 <= p["flip"] <= 1:
        raise ConfigError("[common] shift 应在 [0,560)，flip 应在 [0,1]")
    if set(cfg) - {"model", "optim", "run", "pseudo", "common"} or "pseudo" not in cfg:
        raise ConfigError("[common] 只配合 RoMa 的 [pseudo]，不能混入其他训练成分")
    p = cfg["pseudo"]
    if cfg["model"]["train_res"] != "560" or any(p[k] for k in ("dense", "match_dir", "ema_target", "mask_fill")):
        raise ConfigError("[common] 要求 train_res=560、固定伪仿射，不配合额外目标或 mask_fill")


def data_kw(p):
    return {}


def sample(rng, batch, p, res=560):
    """连续的受控 RNG；返回每对 (dx,dy,flip) 及原网格角点坐标变换 G。"""
    specs, matrices = [], []
    for _ in range(batch):
        dx, dy = rng.integers(-p["shift"], p["shift"] + 1, size=2)
        flip = rng.random() < p["flip"]
        G = np.array([[1., 0., dx * HW / res], [0., 1., dy * HW / res]])
        if flip:
            G = compose([[-1, 0, HW], [0, 1, 0]], G)
        specs.append((int(dx), int(dy), bool(flip)))
        matrices.append(G)
    return specs, matrices


def warp_normalized(images, specs):
    """BCHW、已归一化的输入。整数搬移精确补 0，无插值、无循环回卷。"""
    out = torch.zeros_like(images)
    h, w = images.shape[-2:]
    for i, (dx, dy, flip) in enumerate(specs):
        x0, x1, y0, y1 = max(0, dx), min(w, w + dx), max(0, dy), min(h, h + dy)
        if x1 > x0 and y1 > y0:
            out[i, :, y0:y1, x0:x1] = images[i, :, y0-dy:y1-dy, x0-dx:x1-dx]
        if flip:
            out[i] = out[i].flip(-1)
    return out


def targets(An, inverse):
    """目标和双侧有效区；inverse 为归一化的 G⁻¹，交换方向后也适用。"""
    from .pseudo import affine_gt, grid

    def gt(h, w):
        x2, prob = affine_gt(An, h, w)
        g = grid(h, w, An.device)
        src = torch.einsum("bij,hwj->bhwi", inverse[:, :, :2], g) + inverse[:, None, None, :, 2]
        dst = torch.einsum("bij,bhwj->bhwi", inverse[:, :, :2], x2) + inverse[:, None, None, :, 2]
        return x2, prob * (src.abs() < 1).all(-1) * (dst.abs() < 1).all(-1)
    return gt
