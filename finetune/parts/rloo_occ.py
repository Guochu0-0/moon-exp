"""遮挡约束奖励（整对仿射，#249；[rloo] occ = true）的遮挡生成、掩码与梯度大小测量。训练里怎么用见 finetune/parts/rloo.py
模块说明「遮挡约束奖励」。

遮挡照 E40（runs/E40/code/infer.py 的 oc 视角）：光学图上取一块边长 side 原网格 px 的正方形，左上角 (x, y) 在
[0, HW − side] 上按整数均匀抽；SAR 上遮住这块正方形经参照仿射 A（光学 → SAR，原网格角点约定）映射过去的四边形。
被遮处填高斯噪声，均值、标准差取这张图自己的均值、标准差，截到 [0, 1]。与 E40 的两处不同：
- 判定像素是否被遮，看像素中心是否落在正方形（SAR 上为它的像，等价于中心经 A⁻¹ 映回光学后落在正方形内）里；
  E40 用 cv2.fillConvexPoly 按 1/16 px 精度填四边形，只在边界上差一圈像素；
- 训练时 SAR 经过了已知扰动，画面外补 0；SAR 上只遮四边形里的非补零区像素（补零区保持 0，不让画面边界变样），
  噪声的均值、标准差也只取非补零区（rloo.sar_valid）。E40 的 SAR 没有补零区。
网络输入网格（训练为 560）与原网格（HW = 512）只差一个整体缩放：输入像素 j 的中心在原网格的 (j + 0.5)·HW / W。
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..geom import HW, inv_affine


def draw_box(rng: np.random.Generator, side: float) -> tuple[int, int]:
    """左上角 (x, y)（原网格 px，整数），各自在 [0, HW − side] 上均匀抽（同 E40 的 rng.integers(0, HW − side + 1)）。"""
    hi = int(math.floor(HW - side)) + 1
    x, y = rng.integers(0, hi, 2)
    return int(x), int(y)


def _centers(h: int, w: int, device) -> torch.Tensor:
    """h×w 网格各像素中心在原网格里的坐标 (h, w, 2)，(x, y) 顺序，角点约定。"""
    ys = (torch.arange(h, device=device, dtype=torch.float64) + 0.5) * (HW / h)
    xs = (torch.arange(w, device=device, dtype=torch.float64) + 0.5) * (HW / w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], -1)


def _inside(pts: torch.Tensor, box, side: float) -> torch.Tensor:
    x, y = box
    return (pts[..., 0] >= x) & (pts[..., 0] < x + side) & (pts[..., 1] >= y) & (pts[..., 1] < y + side)


def opt_mask(box, side: float, h: int, w: int, device) -> torch.Tensor:
    """光学图 h×w 网格上被遮的像素（中心落在正方形内）→ (h, w) bool。粗格、细化各级的网格同样用它。"""
    return _inside(_centers(h, w, device), box, side)


def sar_mask(box, side: float, A, h: int, w: int, device) -> torch.Tensor:
    """SAR h×w 网格上被遮的像素：中心经 A⁻¹ 映回光学原网格后落在正方形内，即落在正方形经 A 的像（四边形）里 → (h, w) bool。"""
    Ai = torch.from_numpy(np.asarray(inv_affine(np.asarray(A, np.float64)), np.float64)).to(device)
    u = _centers(h, w, device)
    v = torch.einsum("ij,hwj->hwi", Ai[:, :2], u) + Ai[:, 2]
    return _inside(v, box, side)


def fill_noise(img: torch.Tensor, mask: torch.Tensor, gen: torch.Generator, valid: torch.Tensor | None = None):
    """img (1, H, W) ∈ [0, 1]；mask (H, W) bool → 被遮处换成 N(均值, 标准差) 并截到 [0, 1] 的新张量。均值、标准差取
    valid（(H, W) bool，None = 全图）内的像素；valid 里没有像素时退回全图。"""
    x = img[0].float()
    ref = x[valid] if valid is not None and bool(valid.any()) else x.flatten()
    mu, sd = ref.mean(), ref.std()
    noise = (mu + sd * torch.randn(x.shape, device=x.device, generator=gen)).clamp(0.0, 1.0)
    return torch.where(mask, noise, x)[None].to(img.dtype)


@torch.no_grad()
def grads(params) -> list:
    """各参数当前 .grad 的拷贝（None 照留）。"""
    return [None if q.grad is None else q.grad.detach().clone() for q in params]


@torch.no_grad()
def grad_norm(params, base=None) -> float:
    """√Σ‖g − base‖²：g 为各参数当前的 .grad，base 为 grads() 的拷贝（None = 0）。没有梯度的参数按 0。"""
    s = 0.0
    for i, q in enumerate(params):
        g = q.grad
        b = None if base is None else base[i]
        if g is None and b is None:
            continue
        if g is None:
            s += float(b.double().square().sum())
        elif b is None:
            s += float(g.double().square().sum())
        else:
            s += float((g.double() - b.double()).square().sum())
    return math.sqrt(s)


@torch.no_grad()
def split_norms(params, base) -> tuple[float, float, float]:
    """两项的梯度范数（#249）：base 为只有现有项时 grads() 的拷贝，当前 .grad 为两项之和。新项梯度 = 当前 − base。
    返回 (现有项在共用参数上的范数, 新项的范数, 现有项在全部参数上的范数)；共用参数 = 新项梯度不全为 0 的参数
    （新项梯度在其余参数上为 0，它的范数不受这个限定影响）。"""
    s_main, s_occ, s_all = 0.0, 0.0, 0.0
    for q, b in zip(params, base):
        g = q.grad
        bm = 0.0 if b is None else float(b.double().square().sum())
        s_all += bm
        if g is None:
            continue
        d = float((g.double() - (0.0 if b is None else b.double())).square().sum())
        if d > 0:
            s_occ += d
            s_main += bm
    return math.sqrt(s_main), math.sqrt(s_occ), math.sqrt(s_all)


@torch.no_grad()
def restore(params, saved):
    """把各参数的 .grad 恢复成 grads() 存下的拷贝（只量不学时撤掉新项的梯度）。"""
    for q, g in zip(params, saved):
        q.grad = None if g is None else g
