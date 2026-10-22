"""几何模型（#195 的 a 层）：策略为以 M 为中心、标准差 s 的二维各向同性高斯，奖励 r = exp(−½(|a − R|/δ)²)，
真值误差 e = |a − T|。两种更新对期望真值误差的一阶变化 ΔE[e] 都有闭式（长度单位一律 px，张量 float64）：

- 细化（更新均值）：∇_M E[r] = (δ²/v)·exp(−|M − R|²/2v)·(R − M)/v，v = s² + δ²；
  ∇_M E[e] = g(|M − T|, s)·(M − T)/|M − T|，g 为 Rice 均值对 |M − T| 的导数（rice_slope，∈ [0, 1)）；
  ΔE[e] = ∇E[r]·∇E[e]。两个梯度分别指向 R、背离 T，所以 ΔE[e] < 0 当且仅当 (M − R)·(M − T) > 0，即 M 在以 TR 为直径的圆外。
- 粗级（每个锚点一个独立 logit，Δθ_j ∝ π_j(r_j − r̄)）：ΔE[e] = Σ_j π_j²(r_j − r̄)(e_j − ē)。把锚点换成连续位置、
  π_j 换成密度 p 乘锚点面积 A（64 × 64 锚点铺满 512 px，A = 8² px²）：ΔE[e] = A ∫ p²(r − r̄)(e − ē) da
  = A/(4πs²) · E_q[(r − r̄)(e − ē)]，q = N(M, s²/2)，r̄、ē 为 p 下的均值；各项都是高斯积或 Rice 均值，有闭式。
"""
from __future__ import annotations

import math

import torch

SQ = math.sqrt(math.pi / 2)
AREA = (512 / 64) ** 2        # 一个粗级锚点占的面积（px²）


def rice_mean(nu, s):
    """a ~ N(M, s²I)（二维）时 E|a − T|，nu = |M − T|。"""
    x = nu ** 2 / (4 * s ** 2)
    return s * SQ * ((1 + 2 * x) * torch.special.i0e(x) + 2 * x * torch.special.i1e(x))


def rice_slope(nu, s):
    """d E|a − T| / d nu = √(π/2)·(nu/2s)·e^{−x}(I0(x) + I1(x))，x = nu²/4s²。"""
    x = nu ** 2 / (4 * s ** 2)
    return SQ * nu / (2 * s) * (torch.special.i0e(x) + torch.special.i1e(x))


def gauss_mean(m2, s2, d2):
    """a ~ N(M, s2·I) 时 E[exp(−|a − R|²/2·d2)]，m2 = |M − R|²。"""
    v = s2 + d2
    return d2 / v * torch.exp(-m2 / (2 * v))


def fine_a(M, T, R, sig, delta):
    """细化：M、T、R (...,2)，sig、delta 标量或可广播 → ΔE[e] (...)。"""
    dR, dT = R - M, M - T
    v = sig ** 2 + delta ** 2
    gr = gauss_mean((dR ** 2).sum(-1), sig ** 2, delta ** 2) / v               # ∇E[r] = gr·(R − M)
    nu = dT.norm(dim=-1)
    ge = rice_slope(nu, sig) / nu.clamp(min=1e-300)                          # ∇E[e] = ge·(M − T)
    return gr * ge * (dR * dT).sum(-1)


def coarse_a(M, T, R, s, delta, area=AREA):
    """粗级：M、T、R (...,2)，s (...) → ΔE[e] (...)。"""
    s2, d2, h2 = s ** 2, delta ** 2, s ** 2 / 2
    mR2 = ((M - R) ** 2).sum(-1)
    rbar = gauss_mean(mR2, s2, d2)                                           # p 下
    ebar = rice_mean((M - T).norm(dim=-1), s)
    q_r = gauss_mean(mR2, h2, d2)                                            # q 下
    q_e = rice_mean((M - T).norm(dim=-1), h2.sqrt() if torch.is_tensor(h2) else math.sqrt(h2))
    sp2 = h2 * d2 / (h2 + d2)                                                # r·q ∝ N(mp, sp2)
    w = (d2 / (h2 + d2))[..., None] if torch.is_tensor(d2 / (h2 + d2)) else d2 / (h2 + d2)
    mp = M * w + R * (1 - w)
    sp = sp2.sqrt() if torch.is_tensor(sp2) else math.sqrt(sp2)
    q_re = q_r * rice_mean((mp - T).norm(dim=-1), sp)
    cov = q_re - rbar * q_e - ebar * q_r + rbar * ebar
    return area / (4 * math.pi * s2) * cov
