"""WarpC 损失（#122；Truong et al., Warp Consistency, ICCV 2021, arXiv 2104.03308 §3.4–3.6）作为附加项。

由真实对 (I, J) = (光学, SAR) 和随机 warp W 构造 I′(x) = I(W(x))，再求两项：
- warp 监督：网络对 (I′, I) 的输出应等于 W；
- W-bipath：I′ → J → I 的复合应等于 W，即 m_{J→I}(m_{I′→J}(x)) = W(x)。作为采样坐标的 m_{I′→J} detach
  （原文 §3.5，官方代码 detach_flow_for_warping=True）；按位移写法，m_{I′→J} 本身那一项仍有梯度。

照原文 GLU-Net 在 MegaDepth 上第一阶段的设置（官方代码 DenseMatching train_settings/WarpC/train_WarpC_GLUNet_stage1.py、
utils_data/geometric_transformation_sampling/synthetic_warps_sampling.py；附录 C.2–C.3）：
- W 在单应、TPS、仿射-TPS 中等概率选一种，参数在归一化坐标 [-1, 1] 上均匀采样：单应四角、TPS 3×3 控制点各抖动 ±sigma_h；
  仿射-TPS 为 A = R_α R_shᵀ diag(λ1, λ2) R_sh + t（α、剪切角 ±15°，λ ∈ 1 ± 0.45，t ± 0.25）再接 TPS（±sigma_tps）；
  第一阶段不用弹性形变，bipath 项不用可见性掩码 V。
- 只给 I′ 加光度扰动：亮度、对比度 ×[0.4, 1.6]（ColorJitter 0.6；灰度图上饱和度、色调无作用），p = 0.2 高斯模糊 σ ∈ [0.2, 2]。
- 两项都只在 W(x) 落在 I 内的像素上算；bipath 另去掉 m_{I′→J} 落在 J 外的像素。
- 平衡：两项 detach 后，较小的一项放大到与较大的一项相等（官方代码 warp_consistency_losses.py 的做法；原文 §3.5 为
  λ = L_bipath / L_warp）。

和原文不同（适配 RoMa，自拟）：损失形式用 RoMa 原损失而不是多层 L1——warp 监督项与伪标签项完全相同（粗级锚点分类 +
certainty + 各尺度鲁棒回归），只把目标换成 W；bipath 项对各尺度的 flow 用同一个鲁棒回归。warp 直接在 560 输入网格上采样，
不做原文的 750 → 520 中心裁剪。三次前向 (I′, J)、(J, I)、(I′, I) 并成一个 batch。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .pseudo import grid, robust, roma_loss

PARAMS = {"weight": 1.0,
          "sigma_h": 250 / 750,    # 单应四角、TPS 控制点的抖动幅度（归一化坐标），原文第一阶段 250 / 750
          "sigma_tps": 60 / 750,   # 仿射-TPS 里 TPS 的抖动幅度，原文第一阶段 60 / 750
          "rot": 15.0,             # 仿射的旋转角、剪切角范围（度），π/12
          "scale": 0.45,           # λ1、λ2 ∈ 1 ± scale
          "trans": 0.25,           # 仿射平移（归一化坐标）
          "photo": True}           # I′ 的光度扰动
MODELS = {"roma": {}}


def fill(p):
    return p


def check(cfg):
    pass


def data_kw(p):
    return {}


# ---------- 随机 warp ----------

def _homography(src, dst):
    """4 点 DLT：src → dst 的 3×3。"""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(A))
    return vt[-1].reshape(3, 3) / vt[-1, -1]


def _tps_fit(c, t):
    """薄板样条：控制点 c (N,2) → 目标 t (N,2)，U(r) = r² log r²。返回 (权重 (N,2), 仿射 (3,2))。"""
    n = len(c)
    d2 = ((c[:, None] - c[None]) ** 2).sum(-1)
    K = np.where(d2 > 0, d2 * np.log(d2 + 1e-12), 0.0)
    P = np.c_[np.ones(n), c]
    L = np.zeros((n + 3, n + 3))
    L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
    sol = np.linalg.solve(L, np.r_[t, np.zeros((3, 2))])
    return sol[:n], sol[n:]


def _tps_apply(x, c, wa):
    """x (M,2) torch → (M,2)。"""
    w, a = (torch.from_numpy(v).to(x) for v in wa)
    c = torch.from_numpy(c).to(x)
    d2 = ((x[:, None] - c[None]) ** 2).sum(-1)
    U = torch.where(d2 > 0, d2 * torch.log(d2 + 1e-12), torch.zeros_like(d2))
    return U @ w + a[0] + x @ a[1:]


def _tps(rng, sigma):
    c = np.stack(np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3), indexing="xy"), -1).reshape(-1, 2)
    return c, _tps_fit(c, c + rng.uniform(-sigma, sigma, c.shape))


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def sample_warp(rng, p, h, w, device):
    """W：(h,w,2)，I′ 上每个像素在 I 上的位置（归一化坐标）。"""
    x = grid(h, w, device).reshape(-1, 2).double()
    kind = int(rng.integers(3))
    if kind == 0:      # 单应
        src = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
        H = torch.from_numpy(_homography(src, src + rng.uniform(-p["sigma_h"], p["sigma_h"], src.shape))).to(x)
        y = torch.cat([x, torch.ones_like(x[:, :1])], 1) @ H.T
        out = y[:, :2] / y[:, 2:]
    elif kind == 1:    # TPS
        c, wa = _tps(rng, p["sigma_h"])
        out = _tps_apply(x, c, wa)
    else:              # 仿射-TPS
        r = math.radians(p["rot"])
        rs = _rot(rng.uniform(-r, r))
        L = _rot(rng.uniform(-r, r)) @ rs.T @ np.diag(1 + rng.uniform(-p["scale"], p["scale"], 2)) @ rs
        t = rng.uniform(-p["trans"], p["trans"], 2)
        c, wa = _tps(rng, p["sigma_tps"])
        out = _tps_apply(x, c, wa) @ torch.from_numpy(L.T).to(x) + torch.from_numpy(t).to(x)
    return out.float().reshape(h, w, 2)


def photometric(img, rng):
    """img (1,H,W) [0,1]：亮度、对比度 ×[0.4, 1.6]，p = 0.2 高斯模糊 σ ∈ [0.2, 2]。"""
    x = img * rng.uniform(0.4, 1.6)
    m = x.mean()
    x = ((x - m) * rng.uniform(0.4, 1.6) + m).clamp(0, 1)
    if rng.random() < 0.2:
        from torchvision.transforms.functional import gaussian_blur
        s = float(rng.uniform(0.2, 2.0))
        x = gaussian_blur(x[None], 2 * math.ceil(3 * s) + 1, s)[0]
    return x


# ---------- 损失 ----------

def _resize(t, h, w):
    """(B,H,W,C) → (B,h,w,C)，双线性。"""
    return F.interpolate(t.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)


def split(corresps, k):
    return [{s: {n: v.chunk(k)[i] for n, v in d.items()} for s, d in corresps.items()} for i in range(k)]


def bipath(c_ij, c_ji, Wm, valid):
    """各尺度：comp = m_{I′→J} + [m_{J→I} − id](m_{I′→J}.detach())，对 W 求鲁棒回归。"""
    tot, st = 0.0, {}
    for s in sorted(c_ij, reverse=True):
        m_ij, m_ji = c_ij[s]["flow"].float(), c_ji[s]["flow"].float()
        h, w = m_ij.shape[-2:]
        d = m_ij.detach().permute(0, 2, 3, 1)
        samp = F.grid_sample(m_ji, d, mode="bilinear", padding_mode="zeros", align_corners=False)
        comp = (m_ij + samp - m_ij.detach()).permute(0, 2, 3, 1)
        Wt = _resize(Wm, h, w)
        mask = (_resize(valid[..., None].float(), h, w)[..., 0] > 0.99) & (d.abs() < 1).all(-1)
        l = robust((comp - Wt).norm(dim=-1)[mask], s, comp)
        tot = tot + l
        st[f"bip{s}"] = round(float(l), 5)
    return tot, st


class Term:
    def __init__(self, p, model, ds):
        self.p, self.model = p, model

    def __call__(self, st):
        i0, i1 = st.images()
        B, _, h, w = i0.shape
        Wm = torch.stack([sample_warp(st.rng, self.p, h, w, i0.device) for _ in range(B)])
        ip = F.grid_sample(i0, Wm, mode="bilinear", padding_mode="border", align_corners=False)
        if self.p["photo"]:
            ip = torch.stack([photometric(x, st.rng) for x in ip])
        valid = (Wm.abs() < 1).all(-1)
        c = self.model.forward(torch.cat([ip, i1, ip]), torch.cat([i1, i0, i0]))
        c_ij, c_ji, c_ii = split(c, 3)
        gt = lambda hh, ww: (_resize(Wm, hh, ww), _resize(valid[..., None].float(), hh, ww)[..., 0])
        l_w, s_w = roma_loss(c_ii, None, gt=gt)
        l_b, s_b = bipath(c_ij, c_ji, Wm, valid)
        fw, fb = float(l_w), float(l_b)
        big = max(fw, fb)
        loss = l_w * (big / fw if fw > 0 else 1.0) + l_b * (big / fb if fb > 0 else 1.0)
        stats = {"warpc": round(float(loss), 5), "warpc_sup": round(fw, 5), "warpc_bip": round(fb, 5),
                 "warpc_epe1_px": s_w.get("epe1_px"), "warpc_valid": round(float(valid.float().mean()), 4),
                 **{f"warpc_{k}": v for k, v in s_b.items()}}
        return self.p["weight"] * loss, stats, True
