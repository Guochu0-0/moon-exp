"""细级 RL（首轮 RL 实验，地图 #1；设计见 docs/design/rl-modeling.md）。

单步 bandit：state = patch 对；action = 每个粗匹配的细级 SAR 位置；一对采 K 组，组内 baseline（GRPO 式）。

策略（设计文档 5.1 的 (b)，训练与推理一致）：以细级 soft-argmax μ（归一化窗口坐标，带梯度）为均值的高斯，
协方差 = σ_i² I + σ_g² 11ᵀ（每个坐标轴分别）：σ_i 是逐匹配独立噪声，σ_g 是**全体匹配共享的整体平移**。
共享项是为「整体偏移」设计的探索（诊断票 #23：3–10 px 失败多为逐对共同偏移）；只有独立噪声时 K 组仿射几乎相同，
整对 reward 分不出好坏。log π 的梯度 Σ⁻¹(a − μ) 用 Sherman–Morrison 精确算。推理直接用 μ，零额外开销。

reward：
- 整对（pair）：采样匹配 → 仿射 RANSAC（与评测同口径）→ A_k；光学 CFOG 特征按 A_k warp 到 SAR 网格，
  与 SAR 的 CFOG 取平均余弦。组内标准化成 advantage。RANSAC 失败的样本给组内最低分。
- 逐匹配（match）：到 A_k 的残差 d（原网格 px），d < τ 时 exp(−d²/2δ²)，否则 −0.1；以同一匹配 K 次的均值为 baseline。
坐标约定同 finetune/pseudo.py。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from baselines.ransac import fit_affine

from .pseudo import apply_affine, in_to_orig

NORI = 9
MARGIN = 24     # 算相似度时去掉的边框（输入网格 px）
DELTA, TAU, OUT_R = 1.5, 3.0, -0.1


def _gauss(sigma, dev):
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _blur(x, sigma):
    g = _gauss(sigma, x.device)
    C, r = x.shape[1], len(g) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


@torch.no_grad()
def cfog(img):
    """img (B,1,H,W) → (B,9,H,W)，逐像素 L2 归一化。与 baselines/diagnose_reward.py 的 numpy 版同一定义。"""
    img = _blur(img.float(), 1.0)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device).view(1, 1, 3, 3)
    p = F.pad(img, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(p, kx), F.conv2d(p, kx.transpose(2, 3))
    th = torch.arange(NORI, device=img.device) * math.pi / NORI
    Fm = (gx * th.cos().view(1, -1, 1, 1) + gy * th.sin().view(1, -1, 1, 1)).abs()
    Fm = _blur(Fm, 0.8)
    Fm = 0.25 * Fm.roll(1, 1) + 0.5 * Fm + 0.25 * Fm.roll(-1, 1)
    return Fm / (Fm.norm(dim=1, keepdim=True) + 1e-6)


@torch.no_grad()
def gradmag(img):
    """img (B,1,H,W) → 梯度幅值 (B,1,H,W)。与 baselines/diagnose_reward.py 的 gradncc 同一定义。"""
    img = _blur(img.float(), 1.0)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device).view(1, 1, 3, 3)
    p = F.pad(img, (1, 1, 1, 1), mode="replicate")
    return (F.conv2d(p, kx) ** 2 + F.conv2d(p, kx.transpose(2, 3)) ** 2).sqrt()


FEATS = {"cfog": cfog, "gradncc": gradmag}


@torch.no_grad()
def pair_scores(F0, F1, As, s, kind="cfog"):
    """F0/F1: (C,H,W) 光学 / SAR 的特征（输入网格）；As: 原网格角点约定的 2×3 列表。
    kind=cfog：重叠区平均余弦；gradncc：重叠区 NCC。诊断（修正后）：Val 上 gradncc 的峰离真值更近，见 PR 说明。"""
    C, H, W = F1.shape
    Ms = []
    for A in As:
        L, t = A[:, :2], A[:, 2] / s                       # 角点约定下缩放只改平移
        Ac = np.c_[L, t + L @ [0.5, 0.5] - 0.5]            # → 输入网格中心约定
        Ms.append(np.linalg.inv(np.r_[Ac, [[0, 0, 1]]])[:2])
    M = torch.tensor(np.stack(Ms), dtype=torch.float32, device=F1.device)    # SAR 像素 → 光学像素
    ys, xs = torch.meshgrid(torch.arange(H, device=F1.device, dtype=torch.float32),
                            torch.arange(W, device=F1.device, dtype=torch.float32), indexing="ij")
    P = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(-1, 3)
    src = torch.einsum("kij,nj->kni", M, P)                                  # (K, HW, 2)
    grid = torch.stack([2 * src[..., 0] / (W - 1) - 1, 2 * src[..., 1] / (H - 1) - 1], -1).view(len(As), H, W, 2)
    warped = F.grid_sample(F0.expand(len(As), -1, -1, -1), grid, align_corners=True, padding_mode="zeros")
    inside = (grid.abs() <= 1).all(-1)
    inside[:, :MARGIN] = False; inside[:, -MARGIN:] = False
    inside[:, :, :MARGIN] = False; inside[:, :, -MARGIN:] = False
    w = inside.float()
    n = w.sum((1, 2)).clamp_min(1)
    if kind == "cfog":
        cos = (warped * F1[None]).sum(1)
        return ((cos * w).sum((1, 2)) / n).cpu().numpy()
    a, b = warped[:, 0], F1[0].expand_as(warped[:, 0])
    ma, mb = (a * w).sum((1, 2)) / n, (b * w).sum((1, 2)) / n
    a, b = (a - ma[:, None, None]) * w, (b - mb[:, None, None]) * w
    return ((a * b).sum((1, 2)) / ((a * a).sum((1, 2)) * (b * b).sum((1, 2))).sqrt().clamp_min(1e-9)).cpu().numpy()


def rl_loss(data, s, K=4, sig_g=0.25, sig_i=0.1, w_pair=1.0, w_match=0.0, thr=3.0, feats=None, kind="cfog",
            placebo=False, n_pos=None):
    """一个 batch 的 RL 代理损失。feats = (F0, F1)：batch 的 CFOG（只有 w_pair > 0 时需要）。
    placebo：整对 reward 换成随机数，逐匹配 reward 在匹配之间随机打乱（#51）。n_pos：只用前 n_pos 对（其后是负样本对）。
    返回 (loss, 统计)。"""
    mu_all = data["expec_f"][:, :2]
    dev = mu_all.device
    b_ids = data["b_ids"].cpu().numpy()
    half = (data["W"] // 2) * (data["hw0_i"][0] / data["hw0_f"][0])
    k0_all = data["mkpts0_c"].cpu().numpy().astype(np.float64)
    k1c_all = data["mkpts1_c"].cpu().numpy().astype(np.float64)
    losses, st = [], {"r_pair": [], "r_pair_std": [], "ransac_fail": 0, "r_match": []}
    for b in range(int(data["conf_matrix"].shape[0]) if n_pos is None else n_pos):
        sel = np.nonzero(b_ids == b)[0]
        M = len(sel)
        if M < 8:
            continue
        mu = mu_all[sel]
        eps = sig_g * torch.randn(K, 1, 2, device=dev) + sig_i * torch.randn(K, M, 2, device=dev)
        a = mu.detach()[None] + eps                                           # (K, M, 2)
        k1 = k1c_all[sel][None] + a.cpu().numpy() * half                      # 输入网格
        As, resid = [], np.full((K, M), np.inf)
        for k in range(K):
            Mk = np.c_[in_to_orig(k0_all[sel], s), in_to_orig(k1[k], s), np.ones(M)].astype(np.float32)
            A, _, _ = fit_affine(Mk, thr)
            As.append(A)
            if A is not None:
                resid[k] = np.linalg.norm(apply_affine(A, in_to_orig(k0_all[sel], s)) - in_to_orig(k1[k], s), axis=1)
        ok = np.array([A is not None for A in As])
        st["ransac_fail"] += int((~ok).sum())
        v = a - mu[None]                                                      # 带梯度（经 μ）
        loss_b = mu.sum() * 0
        if w_pair > 0 and ok.sum() >= 2:
            r = np.full(K, np.nan)
            r[ok] = pair_scores(feats[0][b], feats[1][b], [A for A in As if A is not None], s, kind)
            r[~ok] = np.nanmin(r)
            if placebo:   # 安慰剂对照：reward 换成与动作无关的随机数，更新只剩同等幅度的噪声扰动
                r = np.random.default_rng().normal(size=K)
            sd = r.std()
            st["r_pair"].append(float(r[ok].mean())); st["r_pair_std"].append(float(sd))
            # 事后分析用：组内 reward 对共享平移 z 回归出局部坡度；μ 均值看细级输出是否整体移动
            st["z"] = (eps[:, 0, :] * half).cpu().numpy().round(3).tolist()        # 输入网格 px
            st["r"] = [None if not np.isfinite(x) else round(float(x), 6) for x in r]
            st["mu_mean_px"] = (mu.detach().mean(0) * half).cpu().numpy().round(4).tolist()
            if sd > 1e-6:
                adv = torch.tensor((r - r.mean()) / sd, dtype=torch.float32, device=dev)
                if sig_i > 0:
                    c = sig_g ** 2 / (sig_i ** 2 + M * sig_g ** 2)
                    sinv_v = (v - c * v.sum(1, keepdim=True)) / sig_i ** 2       # Σ⁻¹ v，逐轴
                    logp = -0.5 * (v * sinv_v).sum((1, 2)) / M                  # (K,)
                else:
                    # σ_i → 0 的极限：只有共享平移 z，log π = −‖mean_m(a − μ)‖² / 2σ_g²，每个匹配的梯度 = z / (M σ_g²)。
                    # σ_i > 0 时独立噪声项 ε/σ_i² 主导梯度却几乎不影响整对 reward（信号被淹没，见 #27 记录）
                    logp = -0.5 * (v.mean(1) ** 2).sum(-1) / sig_g ** 2
                loss_b = loss_b - w_pair * (adv * logp).mean()
        if w_match > 0 and ok.any():
            rm = np.where(resid < TAU, np.exp(-resid ** 2 / (2 * DELTA ** 2)), OUT_R)[ok]
            if placebo:   # 随机 reward 对照：每组内把 reward 在匹配之间打乱，与该匹配的动作无关
                rng = np.random.default_rng()
                rm = np.stack([rng.permutation(x) for x in rm])
            advm = torch.tensor(rm - rm.mean(0, keepdims=True), dtype=torch.float32, device=dev)
            logpm = -(v[torch.from_numpy(ok).to(dev)] ** 2).sum(-1) / (2 * (sig_i ** 2 + sig_g ** 2))
            loss_b = loss_b - w_match * (advm * logpm).mean()
            st["r_match"].append(float(rm.mean()))
        if not torch.isfinite(loss_b):
            st["nonfinite"] = st.get("nonfinite", 0) + 1
            continue
        losses.append(loss_b)
    loss = torch.stack(losses).sum() if losses else mu_all.sum() * 0
    agg = {k: (round(float(np.mean(v)), 5) if isinstance(v, list) and v and k not in ("z", "r", "mu_mean_px") else v)
           for k, v in st.items()}
    return loss, {("rl_" + k): (None if v == [] else v) for k, v in agg.items()}
