"""粗级闭式期望（「【类 RIPE】粗级闭式期望从 zero-shot 单独训练：是否塌缩」#49）。

按 RIPE++ 匹配器的写法（ℒ_match = −E[R]，闭式期望、不抽样）放到 AnyMatch-LoFTR 粗级 dual-softmax 上：

    loss = −Σ_{(i,j) ∈ 本步粗匹配} P(i,j) · r(i,j) / L

- 粗匹配 = 推理同款（阈值 + MNN）的 b_ids/i_ids/j_ids；P = conf_matrix（带梯度）。
- r：用本步细级匹配跑仿射 RANSAC（与评测同口径，3 px），内点 +1、外点 −1；RANSAC 失败的对全部 −1。
  只对实际输出的匹配给分，其余格不监督（同 RIPE++）。
- L = 粗格数（光学侧 h·w），常数归一化：保留 RIPE++「内点越多 reward 越大」的语义。
- placebo（随机 reward 对照）：r 在同一对的匹配之间随机打乱，数量分布不变，与匹配好坏无关。
- 负样本对（光学与 SAR 来自不同 ROI，#48 建议 ②）：RANSAC 内点给 −1，外点 0；不受 placebo 影响。

监控（每对）：匹配数、内点数、估出的仿射相对 [I|0] 的角点平均位移（原网格 px）、
恒等匹配数（|k1 − k0| < thr，原网格 px）、粗级 dual-softmax 对角元 conf[i,i] 的均值。
坐标约定同 finetune/pseudo.py。
"""
from __future__ import annotations

import numpy as np
import torch

from baselines.ransac import fit_affine

from .pseudo import in_to_orig

HW = 512
CORNERS = np.array([[0, 0], [HW, 0], [0, HW], [HW, HW]], np.float64)   # 角点约定


def identity_dist(A):
    """仿射 A（原网格角点约定）相对 [I|0] 的 patch 四角平均位移，px。A 为 None 返回 None。"""
    if A is None:
        return None
    A = np.asarray(A, np.float64).reshape(2, 3)
    return float(np.linalg.norm(CORNERS @ A[:, :2].T + A[:, 2] - CORNERS, axis=1).mean())


def pair_stats(k0, k1, s, thr=3.0):
    """一对的输入网格匹配点 → (A, 内点掩码, 统计)。A/掩码失败时为 None。"""
    p0, p1 = in_to_orig(k0, s), in_to_orig(k1, s)
    M = np.c_[p0, p1, np.ones(len(k0))].astype(np.float32)
    A, inl, _ = fit_affine(M, thr)
    st = {"n_match": len(k0), "n_inl": int(inl.sum()) if inl is not None else 0,
          "n_ident": int((np.linalg.norm(p1 - p0, axis=1) < thr).sum()),
          "dist_I": None if A is None else round(identity_dist(A), 3)}
    return A, inl, st


def diag_mean(conf):
    """conf (B, L, S)，L == S（两图同尺寸）→ 每对 conf[i,i] 的均值。"""
    L = min(conf.shape[1], conf.shape[2])
    idx = torch.arange(L, device=conf.device)
    return conf.detach()[:, idx, idx].mean(1).cpu().numpy()


def coarse_expect_loss(data, s, neg=None, thr=3.0, placebo=False, rng=None, r_out=-1.0):
    """一个 batch 的粗级闭式期望损失。neg: 长度 B 的 bool，True 为负样本对。r_out：正样本对上外点的分值
    （RIPE++ 为 −1；#49 起步塌到「无匹配」的备选解法之一是取绝对值更小的负分，#51）。返回 (loss, 统计 dict)。"""
    conf = data["conf_matrix"]
    B, L = conf.shape[0], conf.shape[1]
    neg = np.zeros(B, bool) if neg is None else np.asarray(neg, bool)
    rng = rng or np.random.default_rng()
    b_ids = data["b_ids"].cpu().numpy()
    i_ids, j_ids = data["i_ids"], data["j_ids"]
    k0 = data["mkpts0_f"].detach().cpu().numpy().astype(np.float64)
    k1 = data["mkpts1_f"].detach().cpu().numpy().astype(np.float64)
    r = np.zeros(len(b_ids), np.float32)
    dg = diag_mean(conf)
    per = []
    for b in range(B):
        sel = np.nonzero(b_ids == b)[0]
        A, inl, st = pair_stats(k0[sel], k1[sel], s, thr) if len(sel) else (None, None, {
            "n_match": 0, "n_inl": 0, "n_ident": 0, "dist_I": None})
        if neg[b]:
            rb = np.zeros(len(sel), np.float32) if inl is None else -inl.astype(np.float32)
        else:
            rb = np.full(len(sel), r_out, np.float32) if inl is None else np.where(inl, 1.0, r_out).astype(np.float32)
            if placebo:
                rb = rng.permutation(rb)
        r[sel] = rb
        per.append({**st, "diag": round(float(dg[b]), 6), "neg": bool(neg[b])})
    if len(b_ids):
        P = conf[data["b_ids"], i_ids, j_ids]
        loss = -(P * torch.from_numpy(r).to(P)).sum() / (L * B)
    else:
        loss = conf.sum() * 0
    pos = [p for p in per if not p["neg"]]
    ng = [p for p in per if p["neg"]]
    agg = lambda xs, k: (round(float(np.mean([x[k] for x in xs if x[k] is not None])), 4)
                         if any(x[k] is not None for x in xs) else None)
    st = {"cexp": round(float(loss), 6)}
    for tag, xs in (("", pos), ("neg_", ng)):
        if xs:
            st.update({tag + k: agg(xs, k) for k in ("n_match", "n_inl", "n_ident", "dist_I", "diag")})
    return loss, st
