"""SCENES 式伪标签：用模型自己匹配估出的仿射当几何真值，按上游 LoFTR 的原始监督形式出粗、细两级目标。

对应票「复现 SCENES 式伪标签微调 baseline」#26。SCENES（arXiv:2401.10886）用极线几何做监督；
我们的几何模型是仿射，点到点的对应是确定的，所以直接套用 LoFTR 原始（深度 + 位姿 GT）监督的形式，
只把 GT 换成伪仿射：

- 伪仿射：细级匹配（推理同款）→ 原网格 → baselines.ransac.fit_affine（与评测同口径，3 px）。
  SCENES 的做法是**离线**：用预训练模型估一次，训练中不重估，并筛掉匹配 < 100 或内点 < 20 的对
  （finetune/label.py）。另可选**在线**：每步用当前模型自己的匹配重估（实测会塌缩，见 runs/S2）。
- 粗级：光学侧每个粗格中心经伪仿射落到 SAR 侧，取最近的粗格 j，(i, j) 为正样本；落在图外的 i 不监督。
  损失同上游 dual-softmax + sparse_spvs：只对正样本做 focal loss（loftr_loss.py compute_coarse_loss）。
- 细级：对推理同款的每个粗匹配，SAR 侧目标 = 伪仿射(mkpts0_c)，换成 5×5 窗口的归一化偏移；
  |偏移|∞ < 1（落在窗口内）的才监督。损失同上游 l2_with_std。

坐标约定：网络输入网格（长边 640）上「整数 = 像素中心」；原网格 512；fit_affine 返回的 A 是角点约定
（两侧 +0.5 后估计，见 baselines/fit.py）。
"""
from __future__ import annotations

import numpy as np
import torch

from baselines.ransac import fit_affine

FOCAL_ALPHA, FOCAL_GAMMA = 0.25, 2.0   # 上游 default.py 的 LOSS.FOCAL_ALPHA / FOCAL_GAMMA
FINE_CORRECT_THR = 1.0                 # 上游 LOSS.FINE_CORRECT_THR：归一化窗口坐标


def in_to_orig(p, s):
    """输入网格 → 原网格（中心约定），s = 原尺寸 / 输入尺寸。与 baselines.adapters.base.to_original 相同。"""
    return (p + 0.5) * s - 0.5


def orig_to_in(p, s):
    return (p + 0.5) / s - 0.5


def apply_affine(A, p):
    """A: 2×3（角点约定）；p: N×2 中心约定 → N×2 中心约定。"""
    q = p + 0.5
    return q @ A[:, :2].T + A[:, 2] - 0.5


def fit_pseudo(kp0, kp1, s, thr=3.0, min_inliers=0):
    """输入网格上的一对匹配点集 → 原网格上的伪仿射。返回 (A 或 None, 内点数)。"""
    M = np.c_[in_to_orig(kp0, s), in_to_orig(kp1, s), np.ones(len(kp0))].astype(np.float32)
    A, inl, _ = fit_affine(M, thr)
    n = int(inl.sum()) if inl is not None else 0
    if A is None or n < min_inliers:
        return None, n
    return A, n


def coarse_targets(A, hw_c, scale_c, s):
    """一对的粗级正样本。hw_c: 粗网格 (h, w)；scale_c: 输入网格 / 粗网格（8）。
    返回 (i_ids, j_ids) 两个 int64 数组：光学格 i 经伪仿射落到的 SAR 格 j（最近格），图外的不要。"""
    h, w = hw_c
    ii = np.arange(h * w)
    p0 = np.stack([ii % w, ii // w], 1).astype(np.float64) * scale_c        # 与 mkpts0_c 同一套格点坐标
    p1 = orig_to_in(apply_affine(A, in_to_orig(p0, s)), s) / scale_c
    j = np.rint(p1).astype(np.int64)
    ok = (j[:, 0] >= 0) & (j[:, 0] < w) & (j[:, 1] >= 0) & (j[:, 1] < h)
    return ii[ok], j[ok, 1] * w + j[ok, 0]


def fine_targets(A, mkpts0_c, mkpts1_c, half_window_px, s):
    """每个粗匹配的细级目标：归一化窗口坐标 (M, 2)。half_window_px = (W // 2) × 细网格步长（输入网格像素）。"""
    t = orig_to_in(apply_affine(A, in_to_orig(mkpts0_c, s)), s)
    return (t - mkpts1_c) / half_window_px


def coarse_focal(conf, b, i, j):
    """上游 sparse_spvs + dual_softmax：只对正样本 focal。conf: (B, L, S)。"""
    if len(b) == 0:
        return conf.sum() * 0
    p = conf[b, i, j].clamp(1e-6, 1 - 1e-6)
    return (-FOCAL_ALPHA * (1 - p) ** FOCAL_GAMMA * p.log()).mean()


def fine_l2_std(expec_f, gt, mask):
    """上游 l2_with_std：权重 = 1/std 归一化后 detach，只算 mask（窗口内且伪仿射有效）的匹配。"""
    if not bool(mask.any()):
        return expec_f.sum() * 0
    inv = 1.0 / expec_f[:, 2].clamp_min(1e-10)
    wgt = (inv / inv.mean()).detach()
    l2 = ((gt[mask] - expec_f[mask, :2]) ** 2).sum(-1)
    return (l2 * wgt[mask]).mean()


def inlier_cells(A, k0, k1, i_ids, s, thr):
    """本步匹配里相对伪仿射残差 < thr（原网格 px）的那些光学粗格。"""
    d = np.linalg.norm(apply_affine(A, in_to_orig(k0, s)) - in_to_orig(k1, s), axis=1)
    return np.unique(i_ids[d < thr])


def pseudo_loss(data, s, affines=None, thr=3.0, min_inliers=0, w_coarse=1.0, w_fine=1.0, coarse_set="all", n_pos=None):
    """一个 batch 的伪标签损失。data 是 LoFTR 前向后的字典（模块 eval、开梯度，见 finetune/model.py）。
    s = 原尺寸 / 输入尺寸（标量）。affines：离线伪仿射，每对一个 2×3 或 None（None 的对不监督）；
    不给则用本步匹配在线估计。coarse_set：all = 所有落在图内的光学格（上游 LoFTR 形式）；
    inliers = 只取本步内点所在的格。n_pos：只监督前 n_pos 对（其后是负样本对，#51）。返回 (loss, 统计 dict)。"""
    conf, expec_f = data["conf_matrix"], data["expec_f"]
    B = conf.shape[0]
    hw_c = tuple(int(x) for x in data["hw0_c"])
    scale_c = data["hw0_i"][0] / data["hw0_c"][0]
    half = (data["W"] // 2) * (data["hw0_i"][0] / data["hw0_f"][0])

    b_ids = data["b_ids"].cpu().numpy()
    i_all = data["i_ids"].cpu().numpy()
    k0 =data["mkpts0_c"].cpu().numpy().astype(np.float64)
    k1c = data["mkpts1_c"].cpu().numpy().astype(np.float64)
    k1f = data["mkpts1_f"].detach().cpu().numpy().astype(np.float64)

    cb, ci, cj = [], [], []
    gt = np.zeros((len(b_ids), 2))
    valid = np.zeros(len(b_ids), bool)
    n_inl, used = [], 0
    for k in range(B if n_pos is None else n_pos):
        sel = b_ids == k
        if affines is None:
            A, n = fit_pseudo(k0[sel], k1f[sel], s, thr, min_inliers)
            n_inl.append(n)
        else:
            A = affines[k]
        if A is None:
            continue
        used += 1
        i, j = coarse_targets(A, hw_c, scale_c, s)
        if coarse_set == "inliers":   # 只监督本步内点所在的格：全格子稠密监督在在线模式下会塌（S2）
            keep = np.isin(i, inlier_cells(A, k0[sel], k1f[sel], i_all[sel], s, thr))
            i, j = i[keep], j[keep]
        cb.append(np.full(len(i), k)); ci.append(i); cj.append(j)
        g = fine_targets(A, k0[sel], k1c[sel], half, s)
        gt[sel] = g
        valid[sel] = np.abs(g).max(1) < FINE_CORRECT_THR

    dev = conf.device
    cat = lambda xs: torch.from_numpy(np.concatenate(xs) if xs else np.zeros(0, np.int64)).to(dev)
    l_c = coarse_focal(conf, cat(cb), cat(ci), cat(cj))
    if len(b_ids):
        l_f = fine_l2_std(expec_f, torch.from_numpy(gt).to(expec_f), torch.from_numpy(valid).to(dev))
    else:
        l_f = conf.sum() * 0
    loss = w_coarse * l_c + w_fine * l_f
    return loss, {"coarse": float(l_c), "fine": float(l_f), "n_inliers": n_inl, "pairs_used": used,
                  "n_coarse_pos": int(sum(len(x) for x in ci)), "n_fine": int(valid.sum()), "n_match": len(b_ids)}
