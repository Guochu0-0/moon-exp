"""离线伪标签监督（SCENES 式，「复现 SCENES 式伪标签微调 baseline」#26；迭代打标见 #53）。

伪仿射由 finetune.label 事先用老师模型在 Train 上估好（每对一个，训练中不变），只在 keep 的对上训练；
top < 1 只用内点数最多的前这一比例（#53）。扰动 geo 打开时 SAR 被已知 T warp 过，标签精确变为 T∘A。

LoFTR（SCENES 用极线几何监督；我们的几何模型是仿射，点到点对应是确定的，所以直接套用 LoFTR 原始监督形式，
只把 GT 换成伪仿射）：
- 粗级：光学侧每个粗格中心经伪仿射落到 SAR 侧，取最近的粗格 j，(i, j) 为正样本；落在图外的 i 不监督。
  损失同上游 dual-softmax + sparse_spvs：只对正样本做 focal loss（loftr_loss.py compute_coarse_loss）。
- 细级：对推理同款的每个粗匹配，SAR 侧目标 = 伪仿射(mkpts0_c)，换成 5×5 窗口的归一化偏移；
  |偏移|∞ < 1（落在窗口内）的才监督。损失同上游 l2_with_std。

RoMa：RoMa 原损失（romatch/losses/robust_loss.py，AnyMatch / MINIMA 的训练配置）照抄重写，GT 由 patch 级仿射闭式换算：
A = [L|t]（原网格角点约定，光学 → SAR）⇒ 归一化坐标下 A_n = [L | (L−I)·1 + t/256]。以概率 swap 交换光学 / SAR 方向
（标签取逆），训 B→A。这一项自己做前向（输入可能交换过）。

坐标约定见 finetune/geom.py。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ConfigError
from ..geom import HW, apply_affine, compose, in_to_orig, inv_affine, orig_to_in
from ..label import load_labels

PARAMS = {"labels": "",         # finetune.label 格式的离线伪仿射（jsonl）
          "top": 1.0,           # 只用 keep 的对里内点数最多的前这一比例
          "weight": 1.0}
MODELS = {"loftr": {"w_coarse": 1.0, "w_fine": 1.0},
          "roma": {"swap": 0.5}}

FOCAL_ALPHA, FOCAL_GAMMA = 0.25, 2.0   # 上游 LoFTR default.py 的 LOSS.FOCAL_ALPHA / FOCAL_GAMMA
FINE_CORRECT_THR = 1.0                 # 上游 LOSS.FINE_CORRECT_THR：归一化窗口坐标
LOCAL_DIST = {1: 4, 2: 4, 4: 8, 8: 8}  # romatch train_roma_outdoor.py；阈值 (2/512)·d·s，512 恰为原网格
CE_WEIGHT, ALPHA, C = 0.01, 0.5, 1e-4  # romatch robust_loss 的 certainty 权重与鲁棒回归参数


def fill(p):
    return p


def check(cfg):
    if not cfg["pseudo"]["labels"]:
        raise ConfigError("[pseudo] labels 必填")
    if cfg["model"]["name"] == "loftr" and "neg" in cfg:
        raise ConfigError("LoFTR 上 [neg] 不配合 [pseudo]（负样本对并入同一次前向，伪标签不监督它们）")


def data_kw(p):
    return {}


class Term:
    def __init__(self, p, model, ds):
        self.p, self.model = p, model
        self.labels = load_labels(p["labels"], p["top"])
        ds.pairs = [q for q in ds.pairs if self.labels.get(q) is not None]

    def __call__(self, st):
        As = [np.asarray(self.labels[q]) for q in st.batch["pair"]]
        if "T" in st.batch:   # SAR 被已知 T warp 过：标签精确变为 T∘A
            As = [compose(T.numpy(), A) for T, A in zip(st.batch["T"], As)]
        if st.model_name == "loftr":
            loss, stats = loftr_loss(st.out, self.model.s, As, self.p["w_coarse"], self.p["w_fine"])
            return self.p["weight"] * loss, stats, stats["pairs_used"] > 0
        i0, i1 = st.images()
        sw = st.rng.random(st.B) < self.p["swap"]
        m = torch.from_numpy(sw)[:, None, None, None].to(i0.device)
        a, b = torch.where(m, i1, i0), torch.where(m, i0, i1)
        An = torch.from_numpy(np.stack([affine_norm(inv_affine(A) if s else A) for A, s in zip(As, sw)])
                              ).float().to(i0.device)
        loss, stats = roma_loss(self.model.forward(a, b), An)
        return self.p["weight"] * loss, stats, True


# ---------- LoFTR ----------

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


def loftr_loss(data, s, affines, w_coarse=1.0, w_fine=1.0):
    """一个 batch 的伪标签损失。data 是 LoFTR 前向后的字典（模块 eval、开梯度，见 finetune/models/loftr.py）。
    s = 原尺寸 / 输入尺寸（标量）。affines：每对一个 2×3 或 None（None 的对不监督）。返回 (loss, 统计 dict)。"""
    conf, expec_f = data["conf_matrix"], data["expec_f"]
    hw_c = tuple(int(x) for x in data["hw0_c"])
    scale_c = data["hw0_i"][0] / data["hw0_c"][0]
    half = (data["W"] // 2) * (data["hw0_i"][0] / data["hw0_f"][0])

    b_ids = data["b_ids"].cpu().numpy()
    k0 = data["mkpts0_c"].cpu().numpy().astype(np.float64)
    k1c = data["mkpts1_c"].cpu().numpy().astype(np.float64)

    cb, ci, cj = [], [], []
    gt = np.zeros((len(b_ids), 2))
    valid = np.zeros(len(b_ids), bool)
    used = 0
    for k, A in enumerate(affines):
        if A is None:
            continue
        sel = b_ids == k
        used += 1
        i, j = coarse_targets(A, hw_c, scale_c, s)
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
    return loss, {"coarse": float(l_c), "fine": float(l_f), "n_inliers": [], "pairs_used": used,
                  "n_coarse_pos": int(sum(len(x) for x in ci)), "n_fine": int(valid.sum()), "n_match": len(b_ids)}


# ---------- RoMa ----------

def affine_norm(A):
    """原网格角点约定的 2×3 仿射 → 归一化坐标下的 2×3。"""
    A = np.asarray(A, np.float64).reshape(2, 3)
    L = A[:, :2]
    return np.c_[L, (L - np.eye(2)) @ np.ones(2) + A[:, 2] / (HW / 2)]


def grid(h, w, device):
    ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device)
    xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device)
    return torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)   # (h,w,2)，(x,y)


def affine_gt(An, h, w):
    """An: (B,2,3) 归一化仿射 → x2 (B,h,w,2)、prob (B,h,w)（落在 B 图内为 1）。"""
    g = grid(h, w, An.device)
    x2 = torch.einsum("bij,hwj->bhwi", An[:, :, :2], g) + An[:, None, None, :, 2]
    return x2, (x2.abs() < 1).all(-1).float()


def roma_loss(corresps, An):
    """RoMa 原损失，GT 换成仿射。An: (B,2,3) 归一化。"""
    tot, st, prev_epe = 0.0, {}, None
    for s in sorted(corresps, reverse=True):          # 16, 8, 4, 2, 1
        c = corresps[s]
        flow, cert = c["flow"], c["certainty"]
        h, w = flow.shape[-2:]
        x2, prob = affine_gt(An, h, w)
        if s <= 8 and prev_epe is not None:
            prob = prob * (F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                           < (2 / 512) * LOCAL_DIST[s] * s)
        m = prob > 0.99
        if "gm_cls" in c:
            cls = c["gm_cls"].float()
            K = cls.shape[1]
            r = round(math.sqrt(K))
            with torch.no_grad():
                G = grid(r, r, cls.device).reshape(K, 2)
                tgt = torch.cdist(x2.reshape(len(x2), -1, 2), G[None].expand(len(x2), -1, -1)).argmin(-1)
                tgt = tgt.reshape(x2.shape[:3])
            ce = F.cross_entropy(cls, tgt, reduction="none")[m]
            l_cls = ce.mean() if ce.numel() else cls.sum() * 0
            l_gc = F.binary_cross_entropy_with_logits(c["gm_certainty"][:, 0].float(), prob)
            tot = tot + l_cls + CE_WEIGHT * l_gc
            st[f"cls{s}"] = round(float(l_cls), 5)
        epe = (flow.permute(0, 2, 3, 1).float() - x2).norm(dim=-1)
        cs = C * s
        x = epe[m]
        l_reg = (cs ** ALPHA * ((x / cs) ** 2 + 1) ** (ALPHA / 2)).mean() if x.numel() else flow.sum() * 0
        l_ce = F.binary_cross_entropy_with_logits(cert[:, 0].float(), prob)
        tot = tot + l_reg + CE_WEIGHT * l_ce
        st[f"reg{s}"] = round(float(l_reg), 5)
        if s == 1:
            st["epe1_px"] = round(float(epe[m].median() * HW / 2), 3) if m.any() else None
        prev_epe = epe.detach()
    return tot, st
