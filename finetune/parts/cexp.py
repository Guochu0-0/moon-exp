"""粗级 ±1 期望（「【类 RIPE】粗级闭式期望从 zero-shot 单独训练：是否塌缩」#49；外点分值 #51；RoMa 版 #66）。

按 RIPE++ 匹配器的写法（ℒ_match = −E[R]，闭式期望、不抽样）：每步用当前模型的匹配跑仿射 RANSAC（与评测同口径），
内点 +1、外点 r_out（RIPE++ 为 −1）；RANSAC 失败的对全部记 r_out。负样本对（[neg]）上内点 −1、外点 0。
placebo（随机分数对照）：分数在同一对的匹配 / 像素之间随机打乱，数量分布不变、与匹配好坏无关；负样本对不受影响。

LoFTR：loss = −Σ_{(i,j) ∈ 本步粗匹配} P(i,j) · r(i,j) / L。粗匹配 = 推理同款（阈值 + MNN）；P = conf_matrix（带梯度）；
  r 来自本步细级匹配的 RANSAC；只对实际输出的匹配给分，其余格不监督（同 RIPE++）；L = 粗格数，常数归一化。
RoMa：本步 scale-1 warp 按 certainty 抽 num 个点 → RANSAC → 逐像素 r → 两处闭式期望：粗级锚点概率 P_i(k*)（位置）
  与各尺度 σ(certainty)（取舍，权重 w_cert；正样本对外点的分值 cert_out，默认同 r_out）。refiner 的位移不接信号。

监控（每对）：匹配数、内点数、估出的仿射相对 [I|0] 的角点平均位移（原网格 px）；LoFTR 另有恒等匹配数
（|k1 − k0| < 阈值）、粗级 dual-softmax 对角元 conf[i,i] 的均值；RoMa 另有内点像素比例。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from moonlib.ransac import fit_affine

from ..geom import HW, identity_dist, in_to_orig
from .pseudo import affine_norm, grid

PARAMS = {"weight": 1.0,
          "r_out": -1.0,        # 正样本对上外点的分值
          "ransac": 3.0,        # RANSAC 阈值（原网格 px），与评测同口径
          "placebo": False}     # 随机分数对照
MODELS = {"loftr": {},
          "roma": {"w_cert": 1.0, "cert_out": None, "num": 5000}}


def fill(p):
    if "cert_out" in p and p["cert_out"] is None:
        p["cert_out"] = p["r_out"]
    return p


def check(cfg):
    pass


def data_kw(p):
    return {}


class Term:
    def __init__(self, p, model, ds):
        self.p, self.model = p, model

    def __call__(self, st):
        p = self.p
        neg = [False] * st.B + [True] * st.B if st.neg else None
        if st.model_name == "loftr":
            loss, stats = loftr_loss(st.out, self.model.s, neg, p["ransac"], p["placebo"], st.rng, p["r_out"])
        else:
            loss, stats = roma_loss(st.out, neg, p["ransac"], p["r_out"], p["num"], p["w_cert"], p["placebo"], st.rng,
                                    p["cert_out"])
        return p["weight"] * loss, stats, loss.requires_grad


# ---------- LoFTR ----------

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


def _agg(xs, k):
    vals = [x[k] for x in xs if x[k] is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def loftr_loss(data, s, neg=None, thr=3.0, placebo=False, rng=None, r_out=-1.0):
    """一个 batch 的粗级 ±1 期望。neg: 长度 B 的 bool，True 为负样本对。返回 (loss, 统计 dict)。"""
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
    st = {"cexp": round(float(loss), 6)}
    for tag, xs in (("", [p for p in per if not p["neg"]]), ("neg_", [p for p in per if p["neg"]])):
        if xs:
            st.update({tag + k: _agg(xs, k) for k in ("n_match", "n_inl", "n_ident", "dist_I", "diag")})
    return loss, st


# ---------- RoMa ----------

def sample_matches(flow, cert, num=5000, thresh=0.05, gen=None):
    """一对的 scale-1 warp (2,h,w) 与 certainty logits (1,h,w) → 按推理同款的阈值化 certainty 抽 num 个点。
    返回原网格中心约定的 (p0, p1) 与所抽像素的索引（不做 KDE 平衡）。"""
    h, w = flow.shape[-2:]
    p = cert[0].float().sigmoid().reshape(-1).clone()
    p[p > thresh] = 1
    idx = torch.multinomial(p, min(num, len(p)), replacement=False, generator=gen)
    g = grid(h, w, flow.device).reshape(-1, 2)
    to_px = lambda x: ((x + 1) * (HW / 2) - 0.5).double().cpu().numpy()
    return to_px(g[idx]), to_px(flow.reshape(2, -1).T[idx].float()), idx


def roma_loss(corresps, neg=None, thr=3.0, r_out=-1.0, num=5000, w_cert=1.0, placebo=False, rng=None, cert_out=None):
    """neg: 长度 B 的 bool。cert_out：取舍项里正样本对外点的分值，None = 同 r_out（r_out < 0 时「certainty 全 0」
    是吸收态，M5 实测 450 步内塌缩；取 0 则只奖励内点）。返回 (loss, 统计)。"""
    f1, c1 = corresps[1]["flow"], corresps[1]["certainty"]
    B, _, h, w = f1.shape
    neg = np.zeros(B, bool) if neg is None else np.asarray(neg, bool)
    rng = rng or np.random.default_rng()
    g = grid(h, w, f1.device)
    R = torch.zeros(B, h, w, device=f1.device)
    Rc = torch.zeros(B, h, w, device=f1.device)
    per = []
    for b in range(B):
        with torch.no_grad():
            p0, p1, _ = sample_matches(f1[b].detach(), c1[b].detach(), num)
            A, inl, _ = fit_affine(np.c_[p0, p1, np.ones(len(p0))].astype(np.float32), thr)
            if A is None:
                inl_px = torch.zeros(h, w, dtype=torch.bool, device=f1.device)
            else:
                An = torch.from_numpy(affine_norm(A)).to(f1)
                pred = torch.einsum("ij,hwj->hwi", An[:, :2], g) + An[:, 2]
                inl_px = ((f1[b].detach().permute(1, 2, 0).float() - pred).norm(dim=-1) * (HW / 2)) < thr
            if neg[b]:
                r = -inl_px.float()
            else:
                r = torch.where(inl_px, 1.0, r_out)
                if placebo:
                    r = r.reshape(-1)[torch.from_numpy(rng.permutation(h * w)).to(r.device)].reshape(h, w)
            R[b] = r
            Rc[b] = r if (cert_out is None or neg[b]) else torch.where(r > 0, r, torch.full_like(r, cert_out))
            n_inl = 0 if inl is None else int(inl.sum())
            per.append({"neg": bool(neg[b]), "n_inl": n_inl, "inl_frac": round(float(inl_px.float().mean()), 4),
                        "dist_I": None if A is None else round(identity_dist(A), 3)})
    # 粗级：锚点概率 P_i(k*)，k* = argmax（与 cls_to_flow_refine 的 mode 同）；r̄ = 该格内像素 reward 的均值
    cls = corresps[16]["gm_cls"].float()
    P = cls.softmax(1).max(1).values                                        # (B,40,40)
    rbar = F.adaptive_avg_pool2d(R[:, None], P.shape[-2:])[:, 0]
    l_cls = -(P * rbar).mean()
    # 取舍：各尺度 σ(certainty) · reward（最近邻下采样）
    l_cert = torch.zeros((), device=f1.device)
    for s, c in (corresps.items() if w_cert > 0 else ()):
        cert = c["certainty"][:, 0].float()
        rs = F.interpolate(Rc[:, None], size=cert.shape[-2:], mode="nearest-exact")[:, 0]
        l_cert = l_cert - (cert.sigmoid() * rs).mean()
    loss = l_cls + w_cert * l_cert
    st = {"cexp": round(float(l_cls), 6), "lcert": round(float(l_cert), 6)}
    for tag, xs in (("", [p for p in per if not p["neg"]]), ("neg_", [p for p in per if p["neg"]])):
        if xs:
            st.update({tag + k: _agg(xs, k) for k in ("n_inl", "inl_frac", "dist_I")})
    return loss, st
