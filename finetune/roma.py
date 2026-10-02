"""RoMa 系底座的微调部件（「【RoMa】微调代码接入与实测」#64；方案见 research/roma-finetune 分支的 README）。

- RomaBase：构造与权重加载复用 baselines 的 RomatchAdapter，训练的网络就是评测的网络。DINOv2 不是参数、始终冻结；
  VGG（encoder.cnn）冻结，只训 decoder。前向要 model.train() 才输出 gm_cls，但全部 BN 保持 eval（bs 1–2）。
- 输入：与适配器同口径（原网格 → cv2 双线性长边 640 → uint8 → PIL bicubic 560 → /255 → ImageNet 归一化），
  灰度复制成 3 通道。训练只跑 560 一遍（同 RoMa 原训练），评测照旧 560 → 864、symmetric。
- 伪标签损失 roma_loss：RoMa 原损失（romatch/losses/robust_loss.py，AnyMatch / MINIMA 的训练配置）照抄重写，
  GT 由 patch 级仿射闭式换算：A = [L|t]（原网格角点约定，光学 → SAR）⇒ 归一化坐标下 A_n = [L | (L−I)·1 + t/256]。
- 类 RIPE 损失 ripe_loss（Q4 的对应物）：本步 scale-1 warp 按 certainty 抽点 → 仿射 RANSAC（3 px）→ 逐像素
  reward（内点 +1、外点 r_out；RANSAC 失败全 r_out；负样本对内点 −1、外点 0）→ 两处闭式期望：
  粗级锚点概率 P_i(k*)（位置）与各尺度 σ(certainty)（取舍）。refiner 的位移不接信号。

坐标约定：RoMa 归一化坐标 x_n ∈ (−1, 1)，像素中心 linspace(−1+1/n, 1−1/n)；原网格角点约定 u = 256·(x_n + 1)。
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baselines.adapters.base import long_side_size
from baselines.adapters.romatch import RomatchAdapter
from baselines.ransac import fit_affine

from .coarse import identity_dist

HW = 512
RES = 560
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LOCAL_DIST = {1: 4, 2: 4, 4: 8, 8: 8}   # train_roma_outdoor.py 的配置；阈值 (2/512)·d·s，512 恰为原网格
CE_WEIGHT, ALPHA, C = 0.01, 0.5, 1e-4


class RomaBase:
    def __init__(self, repo, weights, device="cuda", weights_key=None, long_side=640, train_vgg=False):
        self.adapter = RomatchAdapter(repo, weights, device=device, weights_key=weights_key, long_side=long_side)
        self.model, self.device, self.long_side = self.adapter.model, device, long_side
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        for name, p in self.model.named_parameters():
            p.requires_grad_(name.startswith("decoder.") or (train_vgg and name.startswith("encoder.cnn")))

    def resize(self, img: np.ndarray) -> np.ndarray:
        """原网格 float [0,1] → 560² float [0,1]，逐步照适配器 + romatch match() 的预处理（含 uint8 量化）。"""
        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side, df=8)
        if (hn, wn) != (h, w):
            img = cv2.resize(np.ascontiguousarray(img, np.float32), (wn, hn))
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(u8).resize((RES, RES), Image.BICUBIC)
        return np.asarray(pil, np.float32) / 255.0

    def forward(self, image0, image1):
        """image0/1: (B,1,560,560) float [0,1]。返回 romatch 的 corresps（训练态：含 gm_cls、gm_certainty）。"""
        norm = lambda x: (x.expand(-1, 3, -1, -1) - MEAN.to(x)) / STD.to(x)
        return self.model({"im_A": norm(image0), "im_B": norm(image1)}, batched=True)

    def state_dict(self):
        """{'model': ...}，RomatchAdapter(weights_key='model') 直接能读。"""
        return {"model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}


def affine_norm(A):
    """原网格角点约定的 2×3 仿射 → 归一化坐标下的 2×3。"""
    A = np.asarray(A, np.float64).reshape(2, 3)
    L = A[:, :2]
    return np.c_[L, (L - np.eye(2)) @ np.ones(2) + A[:, 2] / (HW / 2)]


def inv_affine(A):
    A = np.asarray(A, np.float64).reshape(2, 3)
    Li = np.linalg.inv(A[:, :2])
    return np.c_[Li, -Li @ A[:, 2]]


def grid(h, w, device):
    ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device)
    xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device)
    return torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)   # (h,w,2)，(x,y)


def affine_gt(An, h, w):
    """An: (B,2,3) 归一化仿射 → x2 (B,h,w,2)、prob (B,h,w)（落在 B 图内为 1）。"""
    g = grid(h, w, An.device)
    x2 = torch.einsum("bij,hwj->bhwi", An[:, :, :2], g) + An[:, None, None, :, 2]
    return x2, (x2.abs() < 1).all(-1).float()


def roma_loss(corresps, An, valid=None):
    """RoMa 原损失，GT 换成仿射。An: (B,2,3) 归一化；valid: (B,) bool，False 的对 prob 全 0（负样本对只剩 certainty BCE）。"""
    tot, st, prev_epe = 0.0, {}, None
    for s in sorted(corresps, reverse=True):          # 16, 8, 4, 2, 1
        c = corresps[s]
        flow, cert = c["flow"], c["certainty"]
        h, w = flow.shape[-2:]
        x2, prob = affine_gt(An, h, w)
        if valid is not None:
            prob = prob * valid[:, None, None].to(prob)
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


def ripe_loss(corresps, neg=None, thr=3.0, r_out=-0.25, num=5000, w_cert=1.0, placebo=False, rng=None, cert_out=None):
    """类 RIPE（Q4 对应物）。neg: 长度 B 的 bool。cert_out：取舍项里正样本对外点的分值，None = 同 r_out
    （r_out < 0 时「certainty 全 0」是吸收态，M5 实测 450 步内塌缩；取 0 则只奖励内点）。返回 (loss, 统计)。"""
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
    agg = lambda xs, k: round(float(np.mean([x[k] for x in xs if x[k] is not None])), 4) if any(
        x[k] is not None for x in xs) else None
    st = {"cexp": round(float(l_cls), 6), "lcert": round(float(l_cert), 6)}
    for tag, xs in (("", [p for p in per if not p["neg"]]), ("neg_", [p for p in per if p["neg"]])):
        if xs:
            st.update({tag + k: agg(xs, k) for k in ("n_inl", "inl_frac", "dist_I")})
    return loss, st
