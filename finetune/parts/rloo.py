"""粗级锚点采样 + RLOO 策略梯度（「【RoMa·RL】首批实验」#169；策略形式、算法与 reward 见 #165，开跑前检查 #168）。

策略：RoMa 粗级（尺度 16，560 输入时 40×40 格）每格对 64×64 = 4096 个锚点的分类 logits（gm_cls）就是一个类别分布。
推理取 mode，再用 mode 与上下左右 4 个邻居锚点的概率加权平均出粗 warp（romatch cls_to_flow_refine）。训练时每格按
softmax(gm_cls / temp) 独立抽一个锚点 k，把 mode 换成 k、其余照旧算出粗 warp，再按推理同款跑尺度 16…1 的细化
（细化不加噪声、不求梯度），得到一份尺度 1 的 warp 与 certainty。每对抽 group 份。
log π = Σ_格 log softmax(gm_cls / temp)[k]。梯度只经 gm_cls 回到粗级（embedding decoder、GP、proj16、DINOv2 的 LoRA）；
romatch 各级之间本来就 detach，细化层与 VGG 收不到梯度。推理结构不变。

reward（每份样本单独算，只用训练对的两张影像和这份输出）：先按推理同款阈值化 certainty 抽 num 个点，RANSAC（3 px，
与评测同口径）拟合仿射 A，然后
- r1：逐点分级残差 max(0, 1 − (r/3)²)，r 为每个像素的 warp 到 A 的距离（原网格 px）；取每个粗格内像素的均值，逐格给分；
- ncc：用 A 把 SAR 拉到光学网格，在重叠区上比两图灰度的 NCC，整对一个数；SAR 的补零区（像素值 < 1/255，再外扩 4 px）不计；
- rand：每格独立 U(0, 1)（随机对照，与 r1 一样逐格给分）。
RANSAC 失败的样本：r1 记 0，ncc 记 −1。

RLOO（Ahmadian 等 2402.14740 的留一基线）：优势 = 本份 reward − 同一对其余 group − 1 份的均值（逐格 reward 逐格算），
不除标准差；loss = −mean_{份, 格} 优势 · log π。不训 certainty、不加 KL。
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from moonlib.ransac import fit_affine

from ..config import ConfigError
from ..geom import HW
from .cexp import sample_matches
from .pseudo import affine_norm, grid

PARAMS = {"reward": "r1",   # r1 | ncc | rand
          "group": 16,      # 每对的样本份数 G
          "temp": 1.0,      # 采样温度
          "num": 5000,      # 每份样本抽来拟合仿射的点数（同 [cexp] RoMa）
          "ransac": 3.0,    # RANSAC 阈值与 r1 的残差尺度（原网格 px）
          "chunk": 8,       # 细化一次并行跑几份样本（只影响显存）
          "weight": 1.0}
MODELS = {"roma": {}}
REWARDS = ("r1", "ncc", "rand")
SCALES = ("16", "8", "4", "2", "1")


def fill(p):
    return p


def check(cfg):
    p = cfg["rloo"]
    if p["reward"] not in REWARDS:
        raise ConfigError(f"[rloo] reward 可选 {REWARDS}，得到 {p['reward']!r}")
    if p["group"] < 2:
        raise ConfigError("[rloo] group 至少为 2（留一基线）")
    if p["temp"] <= 0:
        raise ConfigError("[rloo] temp 应 > 0")
    for other in ("pseudo", "cexp", "neg", "warpc"):
        if other in cfg:
            raise ConfigError(f"[rloo] 不与 [{other}] 同用")
    if cfg["model"]["train_res"] != "560":
        raise ConfigError("[rloo] 只支持 [model] train_res = '560'")


def data_kw(p):
    return {}


# ---------- 前向：粗级（带梯度）与细化（照抄 romatch Decoder.forward 的非 upsample 分支） ----------

def _autocast(dec, x):
    from romatch.utils.utils import get_autocast_params
    dev, on, dt = get_autocast_params(x.device, str(x) == "cuda", dec.amp_dtype)
    return torch.autocast(dev, enabled=on, dtype=dt), on


def features(model, im0, im1):
    """编码器 + 各尺度 proj。返回 (f1, f2)：尺度 → proj 之后的特征（与 Decoder.forward 里的 f1_s、f2_s 相同）。"""
    from finetune.models.roma import _norm
    m = model.model
    fp = m.extract_backbone_features({"im_A": _norm(im0), "im_B": _norm(im1)}, batched=True)
    dec = m.decoder
    f1, f2 = {}, {}
    for s in SCALES:
        a, b = fp[int(s)].chunk(2)
        ac, on = _autocast(dec, a)
        with ac:
            if not on:
                a, b = a.float(), b.float()
            f1[int(s)], f2[int(s)] = dec.proj[s](a), dec.proj[s](b)
    return f1, f2


def coarse(model, f1, f2):
    """尺度 16：GP + embedding decoder。返回 gm_cls (B,4096,h,w) 与粗级 certainty (B,1,h,w)。"""
    dec = model.model.decoder
    a, b = f1[16], f2[16]
    old = torch.zeros(a.shape[0], dec.embedding_decoder.hidden_dim, *a.shape[-2:], device=a.device)
    gp = dec.gps["16"](a, b)
    cls, cert, _ = dec.embedding_decoder(gp, a, old, "16")
    return cls, cert


def anchors(C, device):
    res = round(math.sqrt(C))
    g = torch.meshgrid(*[torch.linspace(-1 + 1 / res, 1 - 1 / res, res, device=device) for _ in range(2)],
                       indexing="ij")
    return torch.stack([g[1], g[0]], -1).reshape(C, 2), res


def anchor_flow(P, k):
    """P：softmax(gm_cls) (B,C,h,w)；k：(B,h,w) 每格选中的锚点 → 粗 warp (B,2,h,w)。
    同 romatch cls_to_flow_refine，只是 mode 换成 k；k = argmax 时与之相同。"""
    G, res = anchors(P.shape[1], P.device)
    idx = torch.stack((k - 1, k, k + 1, k - res, k + res), 1).clamp(0, P.shape[1] - 1).long()   # (B,5,h,w)
    nb = torch.gather(P, 1, idx)[..., None]                                                    # (B,5,h,w,1)
    flow = (nb * G[idx]).sum(1) / nb.sum(1)                                                    # (B,h,w,2)
    return flow.permute(0, 3, 1, 2)


@torch.no_grad()
def refine(model, f1, f2, flow, certainty):
    """从粗 warp 与粗级 certainty 起，跑尺度 16…1 的细化（同 Decoder.forward：每级加位移与 certainty 增量，
    上采样到下一级）。f1、f2 的 batch 可以是 1（自动扩到 flow 的 batch）。返回尺度 1 的 (flow, certainty)。"""
    dec = model.model.decoder
    n = flow.shape[0]
    h, w = f1[1].shape[-2:]
    for s in SCALES:
        ins = int(s)
        a, b = f1[ins], f2[ins]
        if a.shape[0] != n:
            a, b = a.expand(n, -1, -1, -1), b.expand(n, -1, -1, -1)
        d_flow, d_cert = dec.conv_refiner[s](a, b, flow, scale_factor=1, logits=certainty)
        disp = ins * torch.stack((d_flow[:, 0].float() / (dec.refine_init * w),
                                  d_flow[:, 1].float() / (dec.refine_init * h)), dim=1)
        flow, certainty = flow + disp, certainty + d_cert
        if s != "1":
            size = f1[ins // 2].shape[-2:]
            flow = F.interpolate(flow, size=size, mode=dec.flow_upsample_mode)
            certainty = F.interpolate(certainty, size=size, mode=dec.flow_upsample_mode)
    return flow, certainty


def rollout(model, f1, f2, P, cert, k, chunk=8):
    """一对：P (1,C,h,w)、cert (1,1,h,w)、k (G,h,w) → G 份尺度 1 的 (flow (G,2,H,W), certainty (G,1,H,W))。"""
    fl, ce = [], []
    for i in range(0, len(k), chunk):
        kk = k[i:i + chunk]
        fc = anchor_flow(P.expand(len(kk), -1, -1, -1), kk)
        a, b = refine(model, f1, f2, fc, cert.expand(len(kk), -1, -1, -1))
        fl.append(a)
        ce.append(b)
    return torch.cat(fl), torch.cat(ce)


# ---------- reward ----------

def fit(flow, cert, num, thr):
    """一份样本：尺度 1 的 warp (2,H,W) 与 certainty (1,H,W) → (仿射 A 或 None, 内点数)。"""
    return fit_many(flow[None], cert[None], num, thr)[0]


_POOL = None


def fit_many(flows, certs, num, thr):
    """G 份样本各自抽点、RANSAC。抽点在 GPU 上依次做，RANSAC 用线程并行（cv2 放开 GIL；fit_affine 每次调用前重置
    本线程的随机种子，结果与串行相同）。返回 [(A 或 None, 内点数)]。"""
    global _POOL
    from concurrent.futures import ThreadPoolExecutor
    if _POOL is None:
        _POOL = ThreadPoolExecutor(8)
    Ms = []
    for f, c in zip(flows, certs):
        p0, p1, _ = sample_matches(f, c, num)
        Ms.append(np.c_[p0, p1, np.ones(len(p0))].astype(np.float32))
    out = list(_POOL.map(lambda M: fit_affine(M, thr), Ms))
    return [(A, 0 if inl is None else int(inl.sum())) for A, inl, _ in out]


def r1_cells(flow, A, thr, hw_c):
    """逐像素分级残差，按粗格取均值 → (h_c, w_c)。A 为 None 时全 0。"""
    if A is None:
        return torch.zeros(hw_c, device=flow.device)
    h, w = flow.shape[-2:]
    An = torch.from_numpy(affine_norm(A)).to(flow.device).float()
    g = grid(h, w, flow.device)
    pred = torch.einsum("ij,hwj->hwi", An[:, :2], g) + An[:, 2]
    res = (flow.permute(1, 2, 0).float() - pred).norm(dim=-1) * (HW / 2)
    r = (1 - (res / thr) ** 2).clamp(min=0)
    return F.adaptive_avg_pool2d(r[None, None], hw_c)[0, 0]


def sar_valid(sar):
    """SAR 输入 (1,H,W) 上非补零区的掩码：像素值 < 1/255 的视为补零，再外扩 4 px。"""
    bad = (sar < 1 / 255).float()[None]
    return F.max_pool2d(bad, 9, 1, 4)[0, 0] < 0.5


def ncc(opt, sar, valid, A, min_px=1000):
    """用 A 把 SAR 拉到光学网格，在重叠区（落在 SAR 画面内且不在补零区）上算灰度 NCC。A 为 None 或重叠太少时为 −1。"""
    if A is None:
        return -1.0
    h, w = opt.shape[-2:]
    An = torch.from_numpy(affine_norm(A)).to(opt.device).float()
    x2 = torch.einsum("ij,hwj->hwi", An[:, :2], grid(h, w, opt.device)) + An[:, 2]
    s = F.grid_sample(sar[None].float(), x2[None], align_corners=False, mode="bilinear")[0, 0]
    v = F.grid_sample(valid[None, None].float(), x2[None], align_corners=False, mode="bilinear")[0, 0]
    m = (x2.abs() < 1).all(-1) & (v > 0.999)
    if int(m.sum()) < min_px:
        return -1.0
    a, b = opt[0][m].float(), s[m]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / ((a * a).sum() * (b * b).sum()).sqrt().clamp(min=1e-12))


def loo(r):
    """留一优势：r (G, ...) → r − 其余 G−1 份的均值。"""
    G = r.shape[0]
    return r - (r.sum(0, keepdim=True) - r) / (G - 1)


# ---------- 训练成分 ----------

class Term:
    def __init__(self, p, model, ds):
        self.p, self.model = p, model

    def __call__(self, st):
        p, model = self.p, self.model
        i0, i1 = st.images()
        f1, f2 = features(model, i0, i1)
        cls, cert = coarse(model, f1, f2)
        cls = cls.float()
        B, C, h, w = cls.shape
        logp_all = F.log_softmax(cls / p["temp"], 1)
        loss = torch.zeros((), device=cls.device)
        stats = {k: [] for k in ("rew", "rew_sd", "n_inl", "H", "pmode", "nonmode", "d_anchor", "t_roll", "t_fit")}
        for b in range(B):
            with torch.no_grad():
                q = logp_all[b].exp()                                            # (C,h,w)
                k = torch.multinomial(q.reshape(C, -1).T, p["group"], replacement=True).T.reshape(-1, h, w)
                P = cls[b:b + 1].softmax(1)
                fb = {s: x[b:b + 1] for s, x in f1.items()}
                gb = {s: x[b:b + 1] for s, x in f2.items()}
                t0 = time.time()
                flow, ce = rollout(model, fb, gb, P, cert[b:b + 1].float(), k, p["chunk"])
                torch.cuda.synchronize()
                t1 = time.time()
                fits = fit_many(flow, ce, p["num"], p["ransac"])
                stats["t_roll"].append(t1 - t0)
                stats["t_fit"].append(time.time() - t1)
                if p["reward"] == "r1":
                    r = torch.stack([r1_cells(flow[g], A, p["ransac"], (h, w)) for g, (A, _) in enumerate(fits)])
                elif p["reward"] == "ncc":
                    valid = sar_valid(i1[b])
                    r = torch.tensor([ncc(i0[b], i1[b], valid, A) for A, _ in fits], device=cls.device)[:, None, None]
                else:
                    r = torch.rand(p["group"], h, w, device=cls.device)
                adv = loo(r).expand(p["group"], h, w)
                mode = q.argmax(0)
                G_, res = anchors(C, cls.device)
                stats["rew"].append(float(r.mean()))
                stats["rew_sd"].append(float(r.std(0).mean()))
                stats["n_inl"].append(float(np.mean([n for _, n in fits])))
                stats["H"].append(float(-(q * logp_all[b]).sum(0).mean()))
                stats["pmode"].append(float(q.max(0).values.mean()))
                stats["nonmode"].append(float((k != mode).float().mean()))
                stats["d_anchor"].append(float(((G_[k] - G_[mode]).norm(dim=-1) * (HW / 2)).mean()))
            loss = loss - (adv * _logp(logp_all[b], k)).mean() / B
        return p["weight"] * loss, {k: round(float(np.mean(v)), 5) for k, v in stats.items()}, True


def _logp(logp, k):
    """logp (C,h,w)、k (G,h,w) → 每份每格所选锚点的 log π，(G,h,w)。"""
    G, h, w = k.shape
    return torch.gather(logp[None].expand(G, -1, -1, -1), 1, k[:, None])[:, 0]
