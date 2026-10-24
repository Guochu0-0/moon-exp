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
- ref（第二批，#183）：不细化各份样本。先用当前模型对这对图正常推理一次（每格取 mode，细化到尺度 1），同样抽 num
  个点、RANSAC 拟合参照仿射 A_ref；每份样本每格的粗匹配位置（anchor_flow 的输出，即抽到的锚点经邻居加权）与 A_ref
  把格子中心映射到的位置相距 d（原网格 px），按 shape 由 d 逐格给分（distance_reward）。不筛图对，参照拟合得差也照用。
  shape（#191）：ripe（默认，照 RIPE++ 分级奖励）d ≤ tau 时 1 − ½(d/delta)²，d > tau 时 r_out；
  clip：max(1 − ½(d/delta)², r_out)，去掉 ripe 在 d = tau 处的跳变，不用 tau；
  gauss：exp(−½(d/delta)²)，处处光滑单调，不用 tau、r_out。长度参数只有 delta、tau，按级复用时同比缩放即可。
  cauchy（#197）：1 / (1 + ½(d/delta)²)，在 d = 0 附近与 gauss 同为 1 − ½(d/delta)²，远处按 2(delta/d)² 衰减：
  单精度下 gauss 在 d 超过约 14·delta 时为 0（这样的像素没有梯度），cauchy 在任何有限距离上都不为 0。
RANSAC 失败的样本：r1 记 0，ncc 记 −1；ref 的参照拟合失败时整对记 0（这一步没有梯度）。

权重（baseline）：
- loo：RLOO（Ahmadian 等 2402.14740 的留一基线），优势 = 本份 reward − 同一对其余 group − 1 份的均值（逐格 reward
  逐格算），不除标准差；
- none：RIPE++ 式，直接用 reward 当权重，不减平均。
loss = −mean_{份, 格} 权重 · log π。不训 certainty、不加 KL。

细化加噪（#192，只配 reward = ref）：levels 列出加噪的细化级（16、8、4、2、1 的子集，16 自 #197 起；空 = 细化照推理跑、不加噪，即上面的做法）。
此时每对 group 条完整的抽签链：粗级按上面抽锚点 → 各级细化照推理跑，但 levels 里那一级的输出 a 从以细化层输出 μ
（这一级的输入 warp + 细化层算出的位移）为中心、σ = k·s 原网格 px 的二维各向同性高斯里抽，再送进下一级。
这一级的 log π = −‖a − μ‖² / 2σ²（略去常数），μ 带梯度（经本级细化层与本级特征的 proj 回到 VGG；尺度 8–1 的特征只来自 VGG，DINOv2 的 LoRA 只受粗级训练）；各级之间照 romatch 本来就
detach，所以各级的 log π 只经本级的细化层回到特征。尺度 16 的细化层读 DINOv2 特征（proj16），它的梯度经 proj16 回到
DINOv2 的 LoRA。每级的抽签结果逐像素与参照仿射映射的位置相距 d，按与粗级相同的
shape 打分（fine_shape 非空时细化各级改用它，#197），长度参数按级缩放：delta_s = delta·s/8、tau_s = tau·s/8（粗级尺度记为 8，即 E15 / #191 的 delta、tau）；
各级（含粗级）的优势各自 RLOO（减同一位置其余链的均值）。loss = 粗级项 + 各加噪级的 −mean(优势 · log π) 之和。
实现照 E18 的开销测试（runs/E18/code/cost.py）：第一遍不求梯度跑完全部链，记下各加噪级的输入、certainty 与抽到的输出；
第二遍逐级、每 chunk 条链带梯度重算 μ 并立刻反向，细化层参数直接得到梯度，特征上的梯度先累积在叶子上，最后经一个
替身项随粗级损失一起反向穿过编码器（替身项的值换成细化各级损失之和，日志里的 loss 即粗级 + 细化）。

细化训练方式（#201，fine_mode；默认 chain 即上面的整链抽签）：grad、local 不跑抽签链，levels 里的每一级都从正常推理
（即拟合参照仿射的那一次）这一级的输入出发，带梯度重算细化层的输出 μ_s（梯度去向同 chain）。
- grad（精确梯度，诊断用）：loss_s = −mean_像素 r(μ_s)，r 为 level_reward，参照仿射当常数，直接对 μ_s 求导；
- local（逐级局部抽签）：从同一个 μ_s 抽 fine_group 次 x = μ_s + σε（σ 同 chain），按 r(x) 打分、RLOO，
  loss_s = −mean(优势 · log π_s)；抽签只在这一级之内，不往下一级传。奖励只当数值用。
levels 非空时另记正常推理轨迹上尺度 16…1 各级 μ 离参照的距离（dmu_s）；cos_every > 0 时每 cos_every 次调用在第一对图上
算一次 local 与 grad 两种损失对各级细化层参数的梯度的余弦相似度（grad_cos，只读、不影响训练）。
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

PARAMS = {"reward": "r1",   # r1 | ncc | rand | ref
          "group": 16,      # 每对的样本份数 G
          "temp": 1.0,      # 采样温度
          "num": 5000,      # 每份样本抽来拟合仿射的点数（同 [cexp] RoMa）
          "ransac": 3.0,    # RANSAC 阈值与 r1 的残差尺度（原网格 px）
          "chunk": 8,       # 细化一次并行跑几份样本（只影响显存）
          "baseline": "loo",  # loo（减同一对其余份的均值）| none（RIPE++ 式，reward 直接当权重）
          "delta": 8.0,     # ref：分级奖励的尺度（原网格 px）
          "tau": 16.0,      # ref：内点阈值（原网格 px）
          "r_out": -0.1,    # ref：d > tau 的格子得分（ripe）/ 下限（clip）
          "shape": "ripe",  # ref：奖励随距离的形式 ripe | clip | gauss（#191）| cauchy（#197）
          "fine_shape": "", # 细化各级的奖励形式（#197）；空 = 同 shape
          "levels": [],     # 细化中加高斯噪声的级（16、8、4、2、1 的子集，#192、#197）；空 = 细化不加噪
          "k": 0.5,         # 加噪级的 σ = k × 这一级的尺度（原网格 px，#192）
          "fine_mode": "chain",  # 细化训练方式（#201）：chain（整链抽签）| grad（精确梯度）| local（逐级局部抽签）
          "fine_group": 8,  # local：每级每像素的抽签次数
          "cos_every": 0,   # 每几次调用算一次 local 与 grad 梯度的余弦相似度（0 = 不算，#201）
          "weight": 1.0}
MODELS = {"roma": {}}
REWARDS = ("r1", "ncc", "rand", "ref")
SHAPES = ("ripe", "clip", "gauss", "cauchy")
BASELINES = ("loo", "none")
FINE_MODES = ("chain", "grad", "local")
SCALES = ("16", "8", "4", "2", "1")
LEVELS = (16, 8, 4, 2, 1)


def fill(p):
    return p


def check(cfg):
    p = cfg["rloo"]
    if p["reward"] not in REWARDS:
        raise ConfigError(f"[rloo] reward 可选 {REWARDS}，得到 {p['reward']!r}")
    if p["baseline"] not in BASELINES:
        raise ConfigError(f"[rloo] baseline 可选 {BASELINES}，得到 {p['baseline']!r}")
    for key in ("shape", "fine_shape"):
        if p[key] not in SHAPES and not (key == "fine_shape" and p[key] == ""):
            raise ConfigError(f"[rloo] {key} 可选 {SHAPES}，得到 {p[key]!r}")
    if p["group"] < (2 if p["baseline"] == "loo" else 1):
        raise ConfigError("[rloo] group 至少为 2（留一基线；baseline = 'none' 时至少为 1）")
    if p["delta"] <= 0 or p["tau"] <= 0:
        raise ConfigError("[rloo] delta、tau 应 > 0")
    if p["temp"] <= 0:
        raise ConfigError("[rloo] temp 应 > 0")
    for other in ("pseudo", "cexp", "neg", "warpc"):
        if other in cfg:
            raise ConfigError(f"[rloo] 不与 [{other}] 同用")
    lv = p["levels"]
    if any(not isinstance(s, int) or s not in LEVELS for s in lv) or len(set(lv)) != len(lv):
        raise ConfigError(f"[rloo] levels 只能是 {LEVELS} 中不重复的整数，得到 {lv!r}")
    if lv and p["reward"] != "ref":
        raise ConfigError("[rloo] levels 只配 reward = 'ref'")
    if p["k"] <= 0:
        raise ConfigError("[rloo] k 应 > 0")
    if p["fine_mode"] not in FINE_MODES:
        raise ConfigError(f"[rloo] fine_mode 可选 {FINE_MODES}，得到 {p['fine_mode']!r}")
    if (p["fine_mode"] != "chain" or p["cos_every"]) and not lv:
        raise ConfigError("[rloo] fine_mode = 'grad' / 'local' 与 cos_every 要配非空的 levels")
    if p["fine_group"] < (2 if p["baseline"] == "loo" else 1) or p["cos_every"] < 0:
        raise ConfigError("[rloo] fine_group 至少为 2（baseline = 'none' 时至少为 1），cos_every 应 ≥ 0")
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
def refine(model, f1, f2, flow, certainty, traj=None):
    """从粗 warp 与粗级 certainty 起，跑尺度 16…1 的细化（同 Decoder.forward：每级加位移与 certainty 增量，
    上采样到下一级）。f1、f2 的 batch 可以是 1（自动扩到 flow 的 batch）。返回尺度 1 的 (flow, certainty)。
    traj：可选的 dict，填入 尺度 → (这一级的输入 warp, 输入 certainty, 输出 μ)（#201）。"""
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
        if traj is not None:
            traj[ins] = (flow, certainty, flow + disp)
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


# ---------- 细化加噪（#192） ----------

def _disp(dec, s, d_flow, h, w):
    """细化层输出的位移 → 归一化坐标下的 warp 增量（同 refine）。"""
    return int(s) * torch.stack((d_flow[:, 0].float() / (dec.refine_init * w),
                                 d_flow[:, 1].float() / (dec.refine_init * h)), dim=1)


@torch.no_grad()
def rollout_noisy(model, f1, f2, P, cert, k, sig, chunk=8):
    """一对、len(k) 条链：粗 warp（锚点 k）→ 尺度 16…1 的细化（同 refine），sig 里的级（尺度 → σ，归一化坐标）在输出上
    加高斯后再送进下一级。返回 ({尺度: (输入 warp, 输入 certainty, 抽到的输出)}，每项 (N,…)；尺度 1 的 (flow, certainty))。
    sig 为空时尺度 1 的输出与 rollout 相同。"""
    dec = model.model.decoder
    H, W = f1[1].shape[-2:]
    rec = {s: ([], [], []) for s in sig}
    out = ([], [])
    for i in range(0, len(k), chunk):
        kk = k[i:i + chunk]
        n = len(kk)
        flow = anchor_flow(P.expand(n, -1, -1, -1), kk)
        ce = cert.expand(n, -1, -1, -1)
        for s in SCALES:
            a, b = f1[int(s)], f2[int(s)]
            if a.shape[0] != n:
                a, b = a.expand(n, -1, -1, -1), b.expand(n, -1, -1, -1)
            d_flow, d_cert = dec.conv_refiner[s](a, b, flow, scale_factor=1, logits=ce)
            mu = flow + _disp(dec, s, d_flow, H, W)
            if int(s) in sig:
                x = mu + sig[int(s)] * torch.randn_like(mu)
                for lst, v in zip(rec[int(s)], (flow, ce, x)):
                    lst.append(v)
                mu = x
            flow, ce = mu, ce + d_cert
            if s != "1":
                size = f1[int(s) // 2].shape[-2:]
                flow = F.interpolate(flow, size=size, mode=dec.flow_upsample_mode)
                ce = F.interpolate(ce, size=size, mode=dec.flow_upsample_mode)
        out[0].append(flow)
        out[1].append(ce)
    return {s: tuple(torch.cat(v) for v in r) for s, r in rec.items()}, tuple(torch.cat(v) for v in out)


def level_reward(x, A, s, p):
    """一级的抽签输出 x (N,2,h,w) 与参照仿射 A → (逐像素奖励 (N,h,w), 距离 d (N,h,w) px)。长度参数按级缩放：
    delta·s/8、tau·s/8；形式取 fine_shape，空时同粗级的 shape。"""
    return ref_cells(x, A, p["delta"] * s / 8, p["tau"] * s / 8, p["r_out"], p["fine_shape"] or p["shape"])


def fine_surrogate(dec, f1, f2, rec, adv, sig, chunk, backward, factor, w=1.0):
    """第二遍：逐级、每 chunk 条链带梯度重算 μ，loss_s = −w · mean(优势 · log π_s)，立刻 backward(loss 的一部分)。
    f1、f2：尺度 → 这一对的特征（带梯度，batch 1）；rec、sig 见 rollout_noisy；adv：尺度 → 优势 (N,h,w)。
    backward(l) 反向的是 l × factor（训练时为 optim.backward 与 optim.factor()）。细化层参数的梯度在这里就累积好；
    特征上的梯度先存在叶子上，返回一个替身项：随其余损失一起 backward 时，它把这些梯度传回编码器，值为各级损失之和。"""
    H, W = f1[1].shape[-2:]
    leaf1 = {s: f1[s].detach().requires_grad_() for s in rec}
    leaf2 = {s: f2[s].detach().requires_grad_() for s in rec}
    val = 0.0
    for s, (fin, cin, x) in rec.items():
        N = len(x)
        for i in range(0, N, chunk):
            n = min(chunk, N - i)
            d_flow, _ = dec.conv_refiner[str(s)](leaf1[s].expand(n, -1, -1, -1), leaf2[s].expand(n, -1, -1, -1),
                                                 fin[i:i + n], scale_factor=1, logits=cin[i:i + n])
            mu = fin[i:i + n] + _disp(dec, s, d_flow, H, W)
            lp = -(x[i:i + n] - mu).square().sum(1) / (2 * sig[s] ** 2)               # (n,h,w)
            loss = -w * (adv[s][i:i + n] * lp).mean() * n / N
            backward(loss)
            val += float(loss)
    return _surrogate(f1, f2, leaf1, leaf2, factor, val)


def _surrogate(f1, f2, leaf1, leaf2, factor, val):
    """特征叶子上已累积的梯度 → 替身项：随其余损失一起 backward 时把这些梯度传回编码器，值为 val。"""
    sur = sum((f1[s].float() * leaf1[s].grad.float()).sum() + (f2[s].float() * leaf2[s].grad.float()).sum()
              for s in leaf1 if leaf1[s].grad is not None)
    if not torch.is_tensor(sur):
        return torch.zeros((), device=f1[1].device) + val
    sur = sur / factor
    return sur - sur.detach() + val


# ---------- 细化：精确梯度与逐级局部抽签（#201） ----------

def _mu(dec, s, a, b, fin, cin, H, W):
    """尺度 s 的细化层从输入 warp fin、certainty cin 算出的输出 μ（带梯度）。"""
    d_flow, _ = dec.conv_refiner[str(s)](a, b, fin, scale_factor=1, logits=cin)
    return fin + _disp(dec, s, d_flow, H, W)


def level_loss(mu, A, s, sig, p):
    """一级：带梯度的 μ (1,2,h,w) → (loss, (奖励, 距离))。grad：−mean_像素 r(μ)，对 μ 求导；local：从 μ 抽 fine_group 次
    x = μ + σε，按 r(x) 打分、advantage，loss = −mean(优势 · log π)，log π = −‖x − μ‖² / 2σ²。"""
    if p["fine_mode"] == "grad":
        r, d = level_reward(mu, A, s, p)
        return -r.mean(), (r.detach(), d.detach())
    with torch.no_grad():
        x = mu.detach() + sig * torch.randn((p["fine_group"],) + tuple(mu.shape[1:]), device=mu.device)
        r, d = level_reward(x, A, s, p)
        adv = advantage(r, p["baseline"])
    lp = -(x - mu).square().sum(1) / (2 * sig ** 2)
    return -(adv * lp).mean(), (r, d)


def fine_direct(dec, f1, f2, traj, A, sig, p, backward, factor, w=1.0):
    """fine_mode = grad / local 的细化损失：sig 里的每一级从正常推理轨迹 traj（refine 记下的）这一级的输入起，带梯度
    重算 μ，loss_s = w · level_loss，立刻 backward(loss_s)；特征上的梯度与替身项同 fine_surrogate。
    返回 (替身项, {尺度: (奖励, 距离)})。"""
    H, W = f1[1].shape[-2:]
    leaf1 = {s: f1[s].detach().requires_grad_() for s in sig}
    leaf2 = {s: f2[s].detach().requires_grad_() for s in sig}
    val, out = 0.0, {}
    for s in sig:
        fin, cin, _ = traj[s]
        loss, out[s] = level_loss(_mu(dec, s, leaf1[s], leaf2[s], fin, cin, H, W), A, s, sig[s], p)
        loss = w * loss
        backward(loss)
        val += float(loss)
    return _surrogate(f1, f2, leaf1, leaf2, factor, val), out


def grad_cos(dec, f1, f2, traj, A, sig, p, scale=1024.0):
    """同一对图、同一轨迹上 local 与 grad 两种细化损失的梯度的余弦相似度（诊断，用 autograd.grad，不碰 .grad）。
    返回 cos_s（尺度 s 细化层的参数）、cosf_s（尺度 s 的两图特征）、cos（各级细化层参数拼成一个向量）。
    scale 在求导前乘到损失上（余弦与它无关），免得半精度的细化层里梯度下溢。"""
    H, W = f1[1].shape[-2:]
    cos = lambda u, v: float((u * v).sum() / (u.norm() * v.norm()).clamp(min=1e-30))
    out, tot = {}, {"grad": [], "local": []}
    for s in sig:
        a, b = f1[s].detach().requires_grad_(), f2[s].detach().requires_grad_()
        ps = [q for q in dec.conv_refiner[str(s)].parameters() if q.requires_grad]
        fin, cin, _ = traj[s]
        mu = _mu(dec, s, a, b, fin, cin, H, W)
        g = {}
        for mode in ("grad", "local"):
            loss, _ = level_loss(mu, A, s, sig[s], {**p, "fine_mode": mode})
            gr = torch.autograd.grad(loss * scale, ps + [a, b], retain_graph=True, allow_unused=True)
            g[mode] = [torch.zeros_like(q).flatten().float() if x is None else x.flatten().float()
                       for x, q in zip(gr, ps + [a, b])]
            tot[mode] += g[mode][:len(ps)]
        out[f"cos_{s}"] = cos(torch.cat(g["grad"][:len(ps)]), torch.cat(g["local"][:len(ps)]))
        out[f"cosf_{s}"] = cos(torch.cat(g["grad"][len(ps):]), torch.cat(g["local"][len(ps):]))
    out["cos"] = cos(torch.cat(tot["grad"]), torch.cat(tot["local"]))
    return out


# ---------- reward ----------

def fit(flow, cert, num, thr):
    """一份样本：尺度 1 的 warp (2,H,W) 与 certainty (1,H,W) → (仿射 A 或 None, 内点数)。"""
    return fit_many(flow[None], cert[None], num, thr)[0]


_POOL = None


def fit_many(flows, certs, num, thr):
    """G 份样本各自抽点、RANSAC。抽点在 GPU 上依次做，RANSAC 用线程并行（cv2 放开 GIL；fit_affine 每次调用前重置
    本线程的随机种子，结果与串行相同）。返回 [(A 或 None, 内点数)]。
    warp 含非有限值、或 certainty 经 sigmoid 后全为 0 / 非有限（模型发散时出现）的样本抽不出点，记为拟合失败 (None, 0)。"""
    global _POOL
    from concurrent.futures import ThreadPoolExecutor
    if _POOL is None:
        _POOL = ThreadPoolExecutor(8)
    Ms = []
    for f, c in zip(flows, certs):
        pc = c.float().sigmoid()
        if not (bool(torch.isfinite(f).all()) and bool(torch.isfinite(pc).all()) and float(pc.sum()) > 0):
            Ms.append(None)
            continue
        p0, p1, _ = sample_matches(f, c, num)
        Ms.append(np.c_[p0, p1, np.ones(len(p0))].astype(np.float32))
    out = list(_POOL.map(lambda M: (None, None, None) if M is None else fit_affine(M, thr), Ms))
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


def distance_reward(d, shape="ripe", delta=8.0, tau=16.0, r_out=-0.1):
    """离参照的距离 d（张量，px）→ 同形状的奖励。ripe：d ≤ tau 时 1 − ½(d/delta)²，否则 r_out（RIPE++ 分级奖励）；
    clip：max(1 − ½(d/delta)², r_out)；gauss：exp(−½(d/delta)²)；cauchy：1 / (1 + ½(d/delta)²)。
    clip、gauss、cauchy 不用 tau，gauss、cauchy 不用 r_out。"""
    q = 0.5 * (d / delta) ** 2
    if shape == "ripe":
        return torch.where(d <= tau, 1 - q, torch.full_like(d, r_out))
    if shape == "clip":
        return (1 - q).clamp(min=r_out)
    if shape == "gauss":
        return torch.exp(-q)
    if shape == "cauchy":
        return 1 / (1 + q)
    raise ValueError(f"shape 可选 {SHAPES}，得到 {shape!r}")


def ref_cells(fc, A, delta, tau, r_out, shape="ripe"):
    """fc：G 份粗 warp (G,2,h,w)；A：参照仿射（原网格）或 None → (逐格奖励 (G,h,w), 距离 d (G,h,w) 或 None)。
    d 为 fc 与 A 把格子中心映射到的位置之间的距离（原网格 px），按 shape 换成奖励（distance_reward）。"""
    G, _, h, w = fc.shape
    if A is None:
        return torch.zeros(G, h, w, device=fc.device), None
    An = torch.from_numpy(affine_norm(A)).to(fc.device).float()
    pred = torch.einsum("ij,hwj->hwi", An[:, :2], grid(h, w, fc.device)) + An[:, 2]
    d = (fc.permute(0, 2, 3, 1).float() - pred).norm(dim=-1) * (HW / 2)
    return distance_reward(d, shape, delta, tau, r_out), d


def advantage(r, baseline):
    """r (G, ...) → 策略梯度的权重：loo 减留一均值，none 原样。"""
    return loo(r) if baseline == "loo" else r


def loo(r):
    """留一优势：r (G, ...) → r − 其余 G−1 份的均值。"""
    G = r.shape[0]
    return r - (r.sum(0, keepdim=True) - r) / (G - 1)


# ---------- 训练成分 ----------

class Term:
    def __init__(self, p, model, ds):
        self.p, self.model = p, model
        self.calls = 0

    def __call__(self, st):
        p, model = self.p, self.model
        self.calls += 1
        i0, i1 = st.images()
        f1, f2 = features(model, i0, i1)
        cls, cert = coarse(model, f1, f2)
        cls = cls.float()
        B, C, h, w = cls.shape
        logp_all = F.log_softmax(cls / p["temp"], 1)
        loss = torch.zeros((), device=cls.device)
        stats = {k: [] for k in ("rew", "rew_sd", "n_inl", "H", "pmode", "nonmode", "d_anchor", "t_roll", "t_fit")}
        if p["reward"] == "ref":
            stats.update(ref_ok=[], d_ref=[], in_tau=[])
        lv = p["levels"]
        sig = {s_: p["k"] * s_ / (HW / 2) for s_ in lv}       # σ（归一化坐标）
        fine = torch.zeros((), device=cls.device)
        if lv:
            stats.update({f"{n}_{s_}": [] for s_ in lv for n in ("rew", "rew_sd", "d", "d90", "zero")},
                         t_roll_fine=[], t_fine=[], l_coarse=[], l_fine=[])
        for b in range(B):
            with torch.no_grad():
                q = logp_all[b].exp()                                            # (C,h,w)
                k = torch.multinomial(q.reshape(C, -1).T, p["group"], replacement=True).T.reshape(-1, h, w)
                P = cls[b:b + 1].softmax(1)
                fb = {s: x[b:b + 1] for s, x in f1.items()}
                gb = {s: x[b:b + 1] for s, x in f2.items()}
                t0 = time.time()
                if p["reward"] == "ref":    # 只细化不抽签的一份（正常推理）；各份样本只用粗 warp
                    traj = {} if lv else None
                    flow, ce = refine(model, fb, gb, anchor_flow(P, P.argmax(1)), cert[b:b + 1].float(), traj)
                else:
                    flow, ce = rollout(model, fb, gb, P, cert[b:b + 1].float(), k, p["chunk"])
                torch.cuda.synchronize()
                t1 = time.time()
                fits = fit_many(flow, ce, p["num"], p["ransac"])
                stats["t_roll"].append(t1 - t0)
                stats["t_fit"].append(time.time() - t1)
                if p["reward"] == "ref":
                    A_ref = fits[0][0]
                    r, d = ref_cells(anchor_flow(P.expand(p["group"], -1, -1, -1), k), A_ref,
                                     p["delta"], p["tau"], p["r_out"], p["shape"])
                    stats["ref_ok"].append(float(A_ref is not None))
                    if d is not None:
                        stats["d_ref"].append(float(d.mean()))
                        stats["in_tau"].append(float((d <= p["tau"]).float().mean()))
                elif p["reward"] == "r1":
                    r = torch.stack([r1_cells(flow[g], A, p["ransac"], (h, w)) for g, (A, _) in enumerate(fits)])
                elif p["reward"] == "ncc":
                    valid = sar_valid(i1[b])
                    r = torch.tensor([ncc(i0[b], i1[b], valid, A) for A, _ in fits], device=cls.device)[:, None, None]
                else:
                    r = torch.rand(p["group"], h, w, device=cls.device)
                adv = advantage(r, p["baseline"]).expand(p["group"], h, w)
                if lv and A_ref is not None:     # 正常推理轨迹上各级 μ 离参照的距离（#201）
                    for s_, (_, _, mu_) in traj.items():
                        stats.setdefault(f"dmu_{s_}", []).append(float(level_reward(mu_, A_ref, s_, p)[1].mean()))
                if lv and A_ref is not None and p["fine_mode"] == "chain":   # 细化加噪（#192）：全部链再跑一遍，各加噪级逐像素打分、各自 RLOO
                    t2 = time.time()
                    rec, _ = rollout_noisy(model, fb, gb, P, cert[b:b + 1].float(), k, sig, p["chunk"])
                    adv_f = {}
                    for s_ in lv:
                        r_s, d_s = level_reward(rec[s_][2], A_ref, s_, p)
                        adv_f[s_] = advantage(r_s, p["baseline"])
                        _level_stats(stats, s_, r_s, d_s)
                    torch.cuda.synchronize()
                    stats["t_roll_fine"].append(time.time() - t2)
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
            if lv and A_ref is not None:
                t3 = time.time()
                fb, gb = {s_: x[b:b + 1] for s_, x in f1.items()}, {s_: x[b:b + 1] for s_, x in f2.items()}
                dec = model.model.decoder
                if p["fine_mode"] == "chain":
                    fine = fine + fine_surrogate(dec, fb, gb, rec, adv_f, sig, p["chunk"], st.optim.backward,
                                                 st.optim.factor(), p["weight"] / B)
                else:
                    sur, rd = fine_direct(dec, fb, gb, traj, A_ref, sig, p, st.optim.backward, st.optim.factor(),
                                          p["weight"] / B)
                    fine = fine + sur
                    for s_, (r_s, d_s) in rd.items():
                        _level_stats(stats, s_, r_s, d_s)
                if p["cos_every"] and b == 0 and (self.calls - 1) % p["cos_every"] == 0:
                    for key, v in grad_cos(dec, fb, gb, traj, A_ref, sig, p).items():
                        stats.setdefault(key, []).append(v)
                torch.cuda.synchronize()
                stats["t_fine"].append(time.time() - t3)
        if lv:
            stats["l_coarse"].append(float(p["weight"] * loss))
            stats["l_fine"].append(float(fine))
        return p["weight"] * loss + fine, {k: round(float(np.mean(v)), 5) for k, v in stats.items() if v}, True


def _level_stats(stats, s, r, d):
    """一级的奖励 r 与距离 d（(N,h,w)，N 为链数 / 抽签次数）记进日志：均值、链间标准差（N > 1 时）、距离 90% 分位、
    奖励下溢为 0 的比例（#197）。"""
    stats[f"rew_{s}"].append(float(r.mean()))
    if r.shape[0] > 1:
        stats[f"rew_sd_{s}"].append(float(r.std(0).mean()))
    stats[f"d_{s}"].append(float(d.mean()))
    stats[f"d90_{s}"].append(float(torch.quantile(d.flatten()[::4].float(), 0.9)))
    stats[f"zero_{s}"].append(float((r == 0).float().mean()))


def _logp(logp, k):
    """logp (C,h,w)、k (G,h,w) → 每份每格所选锚点的 log π，(G,h,w)。"""
    G, h, w = k.shape
    return torch.gather(logp[None].expand(G, -1, -1, -1), 1, k[:, None])[:, 0]
