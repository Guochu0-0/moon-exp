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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ConfigError
from ..geom import HW, apply_affine, compose, in_to_orig, inv_affine, orig_to_in
from ..label import load_labels, pair_weights

PARAMS = {"labels": "",         # finetune.label 格式的离线伪仿射（jsonl）
          "top": 1.0,           # 只用 keep 的对里内点数最多的前这一比例
          "weight": 1.0,
          "pair_weight": ""}    # 每对损失乘的质量权重（#127）：空 = 不加权 / inlier_ratio 内点率，归一化到均值 1；要求 batch 1
MODELS = {"loftr": {"w_coarse": 1.0, "w_fine": 1.0},
          "roma": {"swap": 0.5,
                   "cls_soft": False,   # 粗级锚点分类的目标：一热（原损失）/ 双线性分到相邻 4 个锚点（#120，自拟）
                   "drop_top": 0.0,     # 每对每个尺度的回归项去掉相对伪仿射残差最大的这一比例像素（#120，DCFlow 2509.24423 §3.3）
                   "dense": "",         # 稠密自标签（#122）：finetune.dense_label 写出的老师 warp 目录；空 = 只用伪仿射
                   "dense_r": 8.0,      # 老师落点与伪仿射落点相差小于这么多（560 网格 px）的像素才用老师目标；≤ 0 = 不设门限
                   "dense_cert": 0.5,   # 老师 certainty 低于此的像素退回伪仿射
                   "reg_tol": 0.0,      # 1、2 两层的回归误差先减去这么多（原网格 px）再截到 0，即小于它不罚（#127，STLD 2404.04556 式）
                   "fine_ramp": [],
                   "match_dir": "",     # 可匹配掩码（#127）：finetune.dense_label 写出的老师 warp 目录；空 = certainty 照旧学「是否在重叠区」
                   "match_r": 3.0,      # 老师落点与伪仿射落点、正反往返回到原处，两者都差小于这么多（原网格 px）才算可匹配
                   "reg_loss": "robust",  # 回归损失（#127）：robust RoMa 原鲁棒项 / mixlap 两分量拉普拉斯混合（SEA-RAFT 2405.14793 §3.2），要 [model] unc_head
                   "mixlap_w": 1.0}}    # mixlap 的权重    # [a, b]：4、2、1 层的全部损失项乘一个第 a 步起从 0 线性升到第 b 步为 1 的系数（#127，尺度课程）；空 = 不用

FOCAL_ALPHA, FOCAL_GAMMA = 0.25, 2.0   # 上游 LoFTR default.py 的 LOSS.FOCAL_ALPHA / FOCAL_GAMMA
FINE_CORRECT_THR = 1.0                 # 上游 LOSS.FINE_CORRECT_THR：归一化窗口坐标
LOCAL_DIST = {1: 4, 2: 4, 4: 8, 8: 8}  # romatch train_roma_outdoor.py；阈值 (2/512)·d·s，512 恰为原网格
CE_WEIGHT, ALPHA, C = 0.01, 0.5, 1e-4  # romatch robust_loss 的 certainty 权重与鲁棒回归参数


def fill(p):
    return p


def check(cfg):
    if not cfg["pseudo"]["labels"]:
        raise ConfigError("[pseudo] labels 必填")
    if cfg["pseudo"].get("dense") and "aug" in cfg and cfg["aug"]["geo"]:
        raise ConfigError("[pseudo] dense 不配合 [aug] geo（老师 warp 按未扰动的影像算）")
    p = cfg["pseudo"]
    if p["pair_weight"] not in ("", "inlier_ratio"):
        raise ConfigError(f"[pseudo] pair_weight 可选 '' / 'inlier_ratio'，得到 {p['pair_weight']!r}")
    if p["pair_weight"] and cfg["optim"]["batch"] != 1:
        raise ConfigError("[pseudo] pair_weight 目前只支持 batch 1（权重乘在整步损失上）")
    fr = p.get("fine_ramp", [])
    if fr and not (len(fr) == 2 and 0 <= fr[0] <= fr[1]):
        raise ConfigError(f"[pseudo] fine_ramp 应为 [a, b]、0 ≤ a ≤ b，得到 {fr!r}")
    if p.get("reg_loss", "robust") not in ("robust", "mixlap"):
        raise ConfigError(f"[pseudo] reg_loss 可选 robust / mixlap，得到 {p['reg_loss']!r}")
    if p.get("reg_loss") == "mixlap" and not cfg["model"].get("unc_head"):
        raise ConfigError("[pseudo] reg_loss = mixlap 要 [model] unc_head = true")
    if p.get("match_dir") and "aug" in cfg and cfg["aug"]["geo"]:
        raise ConfigError("[pseudo] match_dir 不配合 [aug] geo（老师 warp 按未扰动的影像算）")
    if cfg["model"]["name"] == "loftr" and "neg" in cfg:
        raise ConfigError("LoFTR 上 [neg] 不配合 [pseudo]（负样本对并入同一次前向，伪标签不监督它们）")


def data_kw(p):
    return {}


class Term:
    def __init__(self, p, model, ds):
        self.p, self.model = p, model
        self.labels = load_labels(p["labels"], p["top"])
        ds.pairs = [q for q in ds.pairs if self.labels.get(q) is not None]
        self.match = Path(p["match_dir"]) if p.get("match_dir") else None
        if self.match is not None:
            miss = [q for q in ds.pairs if not (self.match / f"{q}.npy").exists()]
            if miss:
                raise FileNotFoundError(f"{self.match} 缺 {len(miss)} 对的老师 warp，如 {miss[:3]}")
        self.pw = pair_weights(p["labels"], ds.pairs, p["pair_weight"]) if p["pair_weight"] else None
        self.n = 0   # 已调用的步数（尺度课程用）
        self.dense = Path(p["dense"]) if p.get("dense") else None
        if self.dense is not None:
            miss = [q for q in ds.pairs if not (self.dense / f"{q}.npy").exists()]
            if miss:
                raise FileNotFoundError(f"{self.dense} 缺 {len(miss)} 对的老师 warp，如 {miss[:3]}")

    def dense_targets(self, pairs, sw, device):
        """(B,3,h,w)：查询图上每个像素的老师落点 (x, y)（归一化）与 certainty；交换方向的对取 SAR → 光学那一份。"""
        D = [np.load(self.dense / f"{q}.npy")[int(s)] for q, s in zip(pairs, sw)]
        return torch.from_numpy(np.stack(D)).to(device).float()

    def fine_w(self):
        """尺度课程（fine_ramp）在当前步的系数。"""
        fr = self.p.get("fine_ramp", [])
        if not fr:
            return 1.0
        a, b = fr
        return float(self.n >= a) if b == a else min(max((self.n - a) / (b - a), 0.0), 1.0)

    def __call__(self, st):
        self.n += 1
        w = self.pw[st.batch["pair"][0]] if self.pw else 1.0
        As = [np.asarray(self.labels[q]) for q in st.batch["pair"]]
        if "T" in st.batch:   # SAR 被已知 T warp 过：标签精确变为 T∘A
            As = [compose(T.numpy(), A) for T, A in zip(st.batch["T"], As)]
        if st.model_name == "loftr":
            loss, stats = loftr_loss(st.out, self.model.s, As, self.p["w_coarse"], self.p["w_fine"])
            return self.p["weight"] * w * loss, stats, stats["pairs_used"] > 0
        i0, i1 = st.images()
        sw = st.rng.random(st.B) < self.p["swap"]
        m = torch.from_numpy(sw)[:, None, None, None].to(i0.device)
        a, b = torch.where(m, i1, i0), torch.where(m, i0, i1)
        An = torch.from_numpy(np.stack([affine_norm(inv_affine(A) if s else A) for A, s in zip(As, sw)])
                              ).float().to(i0.device)
        occ = (st.batch["occ"].to(i0.device), m[:, 0, 0, 0]) if "occ" in st.batch else None
        dense = None
        if self.dense is not None:
            r = self.p["dense_r"]
            dense = (self.dense_targets(st.batch["pair"], sw, i0.device),
                     2 * r / self.model_res() if r > 0 else float("inf"), self.p["dense_cert"])
        match = None
        if self.match is not None:   # 查询方向与反方向的老师落点
            D = [np.load(self.match / f"{q}.npy") for q in st.batch["pair"]]
            f = lambda k: torch.from_numpy(np.stack([d[k(int(s))] for d, s in zip(D, sw)])).to(i0.device).float()
            match = (f(lambda s: s), f(lambda s: 1 - s), self.p["match_r"] * 2 / HW)
        kw = dict(cls_soft=self.p["cls_soft"], drop_top=self.p["drop_top"], occ=occ, dense=dense,
                  reg_tol=self.p["reg_tol"] * 2 / HW, fine_w=self.fine_w(), match=match,
                  mixlap=self.p["mixlap_w"] if self.p["reg_loss"] == "mixlap" else 0.0)
        tr = getattr(self.model, "train_res", "560")
        if tr == "864":   # 560 一遍只给初值（#122）
            with torch.no_grad():
                c = self.model.forward(a, b)
            loss, stats = 0.0, {}
        else:
            c = self.model.forward(a, b)
            loss, stats = roma_loss(c, An, **kw)
        if tr != "560":
            h0, h1 = st.batch["image0_hi"].to(i0.device), st.batch["image1_hi"].to(i0.device)
            l_hi, s_hi = roma_loss(self.model.forward_hi(torch.where(m, h1, h0), torch.where(m, h0, h1), c), An, **kw)
            loss = loss + l_hi
            stats.update(s_hi if tr == "864" else {f"hi_{k}": v for k, v in s_hi.items()})
        if self.pw:
            stats["pair_w"] = round(w, 4)
        if self.p["fine_ramp"]:
            stats["fine_w"] = round(kw["fine_w"], 4)
        return self.p["weight"] * w * loss, stats, True

    def model_res(self):
        from ..models.roma import RES
        return RES


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


def drop_largest(epe, m, frac):
    """m 中每对去掉 epe 最大的 frac 比例像素（按 detach 的 epe 排序），返回新掩码。"""
    out = m.clone()
    e = epe.detach()
    for b in range(len(e)):
        n = int(m[b].sum())
        k = int(n * frac)
        if k:
            thr = e[b][m[b]].kthvalue(n - k).values
            out[b] &= e[b] <= thr
    return out


def soft_anchor_target(x2, r):
    """x2: (B,h,w,2) 归一化坐标 → (B,r²,h,w)：按双线性权重分到相邻 4 个锚点（锚点中心同 grid(r, r)，行优先）。
    落在锚点网格外缘的点并到最近的锚点上。"""
    B, h, w, _ = x2.shape
    g = ((x2 + 1) * r / 2 - 0.5).clamp(0, r - 1)               # 连续的锚点索引
    g0 = g.floor().clamp(max=r - 2) if r > 1 else g.floor()
    f = g - g0
    g0 = g0.long()
    out = x2.new_zeros(B, r * r, h * w)
    for dx in (0, 1):
        for dy in (0, 1):
            wgt = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])
            idx = (g0[..., 1] + dy) * r + (g0[..., 0] + dx)
            out.scatter_add_(1, idx.reshape(B, 1, -1), wgt.reshape(B, 1, -1))
    return out.reshape(B, r * r, h, w)


def occluded(occ, An, h, w):
    """occ = (mask (B,1,H,W) SAR 原网格上被遮的为 1, swapped (B,) bool)。返回 (B,h,w) bool：查询像素对应的 SAR 位置被遮。
    未交换时查询图是光学，SAR 位置 = 伪仿射落点；交换时查询图就是 SAR，位置 = 像素自己。"""
    mask, sw = occ
    g = grid(h, w, An.device)
    x2, _ = affine_gt(An, h, w)
    xy = torch.where(sw[:, None, None, None], g[None].expand_as(x2), x2)
    return F.grid_sample(mask.float(), xy, mode="nearest", align_corners=False)[:, 0] > 0.5


def mix_dense(x2, dense, h, w):
    """x2 (B,h,w,2) 伪仿射落点 → 过门限的像素换成老师落点。返回 (新 x2, 换掉的比例)。"""
    D, r, c_thr = dense
    d = F.interpolate(D, size=(h, w), mode="bilinear", align_corners=False)
    xd, cd = d[:, :2].permute(0, 2, 3, 1), d[:, 2]
    use = (cd >= c_thr) & ((xd - x2).norm(dim=-1) < r)
    return torch.where(use[..., None], xd, x2), float(use.float().mean())


def matchable(match, x2, h, w):
    """可匹配掩码（#127）：match = (查询方向老师 (B,3,S,S)，反方向老师 (B,3,S,S)，r 归一化)。
    像素可匹配 ⇔ 老师落点与伪仿射落点相差 < r，且从老师落点按反方向老师走回来、离原像素 < r。返回 (B,h,w) float。"""
    Dq, Dr, r = match
    xd = F.interpolate(Dq[:, :2], size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    back = F.grid_sample(Dr[:, :2], xd, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    g = grid(h, w, xd.device)[None]
    return (((xd - x2).norm(dim=-1) < r) & ((back - g).norm(dim=-1) < r)).float()


def mixlap_nll(flow, x2, unc, m):
    """两分量拉普拉斯混合的负对数似然（SEA-RAFT 2405.14793 §3.2，#127）：残差按原网格 px、x / y 各算再相加；
    一个分量尺度固定为 1 px（即普通 L1），另一个尺度 e^β、β ∈ [0, 10]；unc (B,2,h,w) = (混合权重 logit, β)。"""
    d = (flow.permute(0, 2, 3, 1).float() - x2).abs() * (HW / 2)            # (B,h,w,2)
    a = unc[:, 0].float()[..., None]
    beta = unc[:, 1].float().clamp(0, 10)[..., None]
    l1 = F.logsigmoid(a) - d - math.log(2)
    l2 = F.logsigmoid(-a) - beta - d / beta.exp() - math.log(2)
    nll = -torch.logaddexp(l1, l2).sum(-1)
    return nll[m].mean() if bool(m.any()) else (flow.sum() + unc.sum()) * 0


def robust(x, s, like):
    """RoMa 的鲁棒回归项（robust_loss.py）：x 为该尺度被监督像素的 EPE（归一化）。没有像素时返回带梯度的 0。"""
    cs = C * s
    return (cs ** ALPHA * ((x / cs) ** 2 + 1) ** (ALPHA / 2)).mean() if x.numel() else like.sum() * 0


def roma_loss(corresps, An, cls_soft=False, drop_top=0.0, occ=None, dense=None, gt=None, reg_tol=0.0, fine_w=1.0,
              match=None, mixlap=0.0):
    """RoMa 原损失，GT 换成仿射。An: (B,2,3) 归一化。#120 新增（默认关）：cls_soft 粗级双线性软目标；
    drop_top 每对去掉回归残差最大的这一比例像素；occ 遮挡扰动，被遮处 prob = 0（不监督位置、certainty 目标为 0）。
    #122：dense = (老师 (B,3,H,W)，距离门限（归一化）, certainty 门限)：两个门限都过的像素把落点目标换成老师的
    （粗级分类与细化回归都用换过的目标；prob 仍按伪仿射是否落在图内）。各尺度只看 corresps 里有的（864 那一遍是 8…1）。
    gt：可选，gt(h, w) → (x2 (B,h,w,2), prob (B,h,w))，代替伪仿射给出目标（WarpC 的 warp 监督，parts/warpc.py）。
    #127：reg_tol（归一化）1、2 两层回归用 max(EPE − reg_tol, 0)；fine_w 乘在 4、2、1 层的回归与 certainty 项上；
    match 给出时 certainty（含粗级 gm_certainty）的目标乘上可匹配掩码（matchable），回归照旧；
    mixlap > 0 时有 unc 输出的尺度（细化层）回归改为 mixlap × 两分量拉普拉斯混合 NLL（mixlap_nll）。"""
    tot, st, prev_epe = 0.0, {}, None
    for s in sorted(corresps, reverse=True):          # 16, 8, 4, 2, 1
        c = corresps[s]
        flow, cert = c["flow"], c["certainty"]
        h, w = flow.shape[-2:]
        x2, prob = gt(h, w) if gt is not None else affine_gt(An, h, w)
        if dense is not None:
            x2, frac = mix_dense(x2, dense, h, w)
            if s == 1:
                st["dense_frac"] = round(frac, 4)
        if occ is not None:
            prob = prob * ~occluded(occ, An, h, w)
        if s <= 8 and prev_epe is not None:
            prob = prob * (F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                           < (2 / 512) * LOCAL_DIST[s] * s)
        m = prob > 0.99
        pc = prob
        if match is not None:
            mk = matchable(match, x2, h, w)
            pc = prob * mk
            if s == 1:
                st["match_frac"] = round(float(mk[m].mean()), 4) if m.any() else None
        if "gm_cls" in c:
            cls = c["gm_cls"].float()
            K = cls.shape[1]
            r = round(math.sqrt(K))
            with torch.no_grad():
                if cls_soft:
                    tgt = soft_anchor_target(x2, r)
                else:
                    G = grid(r, r, cls.device).reshape(K, 2)
                    tgt = torch.cdist(x2.reshape(len(x2), -1, 2), G[None].expand(len(x2), -1, -1)).argmin(-1)
                    tgt = tgt.reshape(x2.shape[:3])
            ce = (-(tgt * F.log_softmax(cls, 1)).sum(1) if cls_soft else F.cross_entropy(cls, tgt, reduction="none"))[m]
            l_cls = ce.mean() if ce.numel() else cls.sum() * 0
            l_gc = F.binary_cross_entropy_with_logits(c["gm_certainty"][:, 0].float(), pc)
            tot = tot + l_cls + CE_WEIGHT * l_gc
            st[f"cls{s}"] = round(float(l_cls), 5)
        epe = (flow.permute(0, 2, 3, 1).float() - x2).norm(dim=-1)
        mr = drop_largest(epe, m, drop_top) if drop_top > 0 else m
        e = (epe - reg_tol).clamp_min(0) if reg_tol > 0 and s <= 2 else epe
        l_reg = mixlap * mixlap_nll(flow, x2, c["unc"], mr) if mixlap > 0 and "unc" in c else robust(e[mr], s, flow)
        l_ce = F.binary_cross_entropy_with_logits(cert[:, 0].float(), pc)
        tot = tot + (fine_w if s <= 4 else 1.0) * (l_reg + CE_WEIGHT * l_ce)
        st[f"reg{s}"] = round(float(l_reg), 5)
        if s == 1:
            st["epe1_px"] = round(float(epe[m].median() * HW / 2), 3) if m.any() else None
        prev_epe = epe.detach()
    return tot, st
