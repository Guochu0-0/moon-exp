"""[rloo] 的逐步记录（「从健康到平凡解：逐步记录自洽坍缩的过程」#209）。只读、不求梯度、不动训练的随机数流
（观察集抽点用自己的发生器），打开与否训练结果相同。

- trace：每步把训练批次的逐格量（step_trace）交给训练循环，写进 <out>/trace/train_<首步>-<末步>.npz；参数更新之后
  在同一对图上再做一次粗匹配（after_update），看这一步让哪些格子的最高分位置动了、往哪里动。
- obs_every > 0：每 obs_every 步（含训练前的第 0 步）在固定观察集上用当前模型做一次训练时的正常推理（Observer），
  写 <out>/trace/obs_<step>.npz。观察集为 Val 里均匀取的 obs_n 对，每对两个版本：原图（version 0），以及一组固定的
  扰动（version 1：按训练的 [aug] 几何与擦除范围、以 (obs_seed, 序号) 播种抽一次，标注按同一 T 变换；不做光度扰动）。

坐标：粗级 h×w 格（560 输入时 40×40，行优先展平），格 (i, j) 的中心在光学原网格 (12.8·(j+0.5), 12.8·(i+0.5))；
锚点 64×64（SAR 网格，编号 k 的中心在 (8·(k%64+0.5), 8·(k//64+0.5))）。位置一律为原网格角点约定 px（u = 256·(x_n + 1)），
仿射为光学 → SAR、原网格角点约定的 2×3（与 preds 同）。数组的第 0 维是 batch 里的第几对（观察集为第几个样本）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..geom import HW, compose, inv_affine
from .pseudo import affine_norm, grid

ANCHOR_GRID = 64     # 锚点格（SAR 网格）
LEVEL_GRID = 35      # 细化各级的量统一汇总到尺度 16 的网格（560 / 16）


def to_px(x):
    """归一化坐标 → 原网格角点约定 px。"""
    return (x + 1) * (HW / 2)


def mapped(A, h, w, device):
    """仿射 A（原网格）把 h×w 网格各点中心映射到的位置 (2,h,w)，px。"""
    An = torch.from_numpy(affine_norm(A)).to(device).float()
    return to_px(torch.einsum("ij,hwj->ihw", An[:, :2], grid(h, w, device)) + An[:, 2, None, None])


def pool(x, n):
    """(…,H,W) → (…,n,n) 均值。"""
    sh = x.shape
    return F.adaptive_avg_pool2d(x.reshape(-1, 1, *sh[-2:]).float(), n).reshape(*sh[:-2], n, n)


def cell_signal(q, k, adv, temp):
    """每格的策略梯度：粗级损失 −mean(a · log π(k)) 对这一格 logits 的负梯度（不计 1/格数 的因子）为
    v = Σ_g a_g (e_{k_g} − q) / (G·temp)。返回 (‖v‖ (h,w), v 在最高分位置上的分量 (h,w)：> 0 即这一步把最高分位置的
    logit 往上推)。q：按训练温度的概率 (C,h,w)；k、adv：(G,h,w)。"""
    G = k.shape[0]
    C, h, w = q.shape
    a = adv.float().reshape(G, -1)
    v = torch.zeros(C, h * w, device=q.device).scatter_add_(0, k.reshape(G, -1).long(), a)
    v = (v - a.sum(0, keepdim=True) * q.reshape(C, -1).float()) / (G * temp)
    mode = q.reshape(C, -1).argmax(0)
    return v.norm(dim=0).reshape(h, w), v.gather(0, mode[None])[0].reshape(h, w)


def coarse_state(P, q, cert, anchor_flow):
    """一对的粗级状态：P 温度 1 的概率 (C,h,w)，q 训练温度的概率；cert 粗级 certainty logits (1,h,w)。"""
    mode = P.argmax(0)
    pos = to_px(anchor_flow(P[None], mode[None])[0])
    return {"mode": mode.reshape(-1).short(), "pmode1": P.max(0).values.half(), "pmode": q.max(0).values.half(),
            "pos": pos.half(), "cert_c": cert[0].float().sigmoid().half()}


def sar_masks(sar, occ=None):
    """SAR 输入 (1,H,W) 上补零区（像素值 < 1/255，即几何扰动把画面移出去的部分）与擦除区（occ，原网格 (1,512,512)）
    在锚点格上的面积比例，×255 存成 uint8。"""
    out = {"pad": (pool((sar[0] < 1 / 255).float(), ANCHOR_GRID) * 255).round().byte()}
    if occ is not None:
        out["occ"] = (pool(occ[0].float(), ANCHOR_GRID) * 255).round().byte()
    return out


def affine_arr(A):
    return np.full((2, 3), np.nan) if A is None else np.asarray(A, np.float64).reshape(2, 3)


def step_trace(q, P, k, r, d, adv, cert, ce, A, n_inl, temp, anchor_flow):
    """训练批次一对图的逐格量。q、P、k、cert 同 cell_signal / coarse_state；r、d、adv：每份每格的奖励、离自估仿射的
    距离（px）、权重 (G,h,w)，d 可以为 None（自估仿射拟合失败）；ce：正常推理尺度 1 的 certainty logits (1,H,W)；
    A、n_inl：自估仿射与它的内点数。"""
    h, w = q.shape[-2:]
    norm, up = cell_signal(q, k, adv, temp)
    mode = q.argmax(0)
    nan = torch.full((h, w), float("nan"), device=q.device)
    out = {**coarse_state(P, q, cert, anchor_flow),
           "rew": r.float().mean(0).half(), "rew_sd": r.float().std(0).half(), "adv": adv.float().abs().mean(0).half(),
           "d": (nan if d is None else d.float().mean(0)).half(), "same": (k == mode).float().mean(0).half(),
           "gnorm": norm.half(), "gup": up.half(), "cert1": pool(ce[0].float().sigmoid(), h).half()}
    return {**{x: v.cpu().numpy() for x, v in out.items()}, "A": affine_arr(A), "n_inl": np.int32(n_inl)}


def level_trace(s, r, d, adv=None):
    """细化尺度 s 一级的抽签：奖励、离自估仿射的距离 (N,h_s,w_s) 与权重 → 链间均值汇总到 35×35。"""
    out = {f"rew_{s}": pool(r.float().mean(0), LEVEL_GRID), f"d_{s}": pool(d.float().mean(0), LEVEL_GRID)}
    if adv is not None:
        out[f"adv_{s}"] = pool(adv.float().abs().mean(0), LEVEL_GRID)
    return {x: v.half().cpu().numpy() for x, v in out.items()}


def traj_trace(traj, A, ref=None):
    """正常推理轨迹上各级细化输出 μ 离仿射 A（以及 ref，如标注）映射的位置的距离，汇总到 35×35。"""
    out = {}
    for s, (_, _, mu) in traj.items():
        hs, ws = mu.shape[-2:]
        for name, M in (("dmu", A), ("dmu_gt", ref)):
            if M is None:
                continue
            dist = (to_px(mu[0].float()) - mapped(M, hs, ws, mu.device)).norm(dim=0)
            out[f"{name}_{s}"] = pool(dist, LEVEL_GRID).half().cpu().numpy()
    return out


def lstsq_affine(src, dst):
    """最小二乘仿射 src → dst（N×2），2×3。"""
    X = np.c_[src, np.ones(len(src))]
    return np.linalg.lstsq(X, dst, rcond=None)[0].T


class Observer:
    """固定观察集（见模块说明）。影像在构造时按训练的预处理读好、放在 CPU 上。"""

    def __init__(self, p, model, ds):
        from workbench.dataset import Dataset

        from .. import augment
        self.p, self.model = p, model
        labels = Dataset(ds.data.root)
        pairs = ds.data.pairs("val")
        idx = np.linspace(0, len(pairs) - 1, p["obs_n"]).round().astype(int)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(model.resize(x), np.float32))[None, None]
        self.items = []
        for n, i in enumerate(idx):
            pair = pairs[i]
            opt = ds.map_opt(ds.data.optical("val", pair)).astype(np.float32)
            sar = ds.map_sar(ds.data.sar("val", pair)).astype(np.float32)
            cp_opt, cp_sar = labels.checkpoints("val", pair)
            rng = np.random.default_rng([p["obs_seed"], n])
            T = augment.sample_T(rng, **ds.aug_kw) if "geo" in ds.aug else np.c_[np.eye(2), np.zeros(2)]
            sar_t = augment.warp(sar, T)
            if "erase" in ds.aug:
                sar_t, _ = augment.erase(sar_t, rng, **ds.erase_kw)
            for v, (img, M) in enumerate(((sar, np.c_[np.eye(2), np.zeros(2)]), (sar_t, T))):
                q = cp_sar @ M[:, :2].T + M[:, 2]                 # 检查点随 SAR 一起变换（同 workbench.perturb）
                self.items.append({"pair": pair, "version": v, "T": M, "image0": t(opt), "image1": t(img),
                                   "cp_opt": cp_opt, "cp_sar": cp_sar, "A_gt": lstsq_affine(cp_opt, q)})

    @torch.no_grad()
    def __call__(self, step):
        """当前模型在观察集上的一次正常推理 → {量: 沿样本叠起的数组}。"""
        from workbench.protocol import pair_error

        from . import rloo as R
        p, model, dev = self.p, self.model, self.model.device
        gen = torch.Generator(device=dev).manual_seed(p["obs_seed"] * 100003 + step)
        rows = []
        for it in self.items:
            i0, i1 = it["image0"].to(dev), it["image1"].to(dev)
            f1, f2 = R.features(model, i0, i1)
            cls, cert = R.coarse(model, f1, f2)
            cls = cls.float()
            P, q = cls[0].softmax(0), (cls[0] / p["temp"]).softmax(0)
            h, w = P.shape[-2:]
            traj = {}
            flow, ce = R.refine(model, f1, f2, R.anchor_flow(P[None], P.argmax(0)[None]), cert.float(), traj)
            (A, n_inl), = R.fit_many(flow, ce, p["num"], p["ransac"], gen=gen)
            row = {x: v.cpu().numpy() for x, v in {**coarse_state(P, q, cert[0], R.anchor_flow), **sar_masks(i1[0])}.items()}
            pos = to_px(R.anchor_flow(P[None], P.argmax(0)[None])[0])
            f1px = to_px(flow[0].float())
            H, W = f1px.shape[-2:]
            for name, M in (("self", A), ("gt", it["A_gt"])):
                if M is None:
                    row[f"d_{name}"] = np.full((h, w), np.nan, np.float16)
                    row[f"e1_{name}"] = np.full((h, w), np.nan, np.float16)
                    continue
                row[f"d_{name}"] = (pos - mapped(M, h, w, dev)).norm(dim=0).half().cpu().numpy()
                row[f"e1_{name}"] = pool((f1px - mapped(M, H, W, dev)).norm(dim=0), h).half().cpu().numpy()
            row.update(traj_trace(traj, A, it["A_gt"]))
            T = it["T"]
            err = pair_error(None if A is None else compose(inv_affine(T), A), it["cp_opt"], it["cp_sar"])
            row.update(flow1=pool(f1px, h).half().cpu().numpy(), cert1=pool(ce[0, 0].float().sigmoid(), h).half().cpu().numpy(),
                       A=affine_arr(A), n_inl=np.int32(n_inl), err=np.float32(err), A_gt=it["A_gt"], T=T,
                       version=np.int8(it["version"]))
            rows.append(row)
        out = {x: np.stack([r[x] for r in rows]) for x in rows[0]}
        out["pair"] = np.array([it["pair"] for it in self.items])
        return out
