"""正反向互逆约束奖励（#255；[rloo] cyc = true）。挂接在 finetune/parts/rloo.py 的 Term 上：__call__ 每对图算完现有项后
把要用的中间量记成 Ctx，最后交给 step()。

现有训练只训"从光学找 SAR"一个方向（光学是第一张图）。这一项另加反向一遍：交换两张图（SAR 第一、光学第二，同评测里
romatch 适配器的 swap=True），要求两个方向的结果互为逆。
- 反向参照 B_ref：参照模型（慢速副本，Term.ref_fit）在交换后的一对图上正常推理、拟合仿射（SAR → 光学，原网格角点约定）。
  拟合失败的对不做这一项。样本筛选只按正向参照，反向参照不另设筛选。
- 正向互逆项：现有项已经抽好的正向样本（粗级 group 份；细化各级从正向正常推理轨迹的 μ_s 另抽 fine_group 次，随机数取自
  独立发生器）每个格子 / 像素抽到的 SAR 落点经 B_ref 映回光学，离这个格子 / 像素中心的距离（原网格 px）按现有奖励的形状
  打分（粗级 shape、delta；细化 fine_shape、delta·s/8，同 level_reward），组内留一法优势，乘在正向的对数概率上。
  画面外掩码沿用现有项（正向参照映射到 SAR 画面外 / 补零区的格子 / 像素优势置零）。
- 反向互逆项：当前模型在交换后的一对图上照正向同样抽签（粗级 group 份、细化各级从反向正常推理轨迹的 μ_s 各抽
  fine_group 次），光学落点经正向参照 A_ref 映回 SAR，离这个 SAR 格子 / 像素中心的距离同样打分。反向样本只用这一项打分。
  反向的画面外掩码：SAR 格子 / 像素中心不在补零区（rloo.sar_valid），且 B_ref 把它映射到光学画面内。
- 都不做 certainty 的 BCE。总损失 = 现有项 + cyc_w × (正向互逆项 + 反向互逆项)，细化部分另照现有项乘 fine_w。

梯度分开量（同遮挡约束奖励 #249）：本成分在内部反向。现有项先反向（保留计算图，正向互逆项还要经同一次前向回传），记下
各参数的梯度；再反向正向互逆项、逐对做反向互逆项并反向。日志 main_gn、cyc_gn 为现有项与（系数 1 的）新项在共用参数上的
梯度范数，cyc_coef = main_gn / cyc_gn，cyc_ratio = cyc_w · cyc_gn / main_gn。cyc_w = 0 时只量不学：新项照算、量完撤掉。
新项的随机数（反向的粗级与细化抽签、正向细化的抽签）取自以 (torch 初始种子, 调用次数, 批内序号, 255) 起的独立发生器，
不动训练的随机数流（只量不学时与不加这一项的训练同一条随机数流）。

另记（只记录）：cyc_ok（反向参照拟合成功的比例）、cyc_res（正向参照与反向参照复合 B_ref∘A_ref 把粗格中心映射到的位置
离原位置的平均距离，px）、cyc_rew_f / cyc_rew_r 与 cyc_d_f / cyc_d_r（正 / 反向互逆项粗级奖励与距离的均值，只取掩码内的
格子）、cyc_rew_f_s / cyc_rew_r_s（细化尺度 s）、cyc_in_r（反向粗级掩码内格子比例）、l_main / l_cyc（两项损失，l_cyc 不乘
cyc_w）、t_cyc（新项耗时）。
"""
from __future__ import annotations

import math
import time
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from ..geom import HW, compose
from . import rloo_occ as OC
from .pseudo import affine_norm, grid


class Ctx(NamedTuple):
    """一对图做互逆项要用的现有项中间量。"""
    b: int          # 批内序号
    A: object       # 正向参照仿射（光学 → SAR，原网格）
    k: object       # 正向粗级抽到的锚点 (G,h,w)
    fc: object      # 正向粗级各份的粗匹配位置 (G,2,h,w)，归一化坐标
    m: object       # 正向粗级画面内掩码 (h,w) 或 None
    masks: dict     # 正向细化各级画面内掩码 {尺度: (h,w)} 或 None
    traj: dict      # 正向正常推理轨迹 {尺度: (输入 warp, 输入 certainty, μ)}


def back_dist(x, B):
    """x：落点 (N,2,h,w)（第二张图的归一化坐标）；B：第二张图 → 第一张图的仿射（原网格）→ 落点经 B 映回第一张图后
    离各格子 / 像素中心的距离 (N,h,w)，原网格 px。"""
    N, _, h, w = x.shape
    Bn = torch.from_numpy(affine_norm(B)).to(x.device).float()
    back = torch.einsum("ij,nhwj->nhwi", Bn[:, :2], x.permute(0, 2, 3, 1).float()) + Bn[:, 2]
    return (back - grid(h, w, x.device)).norm(dim=-1) * (HW / 2)


def coarse_reward(fc, B, p):
    """粗级：(奖励, 距离)，形状与长度参数同现有粗级奖励（shape、delta、tau、r_out）。"""
    from .rloo import distance_reward
    d = back_dist(fc, B)
    return distance_reward(d, p["shape"], p["delta"], p["tau"], p["r_out"]), d


def level_reward(x, B, s, p):
    """细化尺度 s：(奖励, 距离)，同 rloo.level_reward 的形状与按级缩放（delta·s/8、tau·s/8、fine_shape）。"""
    from .rloo import distance_reward
    d = back_dist(x, B)
    return distance_reward(d, p["fine_shape"] or p["shape"], p["delta"] * s / 8, p["tau"] * s / 8, p["r_out"]), d


def rev_mask(B, sar, h, w):
    """反向（SAR 为第一张图）h×w 网格的画面内掩码：格子中心不在 SAR 补零区，且 B（SAR → 光学）把它映射到光学画面内。
    sar：SAR 输入 (1,H,W)。"""
    from .rloo import sar_valid
    g = grid(h, w, sar.device)
    v = F.grid_sample(sar_valid(sar)[None, None].float(), g[None], mode="nearest", align_corners=False)[0, 0] > 0.5
    Bn = torch.from_numpy(affine_norm(B)).to(sar.device).float()
    x2 = torch.einsum("ij,hwj->hwi", Bn[:, :2], g) + Bn[:, 2]
    return v & (x2.abs() < 1).all(-1)


def composite_residual(A, B, h, w):
    """B∘A（光学 → SAR → 光学）把 h×w 格中心映射到的位置离原位置的平均距离（px）。"""
    from .rloo import affine_gap
    return affine_gap(compose(B, A), np.c_[np.eye(2), np.zeros(2)], h, w)


def fine_terms(dec, f1, f2, traj, B, sig, p, backward, factor, w, keep=None, gen=None):
    """细化各级的互逆项（逐级局部抽签，同 rloo.fine_direct 的 local，只是奖励换成 level_reward 的映回距离）：sig 里的每一级
    从轨迹 traj 这一级的输入起带梯度重算 μ_s，抽 fine_group 次 x = μ_s + σε，按映回距离打分、留一优势（keep 乘在优势上），
    loss_s = −w · mean(优势 · log π_s)，立刻 backward(loss_s)。返回 (替身项, {尺度: (奖励, 距离)})，替身项同 rloo._surrogate。"""
    from .rloo import _mu_cert, _surrogate, advantage
    H, W = f1[1].shape[-2:]
    leaf1 = {s: f1[s].detach().requires_grad_() for s in sig}
    leaf2 = {s: f2[s].detach().requires_grad_() for s in sig}
    val, out = 0.0, {}
    for s in sig:
        fin, cin, _ = traj[s]
        mu, _ = _mu_cert(dec, s, leaf1[s], leaf2[s], fin, cin, H, W)
        with torch.no_grad():
            x = mu.detach() + sig[s] * torch.randn((p["fine_group"],) + tuple(mu.shape[1:]), device=mu.device,
                                                   generator=gen)
            r, d = level_reward(x, B, s, p)
            adv = advantage(r, p["baseline"])
            if keep is not None:
                adv = adv * keep[s]
        lp = -(x - mu).square().sum(1) / (2 * sig[s] ** 2)
        loss = -w * (adv * lp).mean()
        backward(loss)
        val += float(loss)
        out[s] = (r, d)
    return _surrogate(f1, f2, leaf1, leaf2, factor, val), out


def _put(stats, key, x, sel):
    """x (N,h,w) 在掩码 sel (h,w) 内的均值记进日志；sel 为 None 时取全部，掩码内没有元素时不记。"""
    if sel is None:
        stats.setdefault(key, []).append(float(x.mean()))
    elif bool(sel.any()):
        stats.setdefault(key, []).append(float(x[:, sel].mean()))


def step(term, st, i0, i1, total, ctx, logp_all, f1, f2, B, sig, stats):
    """一步的现有项与互逆项的反向（见模块说明）。term：rloo.Term；total：现有项（尚未反向）；ctx：[Ctx]；
    logp_all：正向 log softmax(gm_cls / temp) (B,C,h,w)（带梯度）；f1、f2：正向特征（带梯度）。返回这一步的损失数值
    （现有项 + cyc_w × 新项；只量不学时只有现有项）。"""
    from .rloo import _logp, advantage
    p, optim = term.p, st.optim
    params = [q for q in term.model.params if q.requires_grad]
    fac = optim.factor()
    t0 = time.time()
    refs = []
    for c in ctx:
        Bref, _ = term.ref_fit(i1[c.b:c.b + 1], i0[c.b:c.b + 1])
        stats.setdefault("cyc_ok", []).append(float(Bref is not None))
        if Bref is not None:
            stats.setdefault("cyc_res", []).append(composite_residual(c.A, Bref, *c.k.shape[-2:]))
            refs.append((c, Bref))
    val = float(total)
    stats["l_main"] = [val]
    if total.requires_grad:     # 有互逆项时保留计算图：正向互逆项还要经同一次前向回传
        optim.backward(total, retain_graph=bool(refs))
    if not refs:
        return val
    g0 = OC.grads(params)
    w = p["cyc_w"] if p["cyc_w"] > 0 else 1.0
    cw = w * p["weight"]
    cyc_val = 0.0
    dec = term.model.model.decoder
    l_fwd = 0.0
    for c, Bref in refs:        # 正向互逆项：沿用现有项的样本与计算图
        b = c.b
        gen = _gen(term, b, i0.device)
        with torch.no_grad():
            r, d = coarse_reward(c.fc, Bref, p)
            adv = advantage(r, p["baseline"])
            if p["frame_mask"] and c.m is not None:
                adv = adv * c.m
            _put(stats, "cyc_rew_f", r, c.m)
            _put(stats, "cyc_d_f", d, c.m)
        l_c = -(adv * _logp(logp_all[b], c.k)).mean() / B
        keep = None if not p["fine_mask"] or c.masks is None else {s: c.masks[s].float() for s in sig}
        fb, gb = {s: x[b:b + 1] for s, x in f1.items()}, {s: x[b:b + 1] for s, x in f2.items()}
        sur, rd = fine_terms(dec, fb, gb, c.traj, Bref, sig, p, optim.backward, fac, cw * p["fine_w"] / B, keep, gen)
        for s, (r_s, _) in rd.items():
            _put(stats, f"cyc_rew_f_{s}", r_s, None if c.masks is None else c.masks[s])
        l_fwd = l_fwd + cw * l_c + sur
    optim.backward(l_fwd)       # 各对的正向互逆项共用同一次前向的计算图，合起来反向一次
    cyc_val += float(l_fwd)
    for c, Bref in refs:        # 反向互逆项：交换两张图另做一遍
        l_r = _reverse(term, st, i0[c.b:c.b + 1], i1[c.b:c.b + 1], c, Bref, cw / B, sig, stats)
        optim.backward(l_r)
        cyc_val += float(l_r)
    stats["l_cyc"] = [cyc_val / w]
    main_gn, cyc_gn, main_all = (x / fac for x in OC.split_norms(params, g0))
    cyc_gn /= w
    if all(map(math.isfinite, (main_gn, cyc_gn, main_all))):     # GradScaler 溢出的步不记
        stats["main_gn"], stats["cyc_gn"], stats["main_gn_all"] = [main_gn], [cyc_gn], [main_all]
        if cyc_gn > 0:
            stats["cyc_coef"] = [main_gn / cyc_gn]
        if main_gn > 0:
            stats["cyc_ratio"] = [p["cyc_w"] * cyc_gn / main_gn]
    if p["cyc_w"] == 0:      # 只量不学：撤掉新项的梯度
        OC.restore(params, g0)
    else:
        val += cyc_val
    torch.cuda.synchronize()
    stats["t_cyc"] = [time.time() - t0]
    return val


def _gen(term, b, device):
    rng = np.random.default_rng([torch.initial_seed() % 2 ** 32, term.calls, b, 255])
    return torch.Generator(device=device).manual_seed(int(rng.integers(2 ** 62)))


def _reverse(term, st, i0, i1, c, Bref, w, sig, stats):
    """一对图的反向互逆项：当前模型在交换后的一对图（SAR 第一）上照正向抽签，光学落点经正向参照 c.A 映回 SAR 打分。
    细化各级在 fine_terms 里已经反向；返回粗级项 + 细化替身项（都已乘 w，细化另乘 fine_w），由调用者反向。
    i0、i1：光学、SAR (1,1,H,W)。"""
    from . import rloo as R
    p, model = term.p, term.model
    dev = i0.device
    rng = np.random.default_rng([torch.initial_seed() % 2 ** 32, term.calls, c.b, 255, 1])
    gen = torch.Generator(device=dev).manual_seed(int(rng.integers(2 ** 62)))
    f1, f2 = R.features(model, i1, i0)
    cls, cert = R.coarse(model, f1, f2)
    cls = cls.float()
    _, C, h, wd = cls.shape
    logp = F.log_softmax(cls / p["temp"], 1)[0]
    with torch.no_grad():
        q = logp.exp()
        k = torch.multinomial(q.reshape(C, -1).T, p["group"], replacement=True, generator=gen).T.reshape(-1, h, wd)
        P = cls.softmax(1)
        traj = {}
        R.refine(model, f1, f2, R.anchor_flow(P, P.argmax(1)), cert.float(), traj)
        fc = R.anchor_flow(P.expand(p["group"], -1, -1, -1), k)
        r, d = coarse_reward(fc, c.A, p)
        m = rev_mask(Bref, i1[0], h, wd)
        stats.setdefault("cyc_in_r", []).append(float(m.float().mean()))
        adv = R.advantage(r, p["baseline"])
        if p["frame_mask"]:
            adv = adv * m
        _put(stats, "cyc_rew_r", r, m)
        _put(stats, "cyc_d_r", d, m)
        masks = {s: rev_mask(Bref, i1[0], *traj[s][0].shape[-2:]) for s in sig}
        keep = {s: masks[s].float() for s in sig} if p["fine_mask"] else None
    l_c = -(adv * R._logp(logp, k)).mean()
    sur, rd = fine_terms(model.model.decoder, f1, f2, traj, c.A, sig, p, st.optim.backward, st.optim.factor(),
                         w * p["fine_w"], keep, gen)
    for s, (r_s, _) in rd.items():
        _put(stats, f"cyc_rew_r_{s}", r_s, masks[s])
    return w * l_c + sur
