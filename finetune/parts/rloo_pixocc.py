"""遮挡约束奖励（逐像素，「【RoMa·RL】遮挡约束奖励（逐像素）」#250），[rloo] pix_occ 打开时由 rloo.Term 调用。

现有奖励只要求模型靠近EMA 参照在原图上拟合出的一个仿射。这一项另外检查对遮挡稳不稳：同一对图生成 1 个遮挡版本，
没被遮住的像素在遮挡版本上指向 SAR 的位置，要与EMA 参照在原图上给这个像素的逐像素对应相同（不经过仿射）。
它只约束稳定性，不保证对应正确。

遮挡照 E40（runs/E40/code/infer.py）：光学图上边长 pix_occ_side（229）px 的正方形，左上角在图内均匀抽（整数 px，
原网格角点约定）；SAR 上遮住这块正方形经参照仿射映射过去的四边形；被遮处填这张图自己全图均值、标准差（总体标准差）
的高斯噪声，截到 [0, 1]。在网络输入网格（560）上做：像素中心落在多边形内（含边界）即被遮。位置与噪声的随机数取自以
(torch 初始种子, 调用次数) 起的独立发生器，不动训练的随机数流；各卡的 torch 初始种子不同，所以各卡遮的位置不同。

打分：当前模型在遮挡版本上正常推理一次（细化各级的输入取自这次推理），粗级抽 group 次、细化各级从这一级的输入抽
fine_group 次（同 rloo 的 local），抽到的点离这个像素的逐像素参照的距离 d（原网格 px）按现有奖励的形式打分：
粗级 shape / delta，细化 fine_shape / delta·s/8。各自 RLOO。不打分（优势置零，分母不变）的像素：光学图上像素中心在
遮挡正方形内的；逐像素参照落在 SAR 画面外、补零区（rloo.sar_valid，按原 SAR）或 SAR 被遮的四边形内的。
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from ..geom import HW
from . import rloo as R
from .pseudo import grid


def to_px(g):
    """归一化坐标（align_corners=False，像素外沿为 ±1）→ 原网格角点约定 px。"""
    return (g + 1) * (HW / 2)


def square(rng, side):
    """左上角在图内均匀抽的正方形（整数 px，同 E40）→ 角点约定的 4×2 顶点（原网格），顺时针。"""
    x, y = (int(v) for v in rng.integers(0, HW - side + 1, 2))
    return np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], np.float64)


def map_poly(poly, A):
    """多边形顶点（原网格角点约定）经仿射 A（光学 → SAR，角点约定）映射。"""
    A = np.asarray(A, np.float64)
    return poly @ A[:, :2].T + A[:, 2]


def inside(pts, poly):
    """pts (...,2) 原网格角点约定坐标；poly 凸多边形 (n,2)，顶点按顺序（顺、逆时针都行）→ bool (...)，边界算在内。"""
    P = torch.as_tensor(poly, dtype=torch.float64, device=pts.device)
    e = torch.roll(P, -1, 0) - P                                    # (n,2)
    rel = pts.double()[..., None, :] - P                            # (...,n,2)
    cr = e[:, 0] * rel[..., 1] - e[:, 1] * rel[..., 0]              # (...,n)
    return (cr >= 0).all(-1) | (cr <= 0).all(-1)


def occlude(img, poly, gen):
    """img (1,H,W) ∈ [0,1]（网络输入网格）；poly 原网格角点约定的多边形 → (遮挡后的图, 掩码 (H,W) bool)。
    被遮处填 N(全图均值, 全图总体标准差²) 的噪声，截到 [0, 1]。"""
    H, W = img.shape[-2:]
    m = inside(to_px(grid(H, W, img.device)), poly)
    x = img.float()
    noise = x.mean() + x.std(unbiased=False) * torch.randn(x.shape, generator=gen, device=x.device)
    return torch.where(m, noise.clamp(0, 1), x).to(img.dtype), m


def in_sar(tgt, valid):
    """tgt (1,2,h,w) 归一化坐标的 SAR 位置 → (h,w) bool：在画面内且不在补零区（valid：SAR 输入网格上的 sar_valid）。"""
    x2 = tgt[0].permute(1, 2, 0).float()
    v = F.grid_sample(valid[None, None].float(), x2[None], mode="nearest", align_corners=False)[0, 0]
    return (x2.abs() < 1).all(-1) & (v > 0.5)


def keep_mask(tgt, sq, quad, valid):
    """打分的像素 (h,w) bool：光学像素中心不在遮挡正方形 sq 内，逐像素参照 tgt 在 SAR 画面内、不在补零区、不在 SAR 被遮的四边形 quad 内。"""
    h, w = tgt.shape[-2:]
    opt_ok = ~inside(to_px(grid(h, w, tgt.device)), sq)
    return opt_ok & in_sar(tgt, valid) & ~inside(to_px(tgt[0].permute(1, 2, 0)), quad)


def pix_reward(x, tgt, delta, shape):
    """x (N,2,h,w) 抽到的点、tgt (1,2,h,w) 逐像素参照（归一化坐标）→ (奖励 (N,h,w), 距离 d (N,h,w) 原网格 px)。"""
    d = (x.float() - tgt.float()).norm(dim=1) * (HW / 2)
    return R.distance_reward(d, shape, delta), d


def _q(d):
    """一组距离的 (均值, 中位数, 90% 分位)；空时 None。"""
    d = d.flatten().float()
    if d.numel() == 0:
        return None
    return float(d.mean()), float(d.median()), float(torch.quantile(d[::max(1, d.numel() // 100000)], 0.9))


def pair_loss(term, i0, i1, A, pix, w, backward, factor, rng, gen):
    """一对图（i0、i1：(1,H,W) 输入）的新项。A：参照仿射；pix：(粗级参照 warp (1,2,h,w), {尺度: 参照这一级输出})。
    细化各级在 fine_direct 里立刻 backward（乘 w·fine_w）；返回 (w × 粗级项 + 替身项, 日志量)。"""
    p, model = term.p, term.model
    tc, tl = pix
    sq = square(rng, p["pix_occ_side"])
    quad = map_poly(sq, A)
    o2, mo = occlude(i0, sq, gen)
    s2, ms = occlude(i1, quad, gen)
    valid = R.sar_valid(i1)
    f1, f2 = R.features(model, o2[None], s2[None])
    cls, cert = R.coarse(model, f1, f2)
    cls = cls.float()
    C, h, w_ = cls.shape[1:]
    logp = F.log_softmax(cls / p["temp"], 1)[0]
    st = {"occ_opt": float(mo.float().mean()), "occ_sar": float(ms.float().mean())}
    with torch.no_grad():
        q = logp.exp()
        k = torch.multinomial(q.reshape(C, -1).T, p["group"], replacement=True).T.reshape(-1, h, w_)
        P = cls.softmax(1)
        fc = R.anchor_flow(P.expand(p["group"], -1, -1, -1), k)
        r, d = pix_reward(fc, tc, p["delta"], p["shape"])
        kc = keep_mask(tc, sq, quad, valid)
        adv = R.advantage(r, p["baseline"]) * kc
        traj = {}
        R.refine(model, f1, f2, R.anchor_flow(P, P.argmax(1)), cert.float(), traj)
    loss_c = -(adv * R._logp(logp, k)).mean()
    lv = p["levels"]
    sig = {s: p["k"] * s / (HW / 2) for s in lv}
    fshape = p["fine_shape"] or p["shape"]
    keeps = {s: keep_mask(tl[s], sq, quad, valid) for s in lv}
    rew = {s: (lambda x, s=s: pix_reward(x, tl[s], p["delta"] * s / 8, fshape)) for s in lv}
    sur, rd, _ = R.fine_direct(model.model.decoder, f1, f2, traj, None, sig, p, backward, factor, w * p["fine_w"],
                               {s: m.float() for s, m in keeps.items()}, rew=rew)
    st.update(occ_rew_c=float(r.mean()), occ_d_c=float(d.mean()), occ_keep_c=float(kc.float().mean()),
              occ_l_c=float(loss_c))
    for s in lv:
        r_s, d_s = rd[s]
        st.update({f"occ_rew_{s}": float(r_s.mean()), f"occ_d_{s}": float(d_s.mean()),
                   f"occ_keep_{s}": float(keeps[s].float().mean())})
    # 逐像素参照与参照仿射的分歧：各级参照点离参照仿射映射点的距离（参照点在 SAR 画面内的像素）
    for name, t in [("c", tc)] + [(str(s), tl[s]) for s in lv]:
        _, dv = R.ref_cells(t, A, 1.0, 1.0, 0.0, "gauss")
        qs = _q(dv[0][in_sar(t, valid)])
        if qs is not None:
            st.update({f"occ_dv_{name}": qs[0], f"occ_dvm_{name}": qs[1], f"occ_dv90_{name}": qs[2]})
    return w * loss_c + sur, st


def _params(model):
    ps = getattr(model, "params", None)
    if ps is None:
        ps = [q for q in model.model.parameters() if q.requires_grad]
    return list(ps)


def step(term, st, total, job):
    """rloo.Term 一步的收尾（pix_occ 打开时）：先反向现有项 total，再算、反向新项。job：(i0, i1, A_ref, pix) 或 None
    （参照拟合失败、检查没过时不加新项）。返回 (这一步损失的数值, 日志量)。"""
    p, opt = term.p, st.optim
    opt.backward(total)
    val, out = float(total), {}
    apply = p["pix_occ_apply"]
    w = p["pix_occ_w"] if apply else 1.0
    measure = (not apply or (p["pix_occ_gn_every"] and (term.calls - 1) % p["pix_occ_gn_every"] == 0)) and w > 0
    out["occ_on"] = float(job is not None)
    if job is None:
        return val, out
    params = _params(term.model)
    g_old = [None if q.grad is None else q.grad.detach().clone() for q in params] if measure else None
    t0 = time.time()
    key = [torch.initial_seed() % 2 ** 32, term.calls]
    rng = np.random.default_rng(key)
    dev = job[0].device
    gen = torch.Generator(device=dev).manual_seed(int(np.random.default_rng(key + [1]).integers(2 ** 62)))
    fac = opt.factor()
    l_new, stats = pair_loss(term, *job, w, opt.backward, fac, rng, gen)
    opt.backward(l_new)
    out.update(stats)
    if apply:
        val += float(l_new)
    if measure:
        so = sn = ao = an = 0.0
        for q, go in zip(params, g_old):
            gq = q.grad
            if gq is None:
                continue
            gn = gq.detach().float() - (0 if go is None else go.float())
            no = 0.0 if go is None else float(go.float().square().sum())
            nn = float(gn.square().sum())
            ao, an = ao + no, an + nn
            if no > 0 and nn > 0:
                so, sn = so + no, sn + nn
        g_o, g_n = so ** 0.5 / fac, sn ** 0.5 / (fac * w)
        out.update(occ_g_old=g_o, occ_g_new=g_n, occ_g_old_all=ao ** 0.5 / fac, occ_g_new_all=an ** 0.5 / (fac * w))
        if g_n > 0 and np.isfinite(g_o) and np.isfinite(g_n):
            out["occ_g_ratio"] = g_o / g_n
        if not apply:
            for q, go in zip(params, g_old):
                q.grad = go
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out["t_occ"] = time.time() - t0
    return val, {k: round(float(v), 5) if abs(float(v)) < 1e6 else float(v) for k, v in out.items()}
