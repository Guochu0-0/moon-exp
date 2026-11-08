"""已知扰动等变约束奖励（#253；[rloo] eqv = true）：配置检查、新项的计算与两项梯度的分开测量。挂接点在
finetune/parts/rloo.py 的 Term.__call__（只多收集 OccCtx、最后调用 apply）。

做法：每对图照常算完现有项后，另抽一个已知随机仿射 T2（augment.sample_T：绕中心旋转 ±eqv_rot 度、缩放 ±eqv_scale、
平移在半径 eqv_shift 原网格 px 的圆盘内均匀），把已扰动的 SAR 再按 T2 变换（rloo.perturb_sar，图外补 0，同训练扰动）。
当前模型在（光学，T2 变换后的 SAR）上照原图同样做一遍：粗级抽 group 份、细化各级（levels）从这一版正常推理的轨迹
抽签，打分方式与现有项相同，参照换成 T2∘A_ref（A_ref 为原图上的参照仿射，光学 → 原 SAR；T2∘A_ref 为光学 → 变换后
的 SAR）。画面外的格子 / 像素按 T2∘A_ref 与变换后的 SAR（含补零区）判定，照现有项的 frame_mask / fine_mask 置零。
不做 certainty 的 BCE。参照拟合失败的对不做这一项。
总损失 = 现有项 + eqv_w × 新项（新项的粗级与细化部分都乘 eqv_w；细化部分另照现有项乘 fine_w）。两项分开反向、
量梯度范数的做法与遮挡约束奖励（#249，rloo.Term._occ）相同，共用 rloo_occ 的 grads / split_norms / restore；
eqv_w = 0 时只量不学。新项的随机数（T2、粗级与细化的抽签）取自以 (torch 初始种子, 调用次数, 批内序号, 253) 起的独立
发生器，不动训练的随机数流。

日志：main_gn、eqv_gn（现有项与系数为 1 的新项在两项共用参数上的梯度范数）、main_gn_all、eqv_coef = main_gn / eqv_gn、
eqv_ratio = eqv_w · eqv_gn / main_gn、t_eqv；eqv_rot / eqv_scl / eqv_sh（T2 的旋转角绝对值（度）、|缩放 − 1|、平移长度 px）；
eqv_rew 与 eqv_d（变换后一版粗级抽签的奖励均值、抽到的位置离 T2∘A_ref 的平均距离，只取按 T2∘A_ref 在画面内的格子）、
eqv_d0（原图粗级抽到的位置离 A_ref 的平均距离，只取按 A_ref 在画面内的格子）；细化尺度 s 的 eqv_rew_s，
eqv_dmu_s 与 eqv_dmu0_s（变换后 / 原图正常推理轨迹上尺度 s 的 μ 离各自参照的平均距离，各取各自画面内的像素）。
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import augment
from ..config import ConfigError
from ..geom import compose
from . import rloo_occ as OC

PARAMS = {"eqv": False,       # 已知扰动等变约束奖励（#253）
          "eqv_w": 0.0,       # eqv：新项的系数；0 = 只量梯度、不学
          "eqv_rot": 10.0,    # eqv：T2 的旋转范围（± 度）
          "eqv_scale": 0.1,   # eqv：T2 的缩放范围（±）
          "eqv_shift": 64.0}  # eqv：T2 的平移半径（原网格 px，圆盘内均匀）


def check(cfg):
    p = cfg["rloo"]
    if not p["eqv"]:
        return
    if (p["reward"] != "ref" or not p["levels"] or p["fine_mode"] == "chain" or p["mp_n"] or p["gate"] or p["spread"]
            or p["trace"] or p["cos_every"] or p["occ"] or p.get("sim") or p.get("cyc") or p.get("selfpair")
            or cfg["optim"]["batch"] != 1 or cfg["optim"]["accum"] != 1):
        raise ConfigError("[rloo] eqv 只配 reward = 'ref'、非空的 levels 与 fine_mode = 'grad' / 'local'、[optim] batch = 1、"
                          "accum = 1，不与 mp_n、gate、spread、trace、cos_every、occ、sim、cyc、selfpair 同用")
    if p["eqv_w"] < 0 or p["eqv_rot"] < 0 or p["eqv_shift"] < 0 or not 0 <= p["eqv_scale"] < 1:
        raise ConfigError("[rloo] eqv_w、eqv_rot、eqv_shift 应 ≥ 0，eqv_scale 在 [0, 1)")


def draw_T(rng, p):
    """已知随机仿射 T2（原网格角点约定，原 → 新），同训练扰动的 sample_T，平移在圆盘内均匀。"""
    return augment.sample_T(rng, p["eqv_shift"], p["eqv_rot"], p["eqv_scale"], disk=True)


def apply(term, st, i0, i1, total, ctx, B, sig, stats):
    """现有项 total 先反向，再逐对算新项并反向，量两项的梯度范数（同 rloo.Term._occ）。ctx：[rloo.OccCtx]。
    返回这一步的损失数值（现有项 + eqv_w × 新项；只量不学时只有现有项）。"""
    p, optim = term.p, st.optim
    params = [q for q in term.model.params if q.requires_grad]
    fac = optim.factor()
    if total.requires_grad:
        optim.backward(total)
    val = float(total)
    if not ctx:
        return val
    g0 = OC.grads(params)
    w = p["eqv_w"] if p["eqv_w"] > 0 else 1.0
    t0 = time.time()
    new = 0.0
    for c in ctx:
        l_e = pair(term, st, i0[c.b:c.b + 1], i1[c.b:c.b + 1], c, w / B, sig, stats)
        optim.backward(l_e)
        new += float(l_e)
    main_gn, eqv_gn, main_all = (x / fac for x in OC.split_norms(params, g0))
    eqv_gn /= w
    if all(map(math.isfinite, (main_gn, eqv_gn, main_all))):     # GradScaler 溢出的步不记
        stats["main_gn"], stats["eqv_gn"], stats["main_gn_all"] = [main_gn], [eqv_gn], [main_all]
        if eqv_gn > 0:
            stats["eqv_coef"] = [main_gn / eqv_gn]
        if main_gn > 0:
            stats["eqv_ratio"] = [p["eqv_w"] * eqv_gn / main_gn]
    if p["eqv_w"] == 0:      # 只量不学：撤掉新项的梯度
        OC.restore(params, g0)
    else:
        val += new
    torch.cuda.synchronize()
    stats["t_eqv"] = [time.time() - t0]
    return val


def _add(stats, key, x):
    stats.setdefault(key, []).append(float(x))


def pair(term, st, i0, i1, c, w, sig, stats):
    """一对图的新项：抽 T2、变换 SAR，当前模型照现有项抽签、对 T2∘A 打分，画面外优势置零。细化各级在 fine_direct 里
    已经反向；返回粗级项 + 细化替身项（都已乘 w · weight，细化另乘 fine_w），由调用者反向。i0、i1：(1,1,H,W)。"""
    from . import rloo as R     # 运行时取：rloo 导入本模块，且测试会替换 rloo 里的函数
    p, model = term.p, term.model
    b, A, m0, masks0, _, d0, traj0 = c
    dev = i0.device
    rng = np.random.default_rng([torch.initial_seed() % 2 ** 32, term.calls, b, 253])
    gen = torch.Generator(device=dev).manual_seed(int(rng.integers(2 ** 62)))
    T = draw_T(rng, p)
    L = T[:, :2]
    _add(stats, "eqv_rot", abs(math.degrees(math.atan2(L[1, 0], L[0, 0]))))
    _add(stats, "eqv_scl", abs(math.sqrt(abs(np.linalg.det(L))) - 1))
    _add(stats, "eqv_sh", np.linalg.norm(T[:, 2] - (np.eye(2) - L) @ np.full(2, R.HW / 2)))
    s1 = R.perturb_sar(i1[0], T)
    A2 = compose(T, A)
    f1, f2 = R.features(model, i0, s1[None])
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
        r, d = R.ref_cells(fc, A2, p["delta"], p["tau"], p["r_out"], p["shape"])
        m = R.in_frame(A2, s1, h, wd)
        adv = R.advantage(r, p["baseline"])
        if p["frame_mask"]:
            adv = adv * m
        if bool(m.any()):
            _add(stats, "eqv_rew", r[:, m].mean())
            _add(stats, "eqv_d", d[:, m].mean())
        if m0 is None:
            m0 = R.in_frame(A, i1[0], h, wd)
        if bool(m0.any()):
            _add(stats, "eqv_d0", d0[:, m0].mean())
        masks = {s_: R.in_frame(A2, s1, *traj[s_][0].shape[-2:]) for s_ in sig}
        if masks0 is None:
            masks0 = {s_: R.in_frame(A, i1[0], *traj0[s_][0].shape[-2:]) for s_ in sig}
        keep = {s_: v.float() for s_, v in masks.items()} if p["fine_mask"] else None
    l_c = -(adv * R._logp(logp, k)).mean()
    sur, rd, _ = R.fine_direct(model.model.decoder, f1, f2, traj, A2, sig, p, st.optim.backward, st.optim.factor(),
                               w * p["weight"] * p["fine_w"], keep, gen=gen)
    with torch.no_grad():
        for s_, (r_s, _) in rd.items():
            if bool(masks[s_].any()):
                _add(stats, f"eqv_rew_{s_}", r_s[:, masks[s_]].mean())
                _add(stats, f"eqv_dmu_{s_}", R.level_reward(traj[s_][2], A2, s_, p)[1][:, masks[s_]].mean())
            if bool(masks0[s_].any()):
                _add(stats, f"eqv_dmu0_{s_}", R.level_reward(traj0[s_][2], A, s_, p)[1][:, masks0[s_]].mean())
    return w * p["weight"] * l_c + sur
