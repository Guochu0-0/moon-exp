"""同模态自配对约束奖励（#254；[rloo] selfpair = true）。finetune/parts/rloo.py 里只有四处挂接：PARAMS 并入本模块的 PARAMS，
check、data_kw 各调用一次本模块的同名函数，__call__ 末尾调用 step()。

每步、每对图在现有项之外另造一对「自配对」：各一半概率取这对图的光学图（训练批次里的 image0）或扰动之前的 SAR
（数据集另给的 image1_raw，几何、光度、擦除都还没加），记为 X。[aug] 的几何扰动与擦除只加在 SAR 上；photo、blockmask
打开时光学图也会带上它们，E46 的配置两者都关，光学图即原图。抽一个已知随机仿射 T3（augment.sample_T，平移在半径 selfpair_shift 原网格 px 的圆盘内均匀、
旋转 ±selfpair_rot 度、缩放 ±selfpair_scale，默认与难版训练扰动相同），自配对 = (X, T3 变换后的 X)，画面外补 0
（rloo.perturb_sar，同多扰动核对）。X 上每个像素在 T3 X 上的正确位置就是 T3 把它映射到的位置，精确已知。

当前模型在自配对上照原图同样做一遍：粗级抽 group 份、细化各级（levels）从自配对正常推理的轨迹出发照 fine_mode 抽签
或求导，打分方式与现有项相同（粗级 shape / delta，细化 fine_shape、按级缩放），参照仿射换成 T3。画面外：T3 把格子 /
像素中心映射到画面外或补零区的，照现有项处理（frame_mask、fine_mask 时优势置零；cert_bce、fine_cert_bce > 0 时
certainty 对这张掩码做 BCE）。掩码由一张全 1 的图经同样的 T3 变换后按 rloo.sar_valid 判定，与 X 的灰度无关
（光学图里本来就暗的像素不当作补零区）。

总损失 = 现有项 + selfpair_w × 新项（新项的粗级、BCE 与细化部分都乘 selfpair_w；细化另照现有项乘 fine_w）。两项的梯度
照遮挡约束奖励（#249）分开量：先对现有项反向、记下各参数梯度，再对新项反向（rloo_occ.grads / split_norms / restore）。
selfpair_w = 0 时只量不学：新项照算、量完梯度后撤掉。新项的随机数（选模态、T3、粗级与细化的抽签）取自以
(torch 初始种子, 调用次数, 批内序号, 254) 起的独立发生器，不动训练的随机数流。

日志：main_gn、sp_gn 为现有项与（系数 1 的）新项在共用参数上的梯度范数，main_gn_all 为现有项在全部可训练参数上的范数，
sp_coef = main_gn / sp_gn（让两者 1:1 的系数），sp_ratio = selfpair_w · sp_gn / main_gn（实际施加的比例），t_sp（新项耗时）；
sp_sar（这一步选了 SAR 的比例）、sp_in（粗格画面内比例）；以下只在画面内的格子 / 像素上统计：sp_rew、sp_rew_sd（粗级奖励
均值与同一格各份之间的标准差）、sp_d（粗级抽到的位置离 T3 真值的平均距离 px）、sp_dmode（每格取最高分位置时的距离），
细化各级 sp_rew_s、sp_rew_sd_s（local 时为同一像素各次抽签之间的标准差）、sp_d_s，以及正常推理轨迹上尺度 s 的输出离
真值的平均距离 sp_dmu_s。
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import augment
from ..config import ConfigError
from . import rloo_occ as OC

PARAMS = {"selfpair": False,      # 同模态自配对约束奖励（#254）：每对另加 1 对（X，T3 X），按已知的 T3 打分
          "selfpair_w": 0.0,      # selfpair：新项的系数；0 = 只量梯度、不学
          "selfpair_shift": 128.0,  # selfpair：T3 平移的圆盘半径（原网格 px；同难版训练扰动）
          "selfpair_rot": 20.0,   # selfpair：T3 旋转范围（± 度）
          "selfpair_scale": 0.2}  # selfpair：T3 缩放范围（±）


def check(p, cfg):
    if not p["selfpair"]:
        return
    if (p["reward"] != "ref" or not p["levels"] or p["fine_mode"] == "chain" or p["occ"] or p["mp_n"] or p["gate"]
            or p["spread"] or p["trace"] or p["cos_every"] or cfg["optim"]["batch"] != 1 or cfg["optim"]["accum"] != 1):
        raise ConfigError("[rloo] selfpair 只配 reward = 'ref'、非空的 levels 与 fine_mode = 'grad' / 'local'、"
                          "[optim] batch = 1、accum = 1，不与 occ、mp_n、gate、spread、trace、cos_every 同用")
    if p["selfpair_w"] < 0 or p["selfpair_shift"] < 0 or p["selfpair_rot"] < 0 or not 0 <= p["selfpair_scale"] < 1:
        raise ConfigError("[rloo] selfpair_w、selfpair_shift、selfpair_rot 应 ≥ 0，selfpair_scale 在 [0, 1)")


def data_kw(p):
    return {"raw_sar": True} if p.get("selfpair") else {}


def make_pair(opt, sar_raw, rng, p):
    """opt、sar_raw：(1,H,W) 输入网格上的光学图与扰动之前的 SAR → (X, T3 X, 全 1 图经 T3 变换的结果, T3, 是否选了 SAR)。
    T3 为原网格角点约定的 2×3（X → T3 X）。"""
    from .rloo import perturb_sar
    use_sar = bool(rng.random() < 0.5)
    x = sar_raw if use_sar else opt
    T = augment.sample_T(rng, p["selfpair_shift"], p["selfpair_rot"], p["selfpair_scale"], disk=True)
    return x, perturb_sar(x, T), perturb_sar(torch.ones_like(x), T), T, use_sar


def step(term, st, total, B, sig, stats):
    """现有项 total 先反向，再逐对算新项并反向，量两项的梯度范数（见模块说明）。返回这一步的损失数值
    （现有项 + selfpair_w × 新项；只量不学时只有现有项）。"""
    p, optim = term.p, st.optim
    params = [q for q in term.model.params if q.requires_grad]
    fac = optim.factor()
    if total.requires_grad:
        optim.backward(total)
    val = float(total)
    g0 = OC.grads(params)
    w = p["selfpair_w"] if p["selfpair_w"] > 0 else 1.0
    t0 = time.time()
    i0, _ = st.images()
    raw = st.batch["image1_raw"]
    sp_val = 0.0
    for b in range(B):
        l_s = _pair_loss(term, st, i0[b], raw[b].to(i0.device), b, w / B, sig, stats)
        optim.backward(l_s)
        sp_val += float(l_s)
    main_gn, sp_gn, main_all = (x / fac for x in OC.split_norms(params, g0))
    sp_gn /= w
    if all(map(np.isfinite, (main_gn, sp_gn, main_all))):     # GradScaler 溢出的步不记
        stats["main_gn"], stats["sp_gn"], stats["main_gn_all"] = [main_gn], [sp_gn], [main_all]
        if sp_gn > 0:
            stats["sp_coef"] = [main_gn / sp_gn]
        if main_gn > 0:
            stats["sp_ratio"] = [p["selfpair_w"] * sp_gn / main_gn]
    if p["selfpair_w"] == 0:      # 只量不学：撤掉新项的梯度
        OC.restore(params, g0)
    else:
        val += sp_val
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats["t_sp"] = [time.time() - t0]
    return val


def _pair_loss(term, st, opt, sar_raw, b, w, sig, stats):
    """一对图的新项：造自配对，当前模型照现有项抽签、对 T3 打分。细化各级在 fine_direct 里已经反向；返回粗级项（含 BCE）
    + 细化替身项（都已乘 w · weight，细化另乘 fine_w），由调用者反向。opt、sar_raw：(1,H,W)。"""
    from . import rloo as R
    p, model = term.p, term.model
    dev = opt.device
    rng = np.random.default_rng([torch.initial_seed() % 2 ** 32, term.calls, b, 254])
    gen = torch.Generator(device=dev).manual_seed(int(rng.integers(2 ** 62)))
    x, xt, ones_t, T, use_sar = make_pair(opt, sar_raw, rng, p)
    stats.setdefault("sp_sar", []).append(float(use_sar))
    f1, f2 = R.features(model, x[None], xt[None])
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
        r, d = R.ref_cells(fc, T, p["delta"], p["tau"], p["r_out"], p["shape"])
        _, d_mode = R.ref_cells(R.anchor_flow(P, P.argmax(1)), T, p["delta"], p["tau"], p["r_out"], p["shape"])
        m = R.in_frame(T, ones_t, h, wd)
        adv = R.advantage(r, p["baseline"])
        if p["frame_mask"]:
            adv = adv * m
        stats.setdefault("sp_in", []).append(float(m.float().mean()))
        if bool(m.any()):
            stats.setdefault("sp_rew", []).append(float(r[:, m].mean()))
            stats.setdefault("sp_rew_sd", []).append(float(r.std(0)[m].mean()))
            stats.setdefault("sp_d", []).append(float(d[:, m].mean()))
            stats.setdefault("sp_dmode", []).append(float(d_mode[:, m].mean()))
        masks = {s_: R.in_frame(T, ones_t, *traj[s_][0].shape[-2:]) for s_ in sig}
        keep = {s_: masks[s_].float() for s_ in sig} if p["fine_mask"] else None
    loss = -(adv * R._logp(logp, k)).mean()
    if p["cert_bce"]:
        l_cert = F.binary_cross_entropy_with_logits(cert[0, 0].float(), m.float())
        loss = loss + p["cert_bce"] * l_cert
        stats.setdefault("sp_l_cert", []).append(float(l_cert))
    bce = w * p["weight"] * p["fine_cert_bce"]
    sur, rd, _ = R.fine_direct(model.model.decoder, f1, f2, traj, T, sig, p, st.optim.backward, st.optim.factor(),
                               w * p["weight"] * p["fine_w"], keep, masks, bce, gen=gen)
    with torch.no_grad():
        for s_, (r_s, d_s) in rd.items():
            sel = masks[s_]
            if not bool(sel.any()):
                continue
            stats.setdefault(f"sp_rew_{s_}", []).append(float(r_s[:, sel].mean()))
            if r_s.shape[0] > 1:
                stats.setdefault(f"sp_rew_sd_{s_}", []).append(float(r_s.std(0)[sel].mean()))
            stats.setdefault(f"sp_d_{s_}", []).append(float(d_s[:, sel].mean()))
            stats.setdefault(f"sp_dmu_{s_}", []).append(
                float(R.level_reward(traj[s_][2], T, s_, p)[1][:, sel].mean()))
    return w * p["weight"] * loss + sur
