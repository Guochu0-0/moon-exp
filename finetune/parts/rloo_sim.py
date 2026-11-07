"""图像相似度约束奖励（#252；[rloo] sim = true）。训练里的挂接见 finetune/parts/rloo.py 模块说明「图像相似度约束奖励」。

每对图照常算完现有项后，对同一组的每份粗级样本（每格抽到的锚点经邻居加权后的粗匹配位置，合起来是一整张对应场）：
1. 用这份样本全部格子的「格心 → 粗匹配位置」拟合一个仿射，RANSAC 阈值同参照拟合（[rloo] ransac，原网格 px）；
   坐标换算同 cexp.sample_matches（原网格中心约定），拟合函数同 rloo.fit_many（moonlib.ransac.fit_affine）。
   与 fit 的不同只在取点：fit 按尺度 1 的可信度抽 num 个像素，这里取粗级全部 h×w 个格子，不看可信度。
2. 把 SAR 按这个仿射拉到光学网格上，在 SAR 非补零区（rloo.sar_valid）内算灰度 NCC（rloo.ncc）。拟合失败或重叠不足
   1000 px 的样本记 −1（NCC 的最低值）。
3. 组内留一优势：每份样本的 NCC − 同组其余样本 NCC 的均值；乘在这份样本粗级的对数概率上（这张对应场所有格子同一个优势，
   不做画面外置零），loss = −mean_{份, 格}(优势 · log π)，对批内取平均。细化各级不加这一项。
总损失 = 现有项 + sim_w × 新项（新项再乘 [rloo] weight，同现有粗级项）。

两项的梯度分开量：新项对可训练参数的梯度先用 autograd.grad 单独算出（保留计算图），再照常反向现有项；
sim_w > 0 时把 sim_w × 新项梯度加进 .grad，sim_w = 0 时只量不学（.grad 只有现有项，与不加这一项逐位相同）。
日志：main_gn、sim_gn 为现有项与（系数为 1 的）新项在两项共用的参数（新项梯度不全为 0 的参数）上的梯度范数（反缩放后、
多卡平均之前、裁剪之前），main_gn_all 为现有项在全部可训练参数上的范数，sim_coef = main_gn / sim_gn（让两者 1:1 的系数），
sim_ratio = sim_w · sim_gn / main_gn（实际施加的比例）。另记每步训练图对上的：sim_ncc（组内 NCC 均值）、sim_ncc_sd
（组内 NCC 标准差）、sim_adv_sd（新项优势的组内标准差）、sim_fail（记 −1 的样本比例）、sim_diff（NCC 最高的样本与离参照
最近的样本不是同一份为 1；离参照的距离取这份样本在参照画面内格子上的平均，参照拟合失败的对不记）、t_sim（新项耗时）。

验证集诊断（sim_diag = N > 0；只记录、不参与训练）：训练图对没有标注，组内 NCC 与真实误差的秩相关改在有标注的 Val 图对上量。
每 N 次调用每张卡取一对 Val 图（第 n 次诊断取有标注 Val 图对的第 (n·卡数 + rank) mod 总数 对，各卡不重复，约一遍），
照训练的扰动（[aug] 的几何扰动与擦除，随机数取自独立发生器）处理 SAR，当前模型照训练同样抽 group 份粗级样本、
按上面 1–2 打 NCC；真实误差 = 这份样本拟合的仿射换回未扰动 SAR 后、标注点的平均配准误差（同评测，拟合失败记 ∞）。
日志：sim_rho（组内 NCC 与真实误差的 Spearman 秩相关，负 = NCC 高的误差小）、sim_vncc（组内 NCC 均值）、sim_verr（组内
真实误差中位数，px）。多卡时日志为各卡的平均。
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..config import ConfigError
from ..geom import HW, compose, inv_affine

PARAMS = {"sim": False,      # 图像相似度约束奖励（#252）：每份粗级样本拟合仿射后按 SAR 与光学的灰度 NCC 打分
          "sim_w": 0.0,      # sim：新项的系数；0 = 只量梯度、不学
          "sim_diag": 50}    # sim：每几次调用在一对有标注的 Val 图上量一次组内 NCC 与真实误差的秩相关；0 = 不量

_POOL = None


def check(p, cfg):
    if not p["sim"]:
        return
    if p["reward"] != "ref" or p["mp_n"] or p["occ"] or p.get("cyc") or p.get("selfpair")             or cfg["optim"]["batch"] != 1 or cfg["optim"]["accum"] != 1:
        raise ConfigError("[rloo] sim 只配 reward = 'ref'、[optim] batch = 1、accum = 1，不与 mp_n、occ、cyc、selfpair 同用")
    if p["sim_w"] < 0 or not isinstance(p["sim_diag"], int) or p["sim_diag"] < 0:
        raise ConfigError("[rloo] sim_w 应 ≥ 0，sim_diag 应为 ≥ 0 的整数")


def fit_coarse(fc, thr):
    """fc：G 份粗 warp (G,2,h,w)（归一化坐标）→ 每份用全部格子的「格心 → 粗匹配位置」RANSAC 拟合的仿射（原网格，
    同 rloo.fit 的约定）或 None。"""
    global _POOL
    from moonlib.ransac import fit_affine

    from .pseudo import grid
    if _POOL is None:
        _POOL = ThreadPoolExecutor(8)
    G, _, h, w = fc.shape
    to_px = lambda x: ((x + 1) * (HW / 2) - 0.5).double().cpu().numpy()
    p0 = to_px(grid(h, w, fc.device).reshape(-1, 2))
    p1 = to_px(fc.float().reshape(G, 2, -1).transpose(1, 2))
    Ms = [np.c_[p0, q, np.ones(len(p0))].astype(np.float32) if np.isfinite(q).all() else None for q in p1]
    return [None if M is None else fit_affine(M, thr)[0] for M in Ms] if G == 1 else \
        list(_POOL.map(lambda M: None if M is None else fit_affine(M, thr)[0], Ms))


def scores(opt, sar, As):
    """一对图（opt、sar：(1,H,W) 输入）上各仿射的灰度 NCC（rloo.ncc；失败或重叠不足记 −1）→ (G,) 张量。"""
    from . import rloo as R
    valid = R.sar_valid(sar)
    return torch.tensor([R.ncc(opt, sar, valid, A) for A in As], device=opt.device)


@torch.no_grad()
def pair_ctx(p, opt, sar, fc, k, d, m, b, stats):
    """一对图：给同组各份粗级样本打 NCC、算留一优势，记日志。d：各份离参照的距离 (G,h,w) 或 None；m：参照画面内掩码 (h,w)
    或 None。返回 (批内序号, k, 优势 (G,))。"""
    from . import rloo as R
    t0 = time.time()
    s = scores(opt, sar, fit_coarse(fc, p["ransac"]))
    adv = R.loo(s)
    stats.setdefault("sim_ncc", []).append(float(s.mean()))
    stats.setdefault("sim_ncc_sd", []).append(float(s.std()))
    stats.setdefault("sim_adv_sd", []).append(float(adv.std()))
    stats.setdefault("sim_fail", []).append(float((s <= -1).float().mean()))
    if d is not None:
        dm = d.reshape(len(d), -1)
        if m is not None and bool(m.any()):
            dm = dm[:, m.reshape(-1)]
        stats.setdefault("sim_diff", []).append(float(int(s.argmax()) != int(dm.mean(1).argmin())))
    stats.setdefault("t_sim", []).append(time.time() - t0)
    return b, k, adv


def apply(term, st, total, logp_all, ctx, B, stats):
    """新项与现有项一起反向（见模块说明）。ctx：[pair_ctx 的返回]。返回这一步的损失数值（现有项 + sim_w × 新项）。"""
    from . import rloo as R
    p, optim = term.p, st.optim
    if term.sim_diag is not None:
        term.sim_diag(term.calls, stats)
    t0 = time.time()
    params = [q for q in term.model.params if q.requires_grad]
    l_s = sum(-(adv[:, None, None] * R._logp(logp_all[b], k)).mean() for b, k, adv in ctx) / B * p["weight"] \
        if ctx else None
    gs = None
    fac = optim.factor()
    if l_s is not None and l_s.requires_grad:
        gs = torch.autograd.grad(l_s * fac, params, retain_graph=True, allow_unused=True)
    if total.requires_grad:
        optim.backward(total)
    val = float(total)
    if gs is None:
        return val
    s_main = s_sim = s_all = 0.0
    for q, g in zip(params, gs):
        gm = 0.0 if q.grad is None else float(q.grad.double().square().sum())
        s_all += gm
        if g is not None:
            gg = float(g.double().square().sum())
            if gg > 0:
                s_sim += gg
                s_main += gm
    main_gn, sim_gn, main_all = (math.sqrt(x) / fac for x in (s_main, s_sim, s_all))
    if all(map(math.isfinite, (main_gn, sim_gn, main_all))):    # GradScaler 溢出的步不记
        stats["main_gn"], stats["sim_gn"], stats["main_gn_all"] = [main_gn], [sim_gn], [main_all]
        if sim_gn > 0:
            stats["sim_coef"] = [main_gn / sim_gn]
        if main_gn > 0:
            stats["sim_ratio"] = [p["sim_w"] * sim_gn / main_gn]
    if p["sim_w"] > 0:
        with torch.no_grad():
            for q, g in zip(params, gs):
                if g is None:
                    continue
                q.grad = p["sim_w"] * g if q.grad is None else q.grad + p["sim_w"] * g
        val += p["sim_w"] * float(l_s)
    stats["t_sim"] = [sum(stats.get("t_sim", [])) + time.time() - t0]     # 各对打分 + 反向
    return val


def _ranks(x):
    """秩（1 起），并列取平均秩。"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        sel = x == v
        r[sel] = r[sel].mean()
    return r


def spearman(a, b):
    """Spearman 秩相关（并列取平均秩，同 scipy.stats.spearmanr）；训练环境没有 scipy。"""
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


class Diag:
    """验证集诊断（见模块说明）。图对在用到时才读。"""

    def __init__(self, p, model, ds):
        self.p, self.model, self.ds = p, model, ds
        self.pairs = None

    def _item(self, n):
        from workbench.dataset import Dataset

        from .. import augment
        from .. import dist as D
        ds, model = self.ds, self.model
        if self.pairs is None:
            self.labels = Dataset(ds.data.root)
            self.pairs = self.labels.labelled("val")
        i = (n * D.world() + D.rank()) % len(self.pairs)
        pair = self.pairs[i]
        opt = ds.map_opt(ds.data.optical("val", pair)).astype(np.float32)
        sar = ds.map_sar(ds.data.sar("val", pair)).astype(np.float32)
        cp_opt, cp_sar = self.labels.checkpoints("val", pair)
        rng = np.random.default_rng([252, n, D.rank()])
        T = augment.sample_T(rng, **ds.aug_kw) if "geo" in ds.aug else np.c_[np.eye(2), np.zeros(2)]
        sar = augment.warp(sar, T)
        if "erase" in ds.aug:
            sar, _ = augment.erase(sar, rng, **ds.erase_kw)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(model.resize(x), np.float32))[None, None].to(model.device)
        return t(opt), t(sar), T, cp_opt, cp_sar, int(rng.integers(2 ** 62))

    @torch.no_grad()
    def __call__(self, calls, stats):
        """第 calls 次调用：每 sim_diag 次量一次，记进 stats。"""
        p, model = self.p, self.model
        if not p["sim_diag"] or calls % p["sim_diag"]:
            return
        from workbench.protocol import pair_error

        from . import rloo as R
        i0, i1, T, cp_opt, cp_sar, seed = self._item(calls // p["sim_diag"] - 1)
        gen = torch.Generator(device=i0.device).manual_seed(seed)
        f1, f2 = R.features(model, i0, i1)
        cls, _ = R.coarse(model, f1, f2)
        cls = cls.float()
        _, C, h, w = cls.shape
        q = (cls[0] / p["temp"]).softmax(0)
        k = torch.multinomial(q.reshape(C, -1).T, p["group"], replacement=True, generator=gen).T.reshape(-1, h, w)
        fc = R.anchor_flow(cls.softmax(1).expand(p["group"], -1, -1, -1), k)
        As = fit_coarse(fc, p["ransac"])
        s = scores(i0[0], i1[0], As).cpu().numpy()
        err = np.array([pair_error(None if A is None else compose(inv_affine(T), A), cp_opt, cp_sar) for A in As])
        stats["sim_vncc"] = [float(s.mean())]
        stats["sim_verr"] = [float(np.median(err))]
        e = np.where(np.isfinite(err), err, 1e9)        # 失败记 ∞：秩相关只看次序
        if np.ptp(s) > 0 and np.ptp(e) > 0:
            stats["sim_rho"] = [spearman(s, e)]
