"""自洽坍缩诊断（「【RoMa·RL】诊断：自洽坍缩的形态与成因」#206）：只读存档，不训练。

    python runs/E26/code/probe.py <任务>      # 任务 = morph_<方法> | learn_<方法>；产物写进 runs/E26/raw/<任务>.*

方法（METHODS）：一个方法 = 一份存档。七个坍缩运行（E15 loo_t02 / none_t02 / none_t05、E19 clip 取第 1000、4000 步；
E22 f0002、E23 c16_s1、E25 f1_s1 取第 500…4000 步共 8 个），健康对照（E19 gauss8、E23 c16_s2、E25 fr1_s1 的第 4000 步），
以及未训练的模型 zs（AnyMatch 发布的 RoMa 权重）。

载入：照各运行自己的配置（runs/<实验>/configs/<运行>.toml）构造训练态模型（DINOv2 qkv 挂 LoRA），再把存档的
'model' 与 'dinov2_lora' 原样装回去，得到与训练时同一个网络（LoRA 不合并）。zs 用 E19 gauss8 的配置、不装存档
（LoRA 的 B 为 0，与发布权重相同）。冻结参照的运行（fr1_s1）先构造训练成分、再装存档，冻结副本因此是初始模型，同训练时。

morph_<方法>（票的方法 1，验证集全部 652 对有标注的，不加扰动）：对每对做一次训练时的「正常推理」（560 输入一遍：
粗级每格取最高分位置、经邻居加权得粗 warp，跑尺度 16…1 的细化；rloo.features / coarse / refine），然后
- 粗匹配：每格温度 1 的熵（nat）、最高分位置的概率、最高分位置落在 SAR 上的位置（粗 warp，原网格 px）；
  逐对汇总为熵与最高分概率的中位数、最高分位置互不相同的个数、最常见那个位置占的格子比例、粗 warp 落点的离散程度
  （两轴标准差的均方根，px）与质心；
- 细化：尺度 1 warp 落点（560 网格每个像素）的离散程度与质心，落点覆盖 SAR 画面的比例（16 px 方格里有落点的比例），
  可信度（sigmoid）的均值与 > 0.05（推理抽点的阈值）的比例；
- 自估仿射：同训练拟合参照的做法（按可信度抽 5000 点、RANSAC 3 px），A_self 的奇异值、行列式、把图块中心 (256, 256)
  映射到的位置、相对标注的误差（与工作台同一个函数）；粗 warp 离 A_self 的平均距离（同训练日志 d_ref 的口径，只是
  不抽签、用最高分位置）；
- 真值：标注点最小二乘仿射（同 E17）。
另把固定的 6 对（Val 每个 ROI 的第一对）的可信度图、粗级熵图、尺度 1 warp 与粗级最高分位置写成中间结果（inter/<方法>/）。
产物：raw/morph_<方法>.npz（逐对数组）与 .json（pair 列表、耗时）。

learn_<方法>（票的方法 2，验证集 LEARN_N 对，均匀取样，不加扰动）：用这个运行自己的训练成分（rloo.Term，温度、每对抽签
次数、奖励、细化抽签照原配置）在每对上算一步训练损失并反向，不更新参数，记
- 训练日志同款的监控量（rew、rew_sd、d_ref、H、pmode、nonmode、d_anchor、细化各级的 rew_s / d_s 等）；
- 优势（RLOO 权重；baseline = none 时即奖励本身）：粗级与各加噪级的绝对值均值，以及同一位置各份奖励完全相同（优势全为 0）
  的比例；
- 梯度范数：全部参数，以及按部分（粗级 decoder、各级细化层、VGG、LoRA 的 A 与 B），已除去 GradScaler 的放大倍数；
  另记各部分参数的范数。GradScaler 放大后出现非有限梯度时，按更小的放大倍数重算这一对（同一随机种子）。
每对前把 torch 的种子设为 1000 + 序号，抽签可复现。产物：raw/learn_<方法>.npz 与 .json。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from finetune import config as C                      # noqa: E402
from finetune.data import PairSet                     # noqa: E402
from finetune.geom import HW                          # noqa: E402
from finetune.models import roma                      # noqa: E402
from finetune.optim import Optim                      # noqa: E402
from finetune.parts import rloo                       # noqa: E402
from finetune.parts.pseudo import affine_norm, grid   # noqa: E402
from workbench import protocol                        # noqa: E402

MAIN = Path("/remote-home/xufang/YGC/moon-exp/runs")


def _methods():
    out = {}   # 方法 → (实验, 运行, 步；zs 的步为 0)
    for e, r in (("E15", "loo_t02"), ("E15", "none_t02"), ("E15", "none_t05"), ("E19", "clip")):
        for s in (1000, 4000):
            out[f"{e.lower()}_{r}_{s}"] = (e, r, s)
    for e, r in (("E22", "f0002"), ("E23", "c16_s1"), ("E25", "f1_s1")):
        for s in range(500, 4001, 500):
            out[f"{e.lower()}_{r}_{s}"] = (e, r, s)
    for e, r in (("E19", "gauss8"), ("E23", "c16_s2"), ("E25", "fr1_s1")):
        out[f"{e.lower()}_{r}_4000"] = (e, r, 4000)
    out["zs"] = ("E19", "gauss8", 0)
    return out


METHODS = _methods()
TASKS = [f"{t}_{m}" for t in ("morph", "learn") for m in METHODS]
LEARN_N = 100
CELL = 16            # 覆盖率的方格边长（原网格 px）
THRESH = 0.05        # 推理抽点的可信度阈值（cexp.sample_matches）


# ---------- 载入 ----------

def load(method, weights_root):
    """→ (模型, 展开后的配置, 训练成分 Term)。见模块说明「载入」。"""
    exp, run, step = METHODS[method]
    cfg = C.load(REPO / "runs" / exp / "configs" / f"{run}.toml")
    model = roma.Model(cfg, weights_root)
    term = rloo.Term(cfg["rloo"], model, None)            # frozen_ref 时此刻拷下的是初始模型
    if step:
        sd = torch.load(MAIN / exp / "ckpt" / run / f"ckpt_{step}.pt", map_location="cpu")
        model.model.load_state_dict(sd["model"], strict=True)
        lo = sd.get("dinov2_lora", {})
        if set(lo) != set(model.lora):
            raise RuntimeError(f"{method}: 存档的 LoRA 层 {len(lo)} 个与模型 {len(model.lora)} 个不一致")
        with torch.no_grad():
            for k, l in model.lora.items():
                if abs(lo[k]["scale"] - l.scale) > 1e-9:
                    raise RuntimeError(f"{method}: LoRA {k} 的 scale 不一致")
                l.A.copy_(lo[k]["A"])
                l.B.copy_(lo[k]["B"])
    return model, cfg, term


def val_set(model):
    ds = PairSet(os.environ["MOON_DATA"], "val", model.resize, **model.input, seed=0)
    ds.pairs = [q for q in ds.pairs if ds.data.path(ds.src[q], q, "Label").exists()]
    return ds


def labels(data, split, pair):
    """(光学 N×2, SAR N×2)，同 workbench/dataset.py（Label 里 y 存为负值）。"""
    a = np.loadtxt(data.path(split, pair, "Label"), dtype=np.float64, ndmin=2)
    a[:, [1, 3]] *= -1
    return a[:, :2], a[:, 2:4]


def lstsq_affine(p, q):
    return np.linalg.lstsq(np.c_[p, np.ones(len(p))], q, rcond=None)[0].T


def show_pairs(pairs):
    """Val 每个 ROI 的第一对（按 pair 名排序）。"""
    first = {}
    for q in sorted(pairs):
        first.setdefault(q.split("/")[0], q)
    return [first[k] for k in sorted(first)]


# ---------- 方法 1：形态 ----------

def to_px(xn):
    """归一化坐标 → 原网格角点约定 px。"""
    return (xn + 1) * (HW / 2)


def spread(xy):
    """(n,2) px → 两轴标准差的均方根。"""
    return float(np.sqrt(xy.var(0).sum() / 2)) if len(xy) else math.nan


def run_morph(method, a):
    from workbench.records import InterWriter
    model, cfg, _ = load(method, a.weights_root)
    ds = val_set(model)
    pairs = ds.pairs[: a.limit or None]
    show = set(show_pairs(ds.pairs))
    cols = {k: [] for k in ("H_med", "H_mean", "pmode_med", "n_modes", "top_mode_frac", "c_spread", "c_cx", "c_cy",
                            "f_spread", "f_cx", "f_cy", "f_cover", "cert_mean", "cert_hi", "ok", "sv1", "sv2", "det",
                            "ctr_x", "ctr_y", "err_self", "d_self", "n_inl", "gt_ctr_x", "gt_ctr_y", "gt_det")}
    inter = {k: {} for k in ("certainty", "entropy", "warp", "coarse_mode")}
    t0 = time.time()
    with torch.no_grad():
        for i, pair in enumerate(pairs):
            it = ds[ds.pairs.index(pair)]
            i0, i1 = it["image0"][None].cuda(), it["image1"][None].cuda()
            f1, f2 = rloo.features(model, i0, i1)
            cls, cert = rloo.coarse(model, f1, f2)
            cls = cls.float()
            P = cls.softmax(1)
            Hc = -(P * cls.log_softmax(1)).sum(1)[0]                       # (h,w)
            pm, mode = P.max(1)
            fc = rloo.anchor_flow(P, mode)                                  # (1,2,h,w)
            flow, ce = rloo.refine(model, f1, f2, fc, cert.float())
            (A, n_inl), = rloo.fit_many(flow, ce, 5000, 3.0)
            h, w = Hc.shape
            cxy = to_px(fc[0].permute(1, 2, 0).reshape(-1, 2).cpu().numpy())
            fxy = to_px(flow[0].permute(1, 2, 0).reshape(-1, 2).float().cpu().numpy())
            pc = ce[0, 0].float().sigmoid().cpu().numpy()
            modes, cnt = np.unique(mode.cpu().numpy(), return_counts=True)
            inside = (fxy >= 0).all(1) & (fxy < HW).all(1)
            cov = np.zeros((HW // CELL, HW // CELL), bool)
            cov[(fxy[inside, 1] // CELL).astype(int), (fxy[inside, 0] // CELL).astype(int)] = True
            po, qo = labels(ds.data, ds.src[pair], pair)
            G = lstsq_affine(po, qo)
            row = {"H_med": float(Hc.median()), "H_mean": float(Hc.mean()), "pmode_med": float(pm.median()),
                   "n_modes": len(modes), "top_mode_frac": float(cnt.max() / cnt.sum()), "c_spread": spread(cxy),
                   "c_cx": float(cxy[:, 0].mean()), "c_cy": float(cxy[:, 1].mean()), "f_spread": spread(fxy),
                   "f_cx": float(fxy[:, 0].mean()), "f_cy": float(fxy[:, 1].mean()), "f_cover": float(cov.mean()),
                   "cert_mean": float(pc.mean()), "cert_hi": float((pc > THRESH).mean()), "ok": float(A is not None),
                   "n_inl": n_inl, "gt_ctr_x": float(G[0] @ [256, 256, 1]), "gt_ctr_y": float(G[1] @ [256, 256, 1]),
                   "gt_det": float(np.linalg.det(G[:, :2]))}
            if A is not None:
                sv = np.linalg.svd(A[:, :2], compute_uv=False)
                _, d = rloo.ref_cells(fc, A, 8.0, 16.0, -0.1, "gauss")
                row.update(sv1=float(sv[0]), sv2=float(sv[1]), det=float(np.linalg.det(A[:, :2])),
                           ctr_x=float(A[0] @ [256, 256, 1]), ctr_y=float(A[1] @ [256, 256, 1]),
                           err_self=protocol.pair_error(A, po, qo), d_self=float(d.mean()))
            for k in cols:
                cols[k].append(row.get(k, math.nan))
            if pair in show:
                H1, W1 = flow.shape[-2:]
                src = to_px(grid(H1, W1, "cpu").numpy())                      # (H,W,2) 光学像素中心
                inter["certainty"][pair] = pc
                inter["entropy"][pair] = Hc.cpu().numpy()
                inter["warp"][pair] = fxy.reshape(H1, W1, 2) - src
                inter["coarse_mode"][pair] = np.c_[cxy - 0.5, pm[0].reshape(-1).cpu().numpy()]
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    out = RUN / "raw" / (f"_smoke_morph_{method}" if a.limit else f"morph_{method}")
    np.savez_compressed(out.with_suffix(".npz"), **{k: np.asarray(v, np.float64) for k, v in cols.items()})
    out.with_suffix(".json").write_text(json.dumps({"method": method, "src": METHODS[method], "pairs": pairs,
                                                    "sec": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    if a.limit:
        return
    specs = {"certainty": ("scalar", "opt", "可信度（sigmoid），训练时的正常推理（560 一遍）尺度 1", ""),
             "entropy": ("scalar", "opt", "粗级每格的熵（温度 1），40×40 格", "nat"),
             "warp": ("flow", "opt", "尺度 1 的 warp（560 一遍，未经 864 上采样那一遍），光学像素指向 SAR 的位移", "px"),
             "coarse_mode": ("points", "sar", "粗级每格最高分位置在 SAR 上的落点（经邻居加权），值为它的概率", "")}
    for name, (kind, frame, desc, unit) in specs.items():
        with InterWriter(RUN, method, name, "val", kind=kind, frame=frame, desc=desc, unit=unit) as wr:
            for pair, v in inter[name].items():
                wr.write(pair, v)


# ---------- 方法 2：坍缩后是否还在学习 ----------

class _Step:
    def __init__(self, i0, i1, optim):
        self._im, self.optim, self.B = (i0, i1), optim, 1

    def images(self):
        return self._im


def _grad_norms(model, factor):
    parts, lora = model.update_parts()
    out = {}
    for k, ps in parts.items():
        g = [p.grad.float() for p in ps if p.grad is not None]
        out[f"g_{k}"] = float(torch.sqrt(sum((x / factor).square().sum() for x in g))) if g else 0.0
        out[f"w_{k}"] = float(torch.sqrt(sum(p.detach().float().square().sum() for p in ps)))
    for nm, idx in (("loraA", 0), ("loraB", 1)):
        g = [t[idx].grad.float() for t in lora if t[idx].grad is not None]
        out[f"g_{nm}"] = float(torch.sqrt(sum((x / factor).square().sum() for x in g))) if g else 0.0
        out[f"w_{nm}"] = float(torch.sqrt(sum(t[idx].detach().float().square().sum() for t in lora)))
    g = [p.grad.float() for p in model.params if p.grad is not None]
    out["g_total"] = float(torch.sqrt(sum((x / factor).square().sum() for x in g))) if g else 0.0
    return out


def run_learn(method, a):
    model, cfg, term = load(method, a.weights_root)
    ds = val_set(model)
    n = a.limit or LEARN_N
    idx = np.linspace(0, len(ds.pairs) - 1, n).round().astype(int)
    rec_adv = []
    orig_adv = rloo.advantage

    def advantage(r, baseline):                        # 记下每次调用的优势：第一次是粗级，之后是各加噪级
        adv = orig_adv(r, baseline)
        same = (r.std(0) == 0).float().mean() if r.shape[0] > 1 else torch.tensor(math.nan)
        rec_adv.append((float(adv.abs().mean()), float(same)))
        return adv

    rloo.advantage = advantage
    rows, t0 = [], time.time()
    for j, i in enumerate(idx):
        it = ds[int(i)]
        i0, i1 = it["image0"][None].cuda(), it["image1"][None].cuda()
        for scale in (2.0 ** 16, 2.0 ** 8, 1.0):
            optim = Optim(model.params, cfg["optim"], model.amp, model.groups)
            optim.scaler = torch.cuda.amp.GradScaler(init_scale=scale)
            optim.opt.zero_grad(set_to_none=True)
            torch.manual_seed(1000 + j)
            rec_adv.clear()
            loss, stats, _ = term(_Step(i0, i1, optim))
            optim.backward(loss)
            g = _grad_norms(model, optim.factor())
            optim.opt.zero_grad(set_to_none=True)
            if math.isfinite(g["g_total"]):
                break
        row = {"pair": it["pair"], "loss": float(loss), "gscale": scale, **stats, **g}
        names = ["c"] + [str(s) for s in cfg["rloo"]["levels"]]
        for nm, (am, same) in zip(names, rec_adv):
            row[f"adv_{nm}"], row[f"same_{nm}"] = am, same
        rows.append(row)
        if (j + 1) % 20 == 0:
            print(f"{j + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    rloo.advantage = orig_adv
    keys = sorted({k for r in rows for k in r if k != "pair"})
    out = RUN / "raw" / (f"_smoke_learn_{method}" if a.limit else f"learn_{method}")
    np.savez_compressed(out.with_suffix(".npz"),
                        **{k: np.asarray([r.get(k, math.nan) for r in rows], np.float64) for k in keys})
    out.with_suffix(".json").write_text(json.dumps({"method": method, "src": METHODS[method],
                                                    "pairs": [r["pair"] for r in rows], "rloo": cfg["rloo"],
                                                    "sec": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 对（冒烟测试），产物写到 raw/_smoke_<任务>，不写中间结果")
    a = ap.parse_args(argv)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    (RUN / "raw").mkdir(exist_ok=True)
    kind, method = a.task.split("_", 1)
    (run_morph if kind == "morph" else run_learn)(method, a)


if __name__ == "__main__":
    main()
