"""E43（#250）：事后量遮挡版本与原图在同一批没遮住的格子上离逐像素参照的距离，与 E42 Notes 表 4 同类的比值。
不是检验实验，只补一个训练日志里没记的量（E43 训练日志的 occ_d_* 对全部格子取平均，含被遮的，也没记原图那一份）。

做法：用训练同一份配置（configs/main.toml）建模型、训练集（同样的扰动），固定种子 0 从训练集取 N 对（不放回），每对的扰动
在各存档间相同。对每个存档 S（模型 = ckpt_S，S = 0 为 zero-shot）：
- 参照：EMA 参照的权重没有存下（save_optim 关闭），用 ckpt_{S-250} 代替（τ = 0.996 的滑动平均，权重的平均滞后约
  τ/(1-τ) ≈ 249 次更新）；S = 0 与 250 时参照为 zero-shot（S = 0 即只量不学的情形：模型与参照相同）。
- 参照在原图上正常推理，记逐像素对应（粗级取最高分位置，细化各级的输出），并拟合参照仿射（5000 点、RANSAC 3 px）；
  拟合失败的对跳过。
- 遮挡照训练（finetune/parts/rloo_pixocc.py 的同一组函数）：光学图上 229 px 正方形，SAR 上按参照仿射映过去的四边形，
  填噪声。位置与噪声取自以 (种子, 这对的序号) 起的发生器，各存档相同。
- 模型分别在原图与遮挡版本上推理，量三个距离（原网格 px）：粗级抽签（温度 0.3，每格抽 16 次）离粗级逐像素参照的距离、
  粗级最高分位置离它的距离、正常推理尺度 1 的输出离尺度 1 逐像素参照的距离。只在打分的格子上量（训练里新项打分的
  同一种掩码：光学格子不在正方形内，参照点在 SAR 画面内、不在补零区、不在被遮的四边形内），原图与遮挡版本用同一批格子。
每对取格子上的平均，每个存档报各对的中位数与两者之比（遮挡版本 ÷ 原图）。

写 extra/data/occ_ratio.csv（每存档）与 occ_ratio_pairs.csv（每存档每对），并打印。在 gpfs worktree 里单卡跑：

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E43/code/occ_ratio.py > runs/E43/ckpt/occ_ratio.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
Y = "/remote-home/xufang/YGC"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if "GPU" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from baselines.match import seed_all  # noqa: E402
from finetune import config as C  # noqa: E402
from finetune.data import PairSet  # noqa: E402
from finetune.geom import HW  # noqa: E402
from finetune.models import MODELS  # noqa: E402
from finetune.models.roma import load_weights  # noqa: E402
from finetune.parts import PARTS  # noqa: E402
from finetune.parts import rloo as RL  # noqa: E402
from finetune.parts import rloo_pixocc as PO  # noqa: E402
from workbench.launch import require_clean  # noqa: E402

N, SEED = 64, 0
CKPTS = (0, 250, 500, 1000, 2000, 4000, 6000, 7750)
CK = R / "ckpt" / "main"


@torch.no_grad()
def infer(model, i0, i1):
    """正常推理：(粗级 logits (1,C,h,w), 粗级最高分位置的 warp, 细化各级输出 {尺度: μ}, 尺度 1 的 (flow, cert))。"""
    f1, f2 = RL.features(model, i0[None], i1[None])
    cls, cert = RL.coarse(model, f1, f2)
    cls = cls.float()
    P = cls.softmax(1)
    fc = RL.anchor_flow(P, P.argmax(1))
    traj = {}
    flow, ce = RL.refine(model, f1, f2, fc, cert.float(), traj)
    return cls, fc, {s: mu for s, (_, _, mu) in traj.items()}, (flow, ce)


@torch.no_grad()
def dists(model, i0, i1, tc, t1, kc, k1, temp, group, gen):
    """模型在 (i0, i1) 上：粗级抽签、粗级最高分位置、尺度 1 输出离逐像素参照的平均距离（px，只在 kc / k1 格子上）。"""
    cls, fc, mus, _ = infer(model, i0, i1)
    C_, h, w = cls.shape[1:]
    q = (cls[0] / temp).softmax(0)
    k = torch.multinomial(q.reshape(C_, -1).T, group, replacement=True, generator=gen).T.reshape(-1, h, w)
    fs = RL.anchor_flow(cls.softmax(1).expand(group, -1, -1, -1), k)
    px = HW / 2
    d_s = (fs - tc).norm(dim=1) * px            # (group,h,w)
    d_m = (fc - tc).norm(dim=1)[0] * px
    d_1 = (mus[1] - t1).norm(dim=1)[0] * px
    return (float(d_s[:, kc].mean()), float(d_m[kc].mean()), float(d_1[k1].mean()))


def build(cfg):
    mod = MODELS[cfg["model"]["name"]]
    os.environ.setdefault("TORCH_HOME", mod.TORCH_HOME)
    return mod.Model(cfg, f"{Y}/weights", device="cuda")


def load(model, z, s):
    """把存档 s 装进 model（s = 0：zero-shot 的权重 z）。"""
    load_weights(model, z if s == 0 else torch.load(CK / f"ckpt_{s}.pt", map_location="cpu"))
    model.model.eval()


def main():
    require_clean(REPO)
    cfg = C.load(R / "configs" / "main.toml")
    p = cfg["rloo"]
    seed_all(cfg["run"]["seed"])
    model, ref = build(cfg), build(cfg)
    z = model.state_dict()
    kw = {}
    for n in C.parts_of(cfg):
        kw.update(PARTS[n].data_kw(cfg[n]))
    ds = PairSet(f"{Y}/dataset/Moon", cfg["run"]["split"], model.resize, **model.input, seed=cfg["run"]["seed"],
                 resize_hi=getattr(model, "resize_hi", None), **kw)
    idx = np.random.default_rng(SEED).choice(len(ds), N, replace=False)
    pairs = [ds[int(i)] for i in idx]
    rows, summ = [], []
    t0 = time.time()
    for s in CKPTS:
        load(model, z, s)
        load(ref, z, max(0, s - 250))
        per = []
        for j, b in enumerate(pairs):
            i0, i1 = b["image0"].cuda(), b["image1"].cuda()
            _, tc, tl, (zf, zc) = infer(ref, i0, i1)
            A, _ = RL.fit_many(zf, zc, p["num"], p["ransac"])[0]
            if A is None:
                continue
            rng = np.random.default_rng([SEED, j])
            gen = torch.Generator(device="cuda").manual_seed(int(np.random.default_rng([SEED, j, 1]).integers(2 ** 62)))
            sq = PO.square(rng, p["pix_occ_side"])
            quad = PO.map_poly(sq, A)
            o0, _ = PO.occlude(i0, sq, gen)
            o1, _ = PO.occlude(i1, quad, gen)
            valid = RL.sar_valid(i1)
            kc = PO.keep_mask(tc, sq, quad, valid)
            k1 = PO.keep_mask(tl[1], sq, quad, valid)
            if not kc.any() or not k1.any():
                continue
            g0 = torch.Generator(device="cuda").manual_seed(SEED * 1000 + j)
            g1 = torch.Generator(device="cuda").manual_seed(SEED * 1000 + j)
            d0 = dists(model, i0, i1, tc, tl[1], kc, k1, p["temp"], p["group"], g0)
            d1 = dists(model, o0, o1, tc, tl[1], kc, k1, p["temp"], p["group"], g1)
            r = {"ckpt": s, "ref": max(0, s - 250), "pair": b["pair"], "keep_c": round(float(kc.float().mean()), 4),
                 "keep_1": round(float(k1.float().mean()), 4)}
            for name, a, c in zip(("sample", "mode", "s1"), d1, d0):
                r[f"{name}_occ"], r[f"{name}_orig"] = round(a, 3), round(c, 3)
            per.append(r)
        rows += per
        row = {"ckpt": s, "ref": max(0, s - 250), "n_pairs": len(per)}
        for name in ("sample", "mode", "s1"):
            a = float(np.median([r[f"{name}_occ"] for r in per]))
            c = float(np.median([r[f"{name}_orig"] for r in per]))
            row.update({f"{name}_occ": round(a, 2), f"{name}_orig": round(c, 2), f"{name}_ratio": round(a / c, 3),
                        f"{name}_ratio_pairs_med": round(float(np.median([r[f"{name}_occ"] / r[f"{name}_orig"]
                                                                          for r in per])), 3)})
        summ.append(row)
        print(f"{time.time() - t0:.0f}s", row, flush=True)
    out = R / "extra" / "data"
    out.mkdir(parents=True, exist_ok=True)
    for name, rs in (("occ_ratio.csv", summ), ("occ_ratio_pairs.csv", rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)


if __name__ == "__main__":
    main()
