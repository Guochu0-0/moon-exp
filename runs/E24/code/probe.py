"""开跑前的小探针（#201），zero-shot AnyMatch-RoMa、训练设置（configs/_rl.toml）。三项：

1. path：一对训练图，只对细化损失（不含粗匹配项）反向，列出各组参数的梯度范数。grad 与 local 各一次。
   预期：尺度 16、8、4 的细化层、尺度 8、4 的 proj、VGG、尺度 16 的 proj、LoRA 有梯度；尺度 2、1 的细化层与 proj、
   GP、embedding decoder 没有。
2. descent：同一对图（扰动只抽一次），按 g1_s0 的配置（warmup 关掉、学习率恒定为锚点 × 1）训练 --steps 步，
   记每步正常推理轨迹上各级的细化奖励 mean r(μ_s)（精确梯度方式的损失为它的相反数）与离参照的距离。
   descent_fine：同一对图，参照仿射固定为第 0 步的，只更新细化部分（细化层、proj、VGG；粗匹配、LoRA 不动），
   只有精确梯度的细化损失，学习率同上，排除粗匹配与参照一起变的影响。
3. cos：--n 对不同的训练图，各算一次 local 与 grad 两种细化损失对细化层参数的梯度的余弦相似度（rloo.grad_cos）。

    python runs/E24/code/probe.py --out runs/E24/extra/data/probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from finetune import config as C                # noqa: E402
from finetune.data import PairSet               # noqa: E402
from finetune.geom import HW                    # noqa: E402
from finetune.models import roma                # noqa: E402
from finetune.optim import Optim                # noqa: E402
from finetune.parts import aug, rloo            # noqa: E402
from finetune.train import Step                 # noqa: E402

R = REPO / "runs" / "E24"


def group_of(name):
    if name.startswith("decoder.conv_refiner."):
        return "refiner" + name.split(".")[2]
    if name.startswith("decoder.proj."):
        return "proj" + name.split(".")[2]
    if name.startswith("decoder."):
        return "coarse_other"          # GP、embedding decoder
    if name.startswith("encoder.cnn"):
        return "vgg"
    return "other"


def prepare(model, it, p):
    """一对图：特征（带梯度）、正常推理轨迹与参照仿射。"""
    i0, i1 = it["image0"][None].cuda(), it["image1"][None].cuda()
    f1, f2 = rloo.features(model, i0, i1)
    with torch.no_grad():
        cls, cert = rloo.coarse(model, f1, f2)
        P = cls.float().softmax(1)
        traj = {}
        flow, ce = rloo.refine(model, f1, f2, rloo.anchor_flow(P, P.argmax(1)), cert.float(), traj)
        A = rloo.fit(flow[0], ce[0], p["num"], p["ransac"])[0]
    return f1, f2, traj, A


def path_check(model, ds, p, sig):
    out = {}
    named = list(model.model.named_parameters()) + [(f"lora.{k}.{n}", t) for k, l in model.lora.items()
                                                     for n, t in (("A", l.A), ("B", l.B))]
    for mode in ("grad", "local"):
        for _, t in named:
            t.grad = None
        f1, f2, traj, A = prepare(model, ds[0], p)
        sur, _ = rloo.fine_direct(model.model.decoder, f1, f2, traj, A, sig, {**p, "fine_mode": mode},
                                  lambda l: (l * 1024).backward(), 1024.0)
        (sur * 1024).backward()
        g = {}
        for n, t in named:
            if t.requires_grad:
                k = "lora" if n.startswith("lora.") else group_of(n)
                g.setdefault(k, 0.0)
                if t.grad is not None:
                    g[k] += float(t.grad.float().square().sum()) / 1024 ** 2
        out[mode] = {k: float(np.sqrt(v)) for k, v in sorted(g.items())}
        print(mode, out[mode], flush=True)
    for _, t in named:
        t.grad = None
    return out


def descent(cfg, ds, steps):
    model = roma.Model(cfg, os.environ.get("MOON_WEIGHTS", ""))
    o = {**cfg["optim"], "warmup": 0, "sched": "const"}
    optim = Optim(model.params, o, model.amp, model.groups)
    term = rloo.Term(cfg["rloo"], model, ds)
    batch = torch.utils.data.default_collate([ds[0]])
    rng = np.random.default_rng(0)
    rows = []
    for t in range(1, steps + 1):
        st = Step(model, "roma", batch, "cuda", False, rng, optim)
        loss, stats, _ = term(st)
        optim.backward(loss)
        optim.step(t)
        rows.append({"step": t, **{k: v for k, v in stats.items()
                                   if k.startswith(("rew_", "d_", "dmu_")) or k in ("rew", "d_ref", "l_fine")}})
        print(rows[-1], flush=True)
    return rows


def descent_fine(cfg, ds, steps):
    model = roma.Model(cfg, os.environ.get("MOON_WEIGHTS", ""))
    p = {**cfg["rloo"], "fine_mode": "grad"}
    sig = {s: p["k"] * s / (HW / 2) for s in p["levels"]}
    groups = [g for g in model.groups if g["lr_scale"] in (cfg["model"]["refine_lr"], cfg["model"]["vgg_lr"])]
    o = {**cfg["optim"], "warmup": 0, "sched": "const"}
    optim = Optim([q for g in groups for q in g["params"]], o, model.amp, groups)
    it = ds[0]
    _, _, _, A = prepare(model, it, p)
    rows = []
    for t in range(1, steps + 1):
        f1, f2, traj, _ = prepare(model, it, p)
        sur, rd = rloo.fine_direct(model.model.decoder, f1, f2, traj, A, sig, p, optim.backward, optim.factor())
        optim.backward(sur)
        optim.step(t)
        for q in model.params:
            q.grad = None
        rows.append({"step": t, **{f"rew_{s}": float(r.mean()) for s, (r, _) in rd.items()},
                     **{f"d_{s}": float(d.mean()) for s, (_, d) in rd.items()}})
        print("fine", rows[-1], flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    cfg = C.load(R / "configs" / "g1_s0.toml")
    p = cfg["rloo"]
    sig = {s: p["k"] * s / (HW / 2) for s in p["levels"]}
    model = roma.Model(cfg, os.environ.get("MOON_WEIGHTS", ""))
    ds = PairSet(a.data, "train", model.resize, **model.input, seed=0, **aug.data_kw(cfg["aug"]))
    ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // a.n)][: a.n]
    res = {"path": path_check(model, ds, p, sig)}

    cos = []
    for i in range(len(ds)):
        f1, f2, traj, A = prepare(model, ds[i], p)
        if A is None:
            continue
        c = rloo.grad_cos(model.model.decoder, f1, f2, traj, A, sig, {**p, "fine_mode": "local", "fine_group": 8})
        cos.append({"pair": ds.pairs[i], **c})
        print(cos[-1], flush=True)
    res["cos"] = cos
    res["cos_summary"] = {k: {"median": float(np.median([c[k] for c in cos])),
                              "frac_pos": float(np.mean([c[k] > 0 for c in cos]))}
                          for k in cos[0] if k != "pair"}
    del model
    torch.cuda.empty_cache()
    res["descent"] = descent(cfg, ds, a.steps)
    torch.cuda.empty_cache()
    res["descent_fine"] = descent_fine(cfg, ds, a.steps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("cos_summary", res["cos_summary"], flush=True)


if __name__ == "__main__":
    main()
