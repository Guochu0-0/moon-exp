"""E33 只用（#226）：提速前后的梯度对不对得上。

固定一批训练图（训练集里均匀取 N 对，扰动由数据集按索引与种子给出，两次运行完全相同），每对图在同样的随机数种子下
按训练时的做法算一遍损失并反向，得到这一对图上全部可训练参数的梯度（除以 GradScaler 的 scale，即真实梯度）。
两份配置各跑一遍：先用 --save 存下旧做法的梯度，再用 --against 指向它跑新做法，逐对算两份梯度的余弦相似度。
最后把 N 对的梯度加起来做一次参数更新、EMA 参照走一步，比较两边的参数改变量（余弦相似度、相对差）。

    python runs/E33/code/grad_check.py <配置> --out <目录> --save            # 旧做法：存 <目录>/grads.pt
    python runs/E33/code/grad_check.py <配置> --out <目录> --against <旧目录>  # 新做法：写 <目录>/grad_check.json

sdpa 开关是类级替换（finetune/models/roma.py use_sdpa），所以两份配置必须在两个进程里跑。
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

from baselines.match import seed_all  # noqa: E402
from finetune import config as C  # noqa: E402
from finetune.data import PairSet  # noqa: E402
from finetune.models import MODELS  # noqa: E402
from finetune.optim import Optim  # noqa: E402
from finetune.parts import PARTS  # noqa: E402
from finetune.train import Step  # noqa: E402


def flat(params, scale):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).float().reshape(-1) / scale for p in params])


def cos(a, b):
    return float((a.double() @ b.double()) / (a.double().norm() * b.double().norm()).clamp(min=1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--against")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = C.load(a.config)
    o, r, mname = cfg["optim"], cfg["run"], cfg["model"]["name"]
    seed_all(r["seed"])
    mod = MODELS[mname]
    os.environ.setdefault("TORCH_HOME", mod.TORCH_HOME)
    model = mod.Model(cfg, os.environ["MOON_WEIGHTS"], device="cuda")
    names = C.parts_of(cfg)
    kw = {}
    for n in names:
        kw.update(PARTS[n].data_kw(cfg[n]))
    ds = PairSet(os.environ["MOON_DATA"], r["split"], model.resize, **model.input, seed=r["seed"],
                 resize_hi=getattr(model, "resize_hi", None), **kw)
    terms = [(n, PARTS[n].Term(cfg[n], model, ds)) for n in names if PARTS[n].Term is not None]
    optim = Optim(model.params, {**o, "warmup": 0}, model.amp, getattr(model, "groups", None))
    idx = np.linspace(0, len(ds) - 1, a.n).round().astype(int)
    torch.manual_seed(12345)        # 主进程取数据时扰动的随机数（data.py 用 torch.initial_seed()）
    grads, total, stats = [], None, []
    for j, i in enumerate(idx):
        batch = torch.utils.data.default_collate([ds[int(i)]])
        while True:     # 半精度反向溢出时（GradScaler 起步的 scale 偏大），同训练一样把 scale 减半，同一种子重算
            seed_all(1000 + j)
            st = Step(model, mname, batch, "cuda", False, np.random.default_rng(j), optim)
            loss = torch.zeros((), device="cuda")
            for _, term in terms:
                l_, s_, _ = term(st)
                loss = loss + l_
            optim.backward(loss)
            g = flat(model.params, optim.factor())
            if bool(torch.isfinite(g).all()):
                break
            optim.opt.zero_grad(set_to_none=True)
            optim.scaler.update(optim.scaler.get_scale() / 2)
        grads.append(g.cpu())
        total = g.clone() if total is None else total + g
        stats.append({"pair": batch["pair"][0], "loss": float(loss), "d_ref": s_.get("d_ref"),
                      "in_frame": s_.get("in_frame"), "gnorm": float(g.norm())})
        optim.opt.zero_grad(set_to_none=True)
    # 合起来做一次更新：梯度换成 N 对之和（乘回当前 scale，交给 optim.step 照常 unscale），EMA 参照走一步
    sc = optim.factor()
    k = 0
    for p in model.params:
        m = p.numel()
        p.grad = (total[k:k + m] * sc).view_as(p).to(p.dtype)
        k += m
    th0 = torch.cat([p.detach().float().reshape(-1) for p in model.params]).cpu()
    slow = [t.slow for _, t in terms if getattr(t, "slow", None) is not None]
    sh0 = torch.cat([e.detach().float().reshape(-1) for e in slow[0].shadow]).cpu() if slow else None
    upd = optim.step(o["accum"])
    for _, t in terms:
        if hasattr(t, "after_update"):
            t.after_update(None)
    d_th = torch.cat([p.detach().float().reshape(-1) for p in model.params]).cpu() - th0
    d_sh = torch.cat([e.detach().float().reshape(-1) for e in slow[0].shadow]).cpu() - sh0 if slow else None
    res = {"config": a.config, "pairs": stats, "upd": upd}
    if a.save:
        torch.save({"grads": grads, "d_th": d_th, "d_sh": d_sh, "stats": stats}, out / "grads.pt")
    if a.against:
        ref = torch.load(Path(a.against) / "grads.pt")
        res["cos"] = [round(cos(x, y), 5) for x, y in zip(ref["grads"], grads)]
        res["rel_norm"] = [round(float(y.norm() / x.norm()), 5) for x, y in zip(ref["grads"], grads)]
        res["cos_sum"] = round(cos(sum(ref["grads"]), sum(grads)), 5)
        res["cos_update"] = round(cos(ref["d_th"], d_th), 5)
        res["rel_update"] = round(float((ref["d_th"] - d_th).norm() / ref["d_th"].norm()), 5)
        if d_sh is not None:
            res["cos_ema"] = round(cos(ref["d_sh"], d_sh), 5)
            res["rel_ema"] = round(float((ref["d_sh"] - d_sh).norm() / ref["d_sh"].norm()), 5)
        res["ref_pairs"] = ref["stats"]
    (out / "grad_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairs", "ref_pairs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
