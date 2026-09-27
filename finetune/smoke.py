"""最小训练循环：从底座 ckpt 出发，在 patch 对上跑前向 + 反向 + 优化器，顺带实测成本。

    python -m finetune.smoke configs/baselines/anymatch_loftr.json --split train --steps 30 --batch 1 \
        --data $MOON_DATA --weights-root $MOON_WEIGHTS --out $MOON_RESULTS/finetune/smoke [--lr 0] [--save]

loss 是**占位**的，只为证明梯度能同时流到粗、细两级，不是任何训练方案：
    粗级：推理同款粗匹配 (b,i,j) 上 −log conf_matrix 的均值（dual-softmax 概率）；
    细级：expec_f 的 std 均值（5×5 soft-argmax 热图的展宽）。

产物 <out>/stats.json：显存峰值、训练步吞吐、纯推理耗时、仿射 RANSAC 耗时、conf_matrix 尺寸、各模块梯度范数。
--save 另存 <out>/ckpt.pt（baselines 适配器可直接读）。--lr 0 时权重不变，用来验证「接入没改变行为」：
把 ckpt.pt 交给 `baselines.match --weights`，输出应与 zero-shot 的原始点对一致。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from baselines.match import env_info, git_head, seed_all
from baselines.ransac import fit_affine

from .data import PairSet
from .model import Base

REPO = Path(__file__).resolve().parents[1]
WARMUP = 3            # 计时时跳过的前几步（cudnn 选算法、分配器预热）
MODULES = ("backbone", "loftr_coarse", "fine_preprocess", "loftr_fine")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def placeholder_loss(data):
    conf = data["conf_matrix"]
    b, i, j = data["b_ids"], data["i_ids"], data["j_ids"]
    if len(b) == 0:  # 没有粗匹配时仍要让图连着，loss 为 0
        return conf.sum() * 0, {"coarse": 0.0, "fine": 0.0}
    l_c = -torch.log(conf[b, i, j].clamp_min(1e-6)).mean()
    l_f = data["expec_f"][:, 2].mean()
    return l_c + l_f, {"coarse": l_c.item(), "fine": l_f.item()}


def grad_norms(model):
    out = {}
    for name in MODULES:
        g = [p.grad.detach().norm() ** 2 for p in getattr(model, name).parameters() if p.grad is not None]
        out[name] = float(torch.stack(g).sum().sqrt()) if g else 0.0
    return out


def matches_per_pair(base, data, bs, hw):
    """batch 的粗细匹配 → 每对一个 N×5（原网格，中心约定），与 baselines.match 存的格式相同。"""
    b = data["m_bids"].cpu().numpy()
    kp0 = base.to_original(data["mkpts0_f"].detach().cpu().numpy(), *hw)
    kp1 = base.to_original(data["mkpts1_f"].detach().cpu().numpy(), *hw)
    conf = data["mconf"].detach().cpu().numpy()
    return [np.c_[kp0[b == k], kp1[b == k], conf[b == k]].astype(np.float32) for k in range(bs)]


def stats(xs):
    xs = np.asarray(xs, float)
    return {"mean": round(float(xs.mean()), 4), "median": round(float(np.median(xs)), 4),
            "max": round(float(xs.max()), 4), "n": len(xs)} if len(xs) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights_root) / cfg["weights"]
    seed_all(int(cfg.get("seed", 0)))

    base = Base(REPO / cfg["repo"], weights, device=args.device, **cfg.get("params", {}))
    ds = PairSet(args.data, args.split, base.resize, limit=args.steps * args.batch, **cfg["input"])
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                     drop_last=True)
    opt = torch.optim.AdamW([p for p in base.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    hw = (512, 512)  # patch 原尺寸；resize 前的网格
    rec = {"config": cfg, "args": vars(args), "n_pairs_in_split": len(PairSet(args.data, args.split, base.resize)),
           "commit": git_head(REPO), "notes": base.adapter.notes, "steps": []}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fwd, bwd, step_t, ransac_t = [], [], [], []
    for it, batch in enumerate(dl):
        i0, i1 = batch["image0"].to(args.device), batch["image1"].to(args.device)
        sync(); t0 = time.time()
        data = base.forward(i0, i1)
        loss, parts = placeholder_loss(data)
        sync(); t1 = time.time()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync(); t2 = time.time()
        g = grad_norms(base.model)
        opt.step()
        sync(); t3 = time.time()

        per_pair = matches_per_pair(base, data, i0.shape[0], hw)
        n_inl = []
        for M in per_pair:
            tr = time.time()
            _, inl, _ = fit_affine(M, args.ransac)
            ransac_t.append(time.time() - tr)
            n_inl.append(int(inl.sum()) if inl is not None else 0)
        if it == 0:
            c = data["conf_matrix"]
            rec["conf_matrix"] = {"shape": list(c.shape), "dtype": str(c.dtype),
                                  "mb_per_pair": round(c[0].numel() * c.element_size() / 2 ** 20, 1),
                                  "hw_i": list(data["hw0_i"]), "hw_c": list(data["hw0_c"]), "hw_f": list(data["hw0_f"]),
                                  "fine_window": int(data["W"])}
        if it >= WARMUP:
            fwd.append(t1 - t0); bwd.append(t2 - t1); step_t.append(t3 - t0)
        rec["steps"].append({"pairs": batch["pair"], "loss": round(loss.item(), 5), **parts,
                             "n_coarse": int(len(data["b_ids"])), "n_inliers": n_inl, "grad_norm": g,
                             "sec_fwd": round(t1 - t0, 4), "sec_bwd": round(t2 - t1, 4)})
        print(f"  step {it}: loss={loss.item():.4f} M={len(data['b_ids'])} inl={n_inl} "
              f"fwd={t1 - t0:.3f}s bwd={t2 - t1:.3f}s", flush=True)

    rec["train_step"] = {"sec_fwd": stats(fwd), "sec_bwd": stats(bwd), "sec_step": stats(step_t),
                         "pairs_per_sec": round(args.batch / float(np.mean(step_t)), 3) if step_t else None}
    rec["ransac_sec"] = stats(ransac_t)
    if torch.cuda.is_available():
        rec["train_peak_mem_gb"] = {"allocated": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                                    "reserved": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}
        torch.cuda.reset_peak_memory_stats()

    # 纯推理（no_grad），与 baselines.match 的 matcher.match 同路径，逐对计时
    inf = []
    with torch.no_grad():
        for k in range(min(len(ds), 20)):
            item = ds[k]
            sync(); t0 = time.time()
            base.forward(item["image0"][None].to(args.device), item["image1"][None].to(args.device))
            sync(); inf.append(time.time() - t0)
    rec["infer_sec"] = stats(inf[WARMUP:])
    if torch.cuda.is_available():
        rec["infer_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)

    if args.save:
        torch.save(base.state_dict(), out / "ckpt.pt")
    rec["env"] = env_info()
    (out / "stats.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("conf_matrix", "train_step", "ransac_sec", "infer_sec")
                      if k in rec}, ensure_ascii=False, indent=1))
    print({k: rec.get(k) for k in ("train_peak_mem_gb", "infer_peak_mem_gb")})


if __name__ == "__main__":
    main()
