"""RoMa 系底座的无标注微调（「【RoMa】微调代码接入与实测」#64）。部件见 finetune/roma.py，循环照 finetune/train.py。

    python -m finetune.train_roma configs/baselines/anymatch_roma__minmax.json --out <dir> \
        --labels <labels.jsonl> --label-top 0.5 --aug geo,photo        # 伪标签（P8 对应物）
    python -m finetune.train_roma ... --w-pseudo 0 --w-ripe 1 --neg     # 类 RIPE（Q4 对应物）

产物同 finetune.train：log.jsonl、ckpt_<step>.pt（{'model': ...}，baselines.match --weights 直接读）、args.json。
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

from .augment import compose
from .data import PairSet
from .label import load_labels
from .roma import RomaBase, affine_norm, inv_affine, ripe_loss, roma_loss
from .train import lr_at

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--init", help="起点 ckpt（{'model': ...}），默认配置里的底座权重")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", help="离线伪仿射（finetune.label 格式）")
    ap.add_argument("--label-top", type=float, default=1.0)
    ap.add_argument("--aug", default="")
    ap.add_argument("--aug-shift", type=float, default=12.0)
    ap.add_argument("--aug-rot", type=float, default=3.0)
    ap.add_argument("--aug-scale", type=float, default=0.03)
    ap.add_argument("--swap", type=float, default=0.5, help="伪标签：以此概率交换光学 / SAR 方向（标签取逆），训 B→A")
    ap.add_argument("--w-pseudo", type=float, default=1.0)
    ap.add_argument("--w-ripe", type=float, default=0.0, help="类 RIPE 闭式期望（finetune/roma.py ripe_loss）")
    ap.add_argument("--r-out", type=float, default=-0.25, help="类 RIPE：正样本对外点的分值（Q4 为 −0.25）")
    ap.add_argument("--w-cert", type=float, default=1.0, help="类 RIPE：certainty 取舍项相对锚点项的权重")
    ap.add_argument("--cert-out", type=float, default=None, help="类 RIPE：取舍项里外点的分值，默认同 --r-out")
    ap.add_argument("--num", type=int, default=5000, help="类 RIPE：每步抽点数")
    ap.add_argument("--neg", action="store_true", help="类 RIPE：负样本对并入同一次前向")
    ap.add_argument("--placebo", action="store_true", help="类 RIPE：reward 在像素间随机打乱（对照）")
    ap.add_argument("--train-vgg", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--warmup-start", type=float, default=0.1)
    ap.add_argument("--sched", default="cosine", choices=("const", "cosine"))
    ap.add_argument("--lr-min", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.0, help="梯度范数裁剪；0 = 不裁（romatch 原配置 0.01）")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--save0", action="store_true", help="存 ckpt_0（lr=0 验收用）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    weights = args.init or str(Path(args.weights_root) / cfg["weights"])
    prm = dict(cfg.get("params", {}))
    if args.init:
        prm["weights_key"] = "model"
    base = RomaBase(REPO / cfg["repo"], weights, device=args.device, train_vgg=args.train_vgg, **prm)
    if args.w_pseudo > 0 and not args.labels:
        ap.error("RoMa 伪标签只做离线（--labels）")
    aug = tuple(x for x in args.aug.split(",") if x)
    ds = PairSet(args.data, "train", base.resize, **cfg["input"], neg=args.neg, seed=args.seed, aug=aug,
                 aug_shift=args.aug_shift, aug_rot=args.aug_rot, aug_scale=args.aug_scale)
    labels = None
    if args.labels:
        labels = load_labels(args.labels, args.label_top)
        ds.pairs = [p for p in ds.pairs if labels.get(p) is not None]
    if args.limit:
        ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // args.limit)][: args.limit]
    g = torch.Generator().manual_seed(args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, generator=g,
                                     num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    params = [p for p in base.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler()   # decoder 内部按 romatch 的 amp_dtype（fp16）autocast
    (out / "args.json").write_text(json.dumps({"args": vars(args), "config": cfg, "weights": weights,
                                               "n_train": len(ds), "n_params": sum(p.numel() for p in params),
                                               "commit": git_head(REPO), "env": env_info()},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    if args.save0:
        torch.save(base.state_dict(), out / "ckpt_0.pt")

    rng = np.random.default_rng(args.seed)
    step, t_start = 0, time.time()
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        while step < args.steps:
            for batch in dl:
                t0 = time.time()
                i0, i1 = batch["image0"].to(args.device), batch["image1"].to(args.device)
                B = len(batch["pair"])
                loss, st = torch.zeros((), device=args.device), {}
                if args.w_pseudo > 0:
                    As = [np.asarray(labels[p]) for p in batch["pair"]]
                    if "T" in batch:
                        As = [compose(T.numpy(), A) for T, A in zip(batch["T"], As)]
                    sw = rng.random(B) < args.swap
                    a, b_ = torch.where(torch.from_numpy(sw)[:, None, None, None].to(i0.device), i1, i0), \
                        torch.where(torch.from_numpy(sw)[:, None, None, None].to(i0.device), i0, i1)
                    An = torch.from_numpy(np.stack([affine_norm(inv_affine(A) if s else A) for A, s in zip(As, sw)])
                                          ).float().to(args.device)
                    corr = base.forward(a, b_)
                    l_ps, st = roma_loss(corr, An)
                    loss = args.w_pseudo * l_ps
                if args.w_ripe > 0:
                    x0, x1 = i0, i1
                    if args.neg:
                        x0, x1 = torch.cat([i0, i0]), torch.cat([i1, batch["image1_neg"].to(args.device)])
                    corr = base.forward(x0, x1)
                    l_r, st_r = ripe_loss(corr, [False] * B + [True] * B if args.neg else None, r_out=args.r_out,
                                          num=args.num, w_cert=args.w_cert, placebo=args.placebo, rng=rng, cert_out=args.cert_out)
                    loss = loss + args.w_ripe * l_r
                    st.update(st_r)
                scaler.scale(loss / args.accum).backward()
                step += 1
                upd = {}
                if step % args.accum == 0:
                    lr = lr_at(step, args.steps, args.lr, args.warmup, args.warmup_start, args.sched, args.lr_min)
                    for gr in opt.param_groups:
                        gr["lr"] = lr
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(params, args.clip if args.clip > 0 else float("inf"))
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    upd = {"lr": float(f"{lr:.4g}"), "gnorm": round(float(gn), 4), "scale": scaler.get_scale()}
                rec = {"step": step, "pairs": batch["pair"], "loss": round(float(loss), 5), **st, **upd,
                       "sec": round(time.time() - t0, 3)}
                if step == 1 and torch.cuda.is_available():
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if step % 50 == 0:
                    log.flush()
                    print(f"step {step}: loss={float(loss):.4f} "
                          + " ".join(f"{k}={v}" for k, v in st.items())
                          + f" mem={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G"
                          + f" elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
                if args.save_every and step % args.save_every == 0:
                    torch.save(base.state_dict(), out / f"ckpt_{step}.pt")
                if step >= args.steps:
                    break
    if args.save_every and step % args.save_every:
        torch.save(base.state_dict(), out / f"ckpt_{step}.pt")


if __name__ == "__main__":
    main()
