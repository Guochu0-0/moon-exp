"""SCENES 式伪标签微调（「复现 SCENES 式伪标签微调 baseline」#26）。损失见 finetune/pseudo.py。

    python -m finetune.train configs/baselines/anymatch_loftr.json --out $MOON_RESULTS/finetune/scenes \
        [--steps 8000] [--lr 1e-5] [--min-inliers 30] [--save-every 1000] \
        [--warmup 500 --sched cosine --clip 0.5 --accum 8]      # 优化配方（「把 S1 调好」#50）
        [--w-pseudo 0 --w-cexp 1 [--neg] [--cexp-placebo]]      # 粗级闭式期望（#49，finetune/coarse.py）

每一步：前向（模块 eval、开梯度）→ 取伪仿射 → 粗、细两级损失 → 反向。
伪仿射两种来源：--labels 给离线文件（SCENES 做法，只在 keep 的对上训练）；不给则每步用本步匹配在线估计，
内点少于 --min-inliers 的对不监督（实测会塌缩，见 runs/S2）。

产物 <out>/：
  log.jsonl          每步一行：loss 分项、内点数、参与监督的对数、耗时
  ckpt_<step>.pt     baselines 适配器可直接读（baselines.match --weights）
  args.json          参数、commit、环境
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from baselines.match import env_info, git_head, seed_all

from .data import PairSet
from .augment import compose
from .label import load_labels
from .coarse import coarse_expect_loss
from .model import Base
from .pseudo import pseudo_loss
from .rl import FEATS, rl_loss

REPO = Path(__file__).resolve().parents[1]
HW = 512  # patch 原尺寸


def lr_at(step, total, lr, warmup=0, warmup_start=0.1, sched="const", lr_min=0.0):
    """第 step 个前向步（1 起）之后那次更新用的 lr。warmup 从 warmup_start·lr 线性升到 lr（同上游 LoFTR 的 linear
    warmup）；之后 const 恒定，cosine 在 [warmup, total] 上从 lr 降到 lr_min·lr。步数按前向步计，与 --accum 无关。"""
    if step < warmup:
        return lr * (warmup_start + (1 - warmup_start) * step / warmup)
    if sched == "const":
        return lr
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr * (lr_min + (1 - lr_min) * 0.5 * (1 + math.cos(math.pi * t)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--init", help="起点 ckpt，默认配置里的底座权重")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", help="finetune.label 的离线伪仿射（SCENES 做法）；不给则每步在线估计")
    ap.add_argument("--label-top", type=float, default=1.0, help="只用内点数最多的前这一比例的标签（课程式子集，#53）")
    ap.add_argument("--aug", default="", help="学生侧扰动，逗号分隔：geo（SAR 已知随机仿射，标签按 T∘A 变换）、photo（#53）")
    ap.add_argument("--aug-shift", type=float, default=12.0, help="geo 的最大平移，原网格 px")
    ap.add_argument("--aug-rot", type=float, default=3.0, help="geo 的最大旋转，度")
    ap.add_argument("--aug-scale", type=float, default=0.03, help="geo 的最大缩放幅度")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ransac", type=float, default=3.0, help="伪仿射的 RANSAC 阈值（原网格 px），与评测同口径")
    ap.add_argument("--min-inliers", type=int, default=20, help="在线模式：伪仿射内点少于此数的对不监督（同 SCENES 的筛选）")
    ap.add_argument("--coarse-set", default="all", choices=("all", "inliers"),
                    help="粗级正样本：all = 所有图内格（上游形式）；inliers = 只取本步内点所在的格")
    ap.add_argument("--w-coarse", type=float, default=1.0)
    ap.add_argument("--w-fine", type=float, default=1.0)
    ap.add_argument("--w-pseudo", type=float, default=1.0, help="伪标签监督项的权重；0 = 不用")
    ap.add_argument("--rl-pair", type=float, default=0.0, help="整对 CFOG reward 的 RL 项权重（finetune/rl.py）")
    ap.add_argument("--reward", default="gradncc", choices=("cfog", "gradncc"), help="整对 reward 的相似度")
    ap.add_argument("--rl-match", type=float, default=0.0, help="逐匹配残差 reward 的 RL 项权重")
    ap.add_argument("--K", type=int, default=4, help="每对采样组数（组内 baseline）")
    ap.add_argument("--sig-g", type=float, default=0.25, help="共享整体平移的采样标准差（归一化窗口坐标，1 = 窗口半宽）")
    ap.add_argument("--sig-i", type=float, default=0.1, help="逐匹配独立噪声的标准差（同上）")
    ap.add_argument("--w-cexp", type=float, default=0.0,
                    help="粗级闭式期望 −Σ P·r 的权重（r = 当前模型 RANSAC 内点 +1 / 外点 −1，finetune/coarse.py，#49）")
    ap.add_argument("--cexp-placebo", action="store_true", help="随机 reward 对照：r 在同一对的匹配之间随机打乱")
    ap.add_argument("--cexp-out", type=float, default=-1.0, help="粗级闭式期望里外点的分值（RIPE++ 为 −1，#51）")
    ap.add_argument("--neg", action="store_true",
                    help="每对再配一个负样本对（SAR 取自其他 ROI），并入同一次前向；负样本对上的内点给 −1（#49）")
    ap.add_argument("--w-l2sp", type=float, default=0.0, help="参数空间锚定（L2-SP）权重")
    ap.add_argument("--placebo", action="store_true", help="安慰剂对照：整对 reward 换成随机数")
    ap.add_argument("--inject-shift", type=int, default=0,
                    help="健全性测试：SAR 输入水平平移的像素数（输入网格），只配合 --w-pseudo 0 使用")
    ap.add_argument("--accum", type=int, default=1, help="梯度累积：每 accum 步（对）更新一次；步数、存 ckpt 仍按前向步计")
    ap.add_argument("--warmup", type=int, default=0, help="线性 warmup 的前向步数；0 = 不用")
    ap.add_argument("--warmup-start", type=float, default=0.1, help="warmup 起点 lr 占 --lr 的比例（上游 LoFTR 为 0.1）")
    ap.add_argument("--sched", default="const", choices=("const", "cosine"), help="warmup 之后的 lr 调度")
    ap.add_argument("--lr-min", type=float, default=0.0, help="cosine 终点 lr 占 --lr 的比例")
    ap.add_argument("--clip", type=float, default=0.0, help="梯度范数裁剪阈值（上游 LoFTR 为 0.5）；0 = 不裁")
    ap.add_argument("--train-modules", default="all", choices=("all", "fine"),
                    help="fine：只训细级模块，粗匹配保持起点不变")
    ap.add_argument("--rl-scope", default="all", choices=("all", "fine"),
                    help="fine：RL 项的梯度只进细级模块，其余项照常训全部模块（一次训练里隔离细级 RL 对粗匹配的破坏，#51）")
    ap.add_argument("--limit", type=int, default=0, help="只用 Train 前 N 对（过拟合测试：RL 能否在固定小集合上推高 reward）")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--save-at", default="", help="额外存 ckpt 的步数，逗号分隔（看早期动态，#49）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    weights = args.init or str(Path(args.weights_root) / cfg["weights"])
    base = Base(REPO / cfg["repo"], weights, device=args.device, **cfg.get("params", {}))
    s = HW / base.long_side  # 原网格 / 输入网格（正方形 patch）

    if args.neg and args.labels:
        ap.error("--neg 只用于在线信号（粗级闭式期望、在线细级项、RL 项），不配合离线伪标签")
    aug = tuple(x for x in args.aug.split(",") if x)
    if "geo" in aug and args.w_pseudo > 0 and not args.labels:
        ap.error("--aug geo 目前只配合离线伪标签（标签按 T∘A 变换）")
    ds = PairSet(args.data, "train", base.resize, **cfg["input"], neg=args.neg, seed=args.seed, aug=aug,
                 aug_shift=args.aug_shift, aug_rot=args.aug_rot, aug_scale=args.aug_scale)
    labels = None
    if args.labels:
        labels = load_labels(args.labels, args.label_top)
        ds.pairs = [p for p in ds.pairs if labels.get(p) is not None]
    if args.limit:
        ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // args.limit)][: args.limit]   # 均匀取样，覆盖各 ROI
    g = torch.Generator().manual_seed(args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, generator=g,
                                     num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    if args.train_modules != "all":   # fine = 只训细级（fine_preprocess + loftr_fine），backbone 与粗级冻结
        keep = ("fine_preprocess", "loftr_fine")
        for name, p in base.model.named_parameters():
            p.requires_grad_(name.startswith(keep))
    params = [p for p in base.model.parameters() if p.requires_grad]
    fine_params = [p for name, p in base.model.named_parameters()
                   if p.requires_grad and name.startswith(("fine_preprocess", "loftr_fine"))]
    params0 = [p.detach().clone() for p in params] if args.w_l2sp > 0 else None   # L2-SP 锚点 = 起点权重
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    (out / "args.json").write_text(json.dumps({"args": vars(args), "config": cfg, "weights": weights,
                                               "n_train": len(ds), "commit": git_head(REPO), "env": env_info()},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    if args.save_every and not args.init:   # 从已有 ckpt 出发时，step 0 就是它，已评过
        torch.save(base.state_dict(), out / "ckpt_0.pt")

    np_rng = np.random.default_rng(args.seed)   # placebo 打乱用
    save_at = {int(x) for x in args.save_at.split(",") if x}
    step, t_start = 0, time.time()
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        while step < args.steps:
            for batch in dl:
                t0 = time.time()
                i1 = batch["image1"].to(args.device)
                if args.inject_shift:   # 健全性测试：SAR 内容右移，reward 的最优点明确偏离当前位置
                    i1 = torch.roll(i1, args.inject_shift, dims=-1)
                i0 = batch["image0"].to(args.device)
                if args.neg:   # 负样本对并入同一次前向：前 B 个是正样本对，后 B 个是负样本对
                    i0, i1 = torch.cat([i0, i0]), torch.cat([i1, batch["image1_neg"].to(args.device)])
                data = base.forward(i0, i1)
                loss, st, active = torch.zeros((), device=args.device), {}, False
                B = len(batch["pair"])
                n_pos = B if args.neg else None
                if args.w_pseudo > 0:
                    affines = None if labels is None else [np.asarray(labels[p]) for p in batch["pair"]]
                    if affines is not None and "T" in batch:   # SAR 被已知 T warp 过：标签精确变为 T∘A
                        affines = [compose(T.numpy(), A) for T, A in zip(batch["T"], affines)]
                    l_ps, st = pseudo_loss(data, s, affines, args.ransac, args.min_inliers, args.w_coarse,
                                           args.w_fine, args.coarse_set, n_pos)
                    loss = args.w_pseudo * l_ps
                    active = st["pairs_used"] > 0
                if args.w_cexp > 0:
                    l_ce, st_ce = coarse_expect_loss(data, s, [False] * B + [True] * B if args.neg else None,
                                                     args.ransac, args.cexp_placebo, np_rng, args.cexp_out)
                    loss = loss + args.w_cexp * l_ce
                    st.update({k: v for k, v in st_ce.items() if k not in st or k == "cexp"})
                    active = active or l_ce.requires_grad
                if args.rl_pair > 0 or args.rl_match > 0:
                    ff = FEATS[args.reward]
                    feats = (ff(data["image0"]), ff(data["image1"])) if args.rl_pair > 0 else None
                    l_rl, st_rl = rl_loss(data, s, args.K, args.sig_g, args.sig_i, args.rl_pair, args.rl_match,
                                          args.ransac, feats, args.reward, args.placebo, n_pos)
                    if args.rl_scope == "fine" and l_rl.requires_grad:   # 只对细级参数求导，先于主 backward
                        rl_grads = torch.autograd.grad(l_rl / args.accum, fine_params, retain_graph=True,
                                                       allow_unused=True)
                        for p, gr in zip(fine_params, rl_grads):
                            if gr is not None:
                                p.grad = gr if p.grad is None else p.grad + gr
                    else:
                        loss = loss + l_rl
                        active = active or l_rl.requires_grad
                    st.update(st_rl, rl=round(float(l_rl), 6))
                if args.w_l2sp > 0:
                    l_sp = sum(((p - p0) ** 2).sum() for p, p0 in zip(params, params0))
                    loss = loss + args.w_l2sp * l_sp
                    st["l2sp"] = round(float(l_sp), 6)
                if active:
                    (loss / args.accum).backward()
                step += 1
                upd = {}
                if step % args.accum == 0:   # 梯度累积：攒 accum 对再更新一次（RL 梯度信噪比低）
                    if any(p.grad is not None for p in params):
                        lr = lr_at(step, args.steps, args.lr, args.warmup, args.warmup_start, args.sched, args.lr_min)
                        for gr in opt.param_groups:
                            gr["lr"] = lr
                        gn = torch.nn.utils.clip_grad_norm_(params, args.clip if args.clip > 0 else float("inf"))
                        opt.step()
                        upd = {"lr": float(f"{lr:.4g}"), "gnorm": round(float(gn), 4)}   # gnorm 为裁剪前
                    opt.zero_grad(set_to_none=True)
                rec = {"step": step, "pairs": batch["pair"], "loss": round(float(loss), 5), **st, **upd,
                       "sec": round(time.time() - t0, 3)}
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if step % 50 == 0:
                    log.flush()
                    keys = ("coarse", "fine", "n_inliers", "pairs_used", "cexp", "n_match", "n_inl", "dist_I",
                            "diag", "neg_n_inl", "neg_n_ident")
                    print(f"step {step}: loss={float(loss):.4f} " + " ".join(f"{k}={st[k]}" for k in keys if k in st)
                          + f" elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
                if (args.save_every and step % args.save_every == 0) or step in save_at:
                    torch.save(base.state_dict(), out / f"ckpt_{step}.pt")
                if step >= args.steps:
                    break
    if args.save_every and step % args.save_every:
        torch.save(base.state_dict(), out / f"ckpt_{step}.pt")


if __name__ == "__main__":
    main()
