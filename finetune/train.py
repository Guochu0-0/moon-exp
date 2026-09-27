"""SCENES 式伪标签微调（「复现 SCENES 式伪标签微调 baseline」#26）。损失见 finetune/pseudo.py。

    python -m finetune.train configs/baselines/anymatch_loftr.json --out $MOON_RESULTS/finetune/scenes \
        [--steps 8000] [--lr 1e-5] [--min-inliers 30] [--save-every 1000]

每一步：前向（模块 eval、开梯度）→ 取伪仿射 → 粗、细两级损失 → 反向。
伪仿射两种来源：--labels 给离线文件（SCENES 做法，只在 keep 的对上训练）；不给则每步用本步匹配在线估计，
内点少于 --min-inliers 的对不监督（后续 RL 方案里的伪标签监督项）。

产物 <out>/：
  log.jsonl          每步一行：loss 分项、内点数、参与监督的对数、耗时
  ckpt_<step>.pt     baselines 适配器可直接读（baselines.match --weights）
  args.json          参数、commit、环境
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

from .data import PairSet
from .label import load_labels
from .model import Base
from .pseudo import pseudo_loss

REPO = Path(__file__).resolve().parents[1]
HW = 512  # patch 原尺寸


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--init", help="起点 ckpt，默认配置里的底座权重")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", help="finetune.label 的离线伪仿射（SCENES 做法）；不给则每步在线估计")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ransac", type=float, default=3.0, help="伪仿射的 RANSAC 阈值（原网格 px），与评测同口径")
    ap.add_argument("--min-inliers", type=int, default=30, help="伪仿射内点少于此数的对不监督")
    ap.add_argument("--w-coarse", type=float, default=1.0)
    ap.add_argument("--w-fine", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=1000)
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

    ds = PairSet(args.data, "train", base.resize, **cfg["input"])
    labels = None
    if args.labels:
        labels = load_labels(args.labels)
        ds.pairs = [p for p in ds.pairs if labels.get(p) is not None]
    g = torch.Generator().manual_seed(args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, generator=g,
                                     num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW([p for p in base.model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=args.wd)
    (out / "args.json").write_text(json.dumps({"args": vars(args), "config": cfg, "weights": weights,
                                               "n_train": len(ds), "commit": git_head(REPO), "env": env_info()},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    if args.save_every:
        torch.save(base.state_dict(), out / "ckpt_0.pt")

    step, t_start = 0, time.time()
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        while step < args.steps:
            for batch in dl:
                t0 = time.time()
                data = base.forward(batch["image0"].to(args.device), batch["image1"].to(args.device))
                affines = None if labels is None else [np.asarray(labels[p]) for p in batch["pair"]]
                loss, st = pseudo_loss(data, s, affines, args.ransac, args.min_inliers, args.w_coarse, args.w_fine)
                opt.zero_grad(set_to_none=True)
                if st["pairs_used"]:
                    loss.backward()
                    opt.step()
                step += 1
                rec = {"step": step, "pairs": batch["pair"], "loss": round(float(loss), 5), **st,
                       "sec": round(time.time() - t0, 3)}
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if step % 50 == 0:
                    log.flush()
                    print(f"step {step}: loss={float(loss):.4f} c={st['coarse']:.4f} f={st['fine']:.4f} "
                          f"inl={st['n_inliers']} used={st['pairs_used']} "
                          f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
                if args.save_every and step % args.save_every == 0:
                    torch.save(base.state_dict(), out / f"ckpt_{step}.pt")
                if step >= args.steps:
                    break
    if args.save_every and step % args.save_every:
        torch.save(base.state_dict(), out / f"ckpt_{step}.pt")


if __name__ == "__main__":
    main()
