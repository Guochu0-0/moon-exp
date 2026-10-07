"""公共训练循环（「训练代码重构」#86）。一个方法 = 一份配置（finetune/config.py），由模型适配器（finetune/models/）
和若干训练成分（finetune/parts/）组成。平时由驱动 scripts/finetune/run.py 调用，也可以单独跑：

    python -m finetune.train runs/<id>/configs/<方法>.toml --out runs/<id>/ckpt/<方法> [--tb runs/<id>/tb/<方法>/scalars]

每一步：取一个 batch → 各成分按登记顺序算损失（需要时共用一次前向：正样本对 + 负样本对并在一起）→ 求和 → 反向；
每 accum 步更新一次。--stop-at 只用于核对：提前停下，lr 调度仍按配置的总步数算。

产物 <out>/：
  log.jsonl          每步一行：总损失、各成分的监控量、lr、梯度范数、耗时
  ckpt_<step>.pt     baselines 适配器可直接读（baselines.match --weights）
  args.json          展开后的完整配置、推理配置、权重、commit、环境
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
from workbench.launch import require_clean
from workbench.tbwriter import ScalarWriter

from . import config as C
from .data import PairSet
from .models import MODELS
from .optim import Optim
from .parts import PARTS

REPO = Path(__file__).resolve().parents[1]
PRINT_KEYS = ("coarse", "fine", "pairs_used", "epe1_px", "cexp", "lcert", "n_match", "n_inl", "inl_frac", "dist_I",
              "diag", "neg_n_inl", "neg_n_ident")


class Step:
    """一步的上下文：batch、共用的前向（第一次用到时才算）、随机数发生器。"""

    def __init__(self, model, model_name, batch, device, neg, rng):
        self.model, self.model_name, self.batch, self.device, self.neg, self.rng = \
            model, model_name, batch, device, neg, rng
        self.B = len(batch["pair"])
        self._out = None

    def images(self):
        return self.batch["image0"].to(self.device), self.batch["image1"].to(self.device)

    @property
    def out(self):
        """正样本对（有 [neg] 时后面接同样多的负样本对）的一次前向。"""
        if self._out is None:
            i0, i1 = self.images()
            if self.neg:
                i0, i1 = torch.cat([i0, i0]), torch.cat([i1, self.batch["image1_neg"].to(self.device)])
            self._out = self.model.forward(i0, i1)
        return self._out


class Ema:
    """权重 EMA（#120；RoMa v2 2511.15706 §3.3 的做法）：每次参数更新后 e ← d·e + (1−d)·θ，只跟踪参与训练的参数；
    存 ckpt 时存 EMA 权重（其余参数、BN 统计与当前模型相同），评测的就是 EMA 模型。"""

    def __init__(self, model, decay):
        self.model, self.decay = model, decay
        self.shadow = {n: p.detach().clone().float() for n, p in model.model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self):
        for n, p in self.model.model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    def state_dict(self):
        sd = self.model.state_dict()
        for n, e in self.shadow.items():
            sd["model"][n] = e.to(sd["model"][n].dtype).cpu()
        return sd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="runs/<id>/configs/<方法>.toml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tb", help="TB 标量目录（runs/<id>/tb/<方法>/scalars）；不给则只写 log.jsonl")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stop-at", type=int, default=0, help="只跑前 N 步（核对用），lr 调度不变")
    args = ap.parse_args(argv)
    require_clean(REPO)   # 先提交再跑（#76）

    cfg = C.load(args.config)
    o, r, mname = cfg["optim"], cfg["run"], cfg["model"]["name"]
    names = C.parts_of(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(r["seed"])
    mod = MODELS[mname]
    os.environ.setdefault("TORCH_HOME", mod.TORCH_HOME)
    model = mod.Model(cfg, args.weights_root, device=args.device)

    kw = {}
    for n in names:
        kw.update(PARTS[n].data_kw(cfg[n]))
    ds = PairSet(args.data, r["split"], model.resize, **model.input, seed=r["seed"], **kw)
    terms = [(n, PARTS[n].Term(cfg[n], model, ds)) for n in names if PARTS[n].Term is not None]
    if r["limit"]:
        ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // r["limit"])][: r["limit"]]   # 均匀取样，覆盖各 ROI
    g = torch.Generator().manual_seed(r["seed"])
    dl = torch.utils.data.DataLoader(ds, batch_size=o["batch"], shuffle=True, generator=g, num_workers=r["workers"],
                                     drop_last=True, persistent_workers=r["workers"] > 0)
    optim = Optim(model.params, o, model.amp, getattr(model, "groups", None))
    ema = Ema(model, o["ema"]) if o["ema"] > 0 else None
    (out / "args.json").write_text(json.dumps(
        {"config": cfg, "source": str(args.config), "infer_config": model.infer_cfg, "weights": model.weights,
         "n_train": len(ds), "n_params": sum(p.numel() for p in model.params), "commit": git_head(REPO),
         "env": env_info()}, ensure_ascii=False, indent=1), encoding="utf-8")
    if r["save_zero"]:
        torch.save(model.state_dict(), out / "ckpt_0.pt")
    save = lambda st: torch.save(ema.state_dict() if ema else model.state_dict(), out / f"ckpt_{st}.pt")

    rng = np.random.default_rng(r["seed"])   # 打乱分数（placebo）、交换方向（RoMa 伪标签）共用
    neg = "neg" in cfg
    total = min(o["steps"], args.stop_at) if args.stop_at else o["steps"]
    save_at = set(r["save_at"])
    step, t_start = 0, time.time()
    tbw = ScalarWriter(args.tb) if args.tb else None
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        while step < total:
            for batch in dl:
                t0 = time.time()
                st = Step(model, mname, batch, args.device, neg, rng)
                loss, stats, active = torch.zeros((), device=args.device), {}, False
                for n, term in terms:
                    l, s, a = term(st)
                    loss = loss + l
                    stats.update({k: v for k, v in s.items() if k not in stats or k == n})   # 先算的成分优先
                    active = active or a
                if active:
                    optim.backward(loss)
                step += 1
                upd = optim.step(step)
                if ema and upd:
                    ema.update()
                rec = {"step": step, "pairs": batch["pair"], "loss": round(float(loss), 5), **stats, **upd,
                       "sec": round(time.time() - t0, 3)}
                if step == 1 and torch.cuda.is_available():
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if tbw:
                    tbw.add_dict(rec, step)
                if step % 50 == 0:
                    log.flush()
                    print(f"step {step}: loss={float(loss):.4f} "
                          + " ".join(f"{k}={stats[k]}" for k in PRINT_KEYS if k in stats)
                          + f" elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
                if (r["save_every"] and step % r["save_every"] == 0) or step in save_at:
                    save(step)
                if step >= total:
                    break
    if tbw:
        tbw.close()
    if r["save_every"] and step % r["save_every"]:
        save(step)


if __name__ == "__main__":
    main()
