"""公共训练循环（「训练代码重构」#86）。一个方法 = 一份配置（finetune/config.py），由模型适配器（finetune/models/）
和若干训练成分（finetune/parts/）组成。平时由驱动 scripts/finetune/run.py 调用，也可以单独跑：

    python -m finetune.train runs/<id>/configs/<方法>.toml --out runs/<id>/ckpt/<方法> [--tb runs/<id>/tb/<方法>/scalars]

每一步：取一个 batch → 各成分按登记顺序算损失（需要时共用一次前向：正样本对 + 负样本对并在一起）→ 求和 → 反向；
每 accum 步更新一次。--stop-at 只用于核对：提前停下，lr 调度仍按配置的总步数算。

产物 <out>/：
  log.jsonl          每步一行：总损失、各成分的监控量、lr、梯度范数、耗时
  ckpt_<step>.pt     baselines 适配器可直接读（baselines.match --weights）；[run] save_optim 时另有 'train'：
                     续训状态（步数、AdamW 与 GradScaler、随机数、数据顺序、各成分的状态，#209）
  args.json          展开后的完整配置、推理配置、权重、commit、环境
  trace/             成分的逐步记录（有成分往 step.trace 里放东西或提供 observe 时，#209，见 finetune/trace.py）

[run] resume 给出 save_optim 存下的 ckpt 时，装回权重与续训状态，从下一步接着训（只支持第一个 epoch 内的存档）；
[run] stop_at 与 --stop-at 一样提前停下，lr 调度仍按 [optim] steps。
[run] halt_above / halt_below（#221）：某个日志量最近 halt_window 个记录的中位数越过阈值时停训，写 <out>/halt.json
（停在第几步、哪个量、中位数），不另存 ckpt（最后一个存档仍是越线前按 save_every / save_at 存的）；正常退出。
成分可以提供 after_update(step)（参数更新之后调用）、observe(step)（每步更新之后与训练前调用，返回要存的数组或 None）、
state_dict() / load_state_dict()（随续训状态存取）、ddp_tensors()（多卡时要检查各卡一致的张量，如 EMA 参照的权重）。

[run] ddp = N > 0（#226）：单机 N 卡数据并行，由 torchrun 起 N 个进程（finetune/dist.py）：

    torchrun --standalone --nproc_per_node=N -m finetune.train <配置> --out ...

每个进程一张卡、每步读 1 个 batch（DistributedSampler 把训练集分给各卡），全部反向做完后各卡梯度取平均，再做同一次
参数更新，所以一次更新用 N × batch 对图。步数、warmup、存档、stop_at 都按更新次数（每个进程的步数）计；
halt_window 按记录计，每次更新进 N 个记录（各卡各一对图的日志量），即 halt_window = 50 × N 时窗口为 50 次更新。
日志 log.jsonl 每次更新一行，日志量为各卡的平均，pairs 为各卡的图对。只有 rank 0 写日志、存档；参数在开训前从 rank 0
广播，每次存档与训练结束时检查各卡参数（及成分的 ddp_tensors）与 rank 0 的最大差，写进 <out>/ddp_sync.jsonl。
提前停由全部记录在各卡上同样算出，所以各卡同时停。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from baselines.match import env_info, git_head, seed_all
from workbench.launch import require_clean
from workbench.tbwriter import ScalarWriter

from . import config as C
from . import dist as D
from .data import PairSet
from .models import MODELS
from .optim import Optim, UpdateMonitor
from .parts import PARTS
from .trace import Loader, TraceWriter, rng_state, set_rng_state

REPO = Path(__file__).resolve().parents[1]
PRINT_KEYS = ("coarse", "fine", "pairs_used", "epe1_px", "cexp", "lcert", "n_match", "n_inl", "inl_frac", "dist_I",
              "diag", "neg_n_inl", "neg_n_ident")


class Step:
    """一步的上下文：batch、共用的前向（第一次用到时才算）、随机数发生器；optim 供要在成分内部分段反向的成分用
    （optim.backward、optim.factor()，#192 细化加噪）。"""

    def __init__(self, model, model_name, batch, device, neg, rng, optim=None):
        self.model, self.model_name, self.batch, self.device, self.neg, self.rng, self.optim = \
            model, model_name, batch, device, neg, rng, optim
        self.B = len(batch["pair"])
        self._out = None
        self.trace = {}     # 成分的逐步记录（#209）：名字 → 数组，第 0 维是 batch 里的第几对

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


class Halt:
    """提前停（#221）：above、below 为 日志量 → 阈值。每步把 stats 里有的量记进各自最近 window 个的窗口；
    窗口满后中位数 > above[k] 或 < below[k] 就返回 {key, op, thr, median, window}，否则 None。NaN 不进窗口。
    跑满 window 步后仍有量一次也没出现（多半是键名写错），打印一次警告。窗口不随续训状态保存，resume 后重新攒。"""

    def __init__(self, above: dict, below: dict, window: int):
        self.rules = [(k, ">", float(v)) for k, v in above.items()] + [(k, "<", float(v)) for k, v in below.items()]
        self.window = window
        self.buf = {k: collections.deque(maxlen=window) for k, _, _ in self.rules}
        self.calls = 0

    def __bool__(self):
        return bool(self.rules)

    def __call__(self, *records: dict):
        """一次更新的日志量：单卡时一个 dict；多卡时每卡一个（#226），都进窗口后再判断一次。"""
        self.calls += 1
        for stats in records:
            for k, _, _ in self.rules:
                if k in stats and not math.isnan(float(stats[k])):
                    self.buf[k].append(float(stats[k]))
        if self.calls == self.window and any(not b for b in self.buf.values()):
            print(f"WARNING halt：前 {self.window} 步日志里没有 {[k for k, b in self.buf.items() if not b]}", flush=True)
        for k, op, thr in self.rules:
            b = self.buf[k]
            if len(b) == self.window:
                med = float(np.median(b))
                if (med > thr) if op == ">" else (med < thr):
                    return {"key": k, "op": op, "thr": thr, "median": round(med, 5), "window": self.window}
        return None


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
    rank = D.init(r["ddp"])      # 多卡（#226）：每个进程绑一张卡；单卡时为 0
    torch.backends.cudnn.benchmark = r["cudnn_benchmark"]
    main_ = rank == 0
    names = C.parts_of(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(r["seed"])
    mod = MODELS[mname]
    os.environ.setdefault("TORCH_HOME", mod.TORCH_HOME)
    model = mod.Model(cfg, args.weights_root, device=args.device)
    if r["ddp"]:
        D.broadcast(model.params)       # 各卡从同一份参数起步（LoRA 的 A 是随机初始化的）
        seed_all(r["seed"] + 1000 * rank)   # 之后各卡的抽签、加噪各不相同

    kw = {}
    for n in names:
        kw.update(PARTS[n].data_kw(cfg[n]))
    ds = PairSet(args.data, r["split"], model.resize, **model.input, seed=r["seed"],
                 resize_hi=getattr(model, "resize_hi", None), **kw)
    terms = [(n, PARTS[n].Term(cfg[n], model, ds)) for n in names if PARTS[n].Term is not None]
    if r["limit"]:
        ds.pairs = ds.pairs[:: max(1, len(ds.pairs) // r["limit"])][: r["limit"]]   # 均匀取样，覆盖各 ROI
    g = torch.Generator().manual_seed(r["seed"] + 1000 * rank)
    # 多卡：DistributedSampler 按 (seed, epoch) 打乱后分给各卡，各卡读不同的图；训练集补齐到卡数的整数倍（#226）
    smp = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=r["seed"]) if r["ddp"] else None
    dl = torch.utils.data.DataLoader(ds, batch_size=o["batch"], shuffle=smp is None, sampler=smp, generator=g,
                                     num_workers=r["workers"], drop_last=True, persistent_workers=r["workers"] > 0)
    optim = Optim(model.params, o, model.amp, getattr(model, "groups", None), reduce=D.average_grads)
    ema = Ema(model, o["ema"]) if o["ema"] > 0 else None
    if r["log_update"] and not hasattr(model, "update_parts"):
        raise C.ConfigError(f"[run] log_update 要求模型提供 update_parts，{mname} 没有")
    mon = UpdateMonitor(*model.update_parts()) if r["log_update"] else None
    if main_:
        (out / "args.json").write_text(json.dumps(
            {"config": cfg, "source": str(args.config), "infer_config": model.infer_cfg, "weights": model.weights,
             "n_train": len(ds), "n_params": sum(p.numel() for p in model.params), "commit": git_head(REPO),
             "env": env_info(), "world": D.world()}, ensure_ascii=False, indent=1), encoding="utf-8")
    if r["save_zero"] and main_:
        torch.save(model.state_dict(), out / "ckpt_0.pt")
    rng = np.random.default_rng(r["seed"] + 1000 * rank)   # 打乱分数（placebo）、交换方向（RoMa 伪标签）共用
    loader = Loader(dl, g)

    def sync_check(st):
        """多卡：各卡参数（及成分的 ddp_tensors）与 rank 0 的最大差，记进 ddp_sync.jsonl（#226）。"""
        if not r["ddp"]:
            return
        rec = {"step": st, "params": D.max_diff(model.params)}
        for n, t in terms:
            ts = t.ddp_tensors() if hasattr(t, "ddp_tensors") else []
            if ts:
                rec[n] = D.max_diff(ts)
        if main_:
            with open(out / "ddp_sync.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"ddp sync step {st}: {rec}", flush=True)

    def save(st):
        sync_check(st)
        if not main_:
            return
        sd = ema.state_dict() if ema else model.state_dict()
        if r["save_optim"]:
            sd["train"] = {"step": st, "optim": optim.state_dict(), "rng": rng_state(rng), "loader": loader.state(),
                           "terms": {n: t.state_dict() for n, t in terms if hasattr(t, "state_dict")}}
        torch.save(sd, out / f"ckpt_{st}.pt")

    neg = "neg" in cfg
    stops = [x for x in (args.stop_at, r["stop_at"]) if x]
    total = min([o["steps"], *stops])
    save_at = set(r["save_at"])
    halt, halted = Halt(r["halt_above"], r["halt_below"], r["halt_window"]), None
    step, t_start = 0, time.time()
    if r["resume"]:
        sd = torch.load(REPO / r["resume"], map_location="cpu")    # 绝对路径时 REPO / 不起作用
        if "train" not in sd:
            raise C.ConfigError(f"[run] resume：{r['resume']} 里没有续训状态（存它的运行要开 save_optim）")
        model.load_state_dict(sd)
        tr = sd["train"]
        optim.load_state_dict(tr["optim"])
        loader.load(tr["loader"])
        for n, t in terms:
            if hasattr(t, "load_state_dict"):
                t.load_state_dict(tr["terms"][n])
        step = tr["step"]
        set_rng_state(tr["rng"], rng)
        del sd, tr
    tbw = ScalarWriter(args.tb) if args.tb and main_ else None
    trw = TraceWriter(out)       # 没有成分放记录时什么也不写（多卡时成分不放记录，见 [rloo] 的检查）

    def observe(st):
        for n, t in terms:
            o_ = t.observe(st) if hasattr(t, "observe") else None
            if o_ is not None:
                trw.obs(st, o_)

    if not r["resume"]:
        observe(0)
    with open(out / "log.jsonl" if main_ else os.devnull, "a", encoding="utf-8") as log:
        while step < total:
            for batch in loader:
                t0 = time.time()
                st = Step(model, mname, batch, args.device, neg, rng, optim)
                loss, stats, active = torch.zeros((), device=args.device), {}, False
                for n, term in terms:
                    l, s, a = term(st)
                    loss = loss + l
                    stats.update({k: v for k, v in s.items() if k not in stats or k == n})   # 先算的成分优先
                    active = active or a
                if active:
                    optim.backward(loss)
                step += 1
                if mon and step % o["accum"] == 0:
                    mon.before()
                upd = optim.step(step)
                if mon and upd:
                    upd.update(mon.after())
                if ema and upd:
                    ema.update()
                for n, term in terms:
                    if hasattr(term, "after_update"):
                        term.after_update(st)
                trw.add(step, st.trace)
                observe(step)
                pairs, loss_v, rows = batch["pair"], round(float(loss), 5), [stats]
                if r["ddp"]:    # 各卡的日志量：日志里取平均，提前停的窗口逐卡都进（#226）
                    rows = D.gather({"pairs": batch["pair"], "loss": float(loss), "stats": stats})
                    pairs = [q for x in rows for q in x["pairs"]]
                    loss_v = round(float(np.mean([x["loss"] for x in rows])), 5)
                    rows = [x["stats"] for x in rows]
                    stats = D.mean_stats(rows)
                rec = {"step": step, "pairs": pairs, "loss": loss_v, **stats, **upd,
                       "sec": round(time.time() - t0, 3)}
                if step == 1 and torch.cuda.is_available():
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if tbw:
                    tbw.add_dict(rec, step)
                if step % 50 == 0 and main_:
                    log.flush()
                    print(f"step {step}: loss={loss_v:.4f} "
                          + " ".join(f"{k}={stats[k]}" for k in PRINT_KEYS if k in stats)
                          + f" elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
                if (r["save_every"] and step % r["save_every"] == 0) or step in save_at:
                    save(step)
                halted = halt(*rows) if halt else None
                if halted:
                    halted["step"] = step
                    if main_:
                        (out / "halt.json").write_text(json.dumps(halted, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
                    print(f"HALT step {step}: {halted}", flush=True)
                if step >= total or halted:
                    break
            if halted:
                break
    if tbw:
        tbw.close()
    trw.flush()
    if r["save_every"] and step % r["save_every"] and not halted:
        save(step)
    elif not (r["save_every"] and step % r["save_every"] == 0):
        sync_check(step)        # 最后一步没有存档时也检查一次（多卡）
    D.close()


if __name__ == "__main__":
    main()
