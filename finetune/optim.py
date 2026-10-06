"""优化配方（「把 S1 调好」#50）：AdamW + 线性 warmup + const / cosine 调度、梯度累积、梯度范数裁剪；
模型要求时（RoMa 的 fp16 autocast）用 GradScaler。参数见 finetune/config.py 的 OPTIM。"""
from __future__ import annotations

import math

import torch


def lr_at(step, total, lr, warmup=0, warmup_start=0.1, sched="const", lr_min=0.0):
    """第 step 个前向步（1 起）之后那次更新用的 lr。warmup 从 warmup_start·lr 线性升到 lr（同上游 LoFTR 的 linear
    warmup）；之后 const 恒定，cosine 在 [warmup, total] 上从 lr 降到 lr_min·lr。步数按前向步计，与 accum 无关。"""
    if step < warmup:
        return lr * (warmup_start + (1 - warmup_start) * step / warmup)
    if sched == "const":
        return lr
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr * (lr_min + (1 - lr_min) * 0.5 * (1 + math.cos(math.pi * t)))


class Optim:
    def __init__(self, params, o: dict, amp: bool, groups=None):
        """groups：可选，[{"params": [...], "lr_scale": x}, ...]，每组的 lr = 调度出的 lr × lr_scale（#120 分模块学习率）；
        不给时全部参数一组。"""
        self.params, self.o = params, o
        groups = groups or [{"params": params, "lr_scale": 1.0}]
        self.opt = torch.optim.AdamW([{"params": g["params"], "lr_scale": g["lr_scale"]} for g in groups],
                                     lr=o["lr"], weight_decay=o["wd"])
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

    def backward(self, loss):
        x = loss / self.o["accum"]
        (self.scaler.scale(x) if self.scaler else x).backward()

    def step(self, step) -> dict:
        """每 accum 步更新一次（有梯度时）。返回要记进日志的 lr、裁剪前的梯度范数（及 GradScaler 的 scale）。"""
        o, upd = self.o, {}
        if step % o["accum"]:
            return upd
        if any(p.grad is not None for p in self.params):
            lr = lr_at(step, o["steps"], o["lr"], o["warmup"], o["warmup_start"], o["sched"], o["lr_min"])
            for g in self.opt.param_groups:
                g["lr"] = lr * g["lr_scale"]
            if self.scaler:
                self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.params, o["clip"] if o["clip"] > 0 else float("inf"))
            if self.scaler:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            upd = {"lr": float(f"{lr:.4g}"), "gnorm": round(float(gn), 4)}
            if self.scaler:
                upd["scale"] = self.scaler.get_scale()
        self.opt.zero_grad(set_to_none=True)
        return upd
