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

    def factor(self) -> float:
        """backward(loss) 实际反向的是 loss × factor()：GradScaler 的当前 scale（没有时为 1）÷ accum。"""
        return (self.scaler.get_scale() if self.scaler else 1.0) / self.o["accum"]

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


class UpdateMonitor:
    """每次参数更新的相对改变量（#197）：各部分 ‖θ_新 − θ_旧‖ / ‖θ_旧‖（范数取部分内全部参数拼成的向量）。
    LoRA 层按等效权重算：ΔW = scale·(B_新 A_新 − B_旧 A_旧)，相对原权重 W；各层合起来为 √ΣΔW² / √ΣW²。
    parts：部分 → 参数列表；lora：[(A, B, scale, W)]。另给 fine（名字以 ref 开头的部分与 vgg 合起来，即细化各级与它们的特征）。
    只计这一步收到梯度的参数（before 在 optim.step 之前、梯度已在时调用）：没加噪的细化级没有梯度、AdamW 不动它们，
    计进分母会把 fine 拉低。整个部分都没有梯度时这一部分不出现在结果里。"""

    def __init__(self, parts: dict, lora=()):
        self.parts, self.lora = parts, list(lora)
        self._old = None

    @torch.no_grad()
    def before(self):
        live = {k: [p for p in ps if p.grad is not None] for k, ps in self.parts.items()}
        self._live = {k: ps for k, ps in live.items() if ps}
        self._old = ({k: [p.detach().float().clone() for p in ps] for k, ps in self._live.items()},
                     [(A.detach().float().clone(), B.detach().float().clone()) for A, B, _, _ in self.lora])

    @torch.no_grad()
    def after(self) -> dict:
        old, old_l = self._old
        num, den = {}, {}
        for k, ps in self._live.items():
            num[k] = sum(float((p.detach().float() - q).square().sum()) for p, q in zip(ps, old[k]))
            den[k] = sum(float(q.square().sum()) for q in old[k])
        if self.lora:
            num["lora"], den["lora"] = 0.0, 0.0
            for (A, B, sc, W), (A0, B0) in zip(self.lora, old_l):
                X = torch.cat([B.detach().float(), -B0], 1)          # B'A' − BA = [B', −B] [A'; A]
                Y = torch.cat([A.detach().float(), A0], 0)
                num["lora"] += sc ** 2 * float(((X.T @ X) * (Y @ Y.T)).sum())    # ‖XY‖² = Σ (XᵀX) ⊙ (YYᵀ)
                den["lora"] += float(W.detach().float().square().sum())
        fine = [k for k in num if k.startswith("ref") or k == "vgg"]
        if fine:
            num["fine"], den["fine"] = sum(num[k] for k in fine), sum(den[k] for k in fine)
        self._old = None
        return {f"upd_{k}": float(f"{math.sqrt(num[k] / den[k]):.4g}") if den[k] > 0 else 0.0 for k in num}
