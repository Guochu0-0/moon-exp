"""锚回起点的权重正则 L2-SP（#122；Li et al. 2018, arXiv 1802.01483；WeSTAR 2511.14238 的「Weight Regularization」）。

损失 = weight · ( Σ‖θ − θ₀‖² + Σ‖ΔW_LoRA‖² )：θ 为参与训练的 decoder / VGG 参数，θ₀ 为训练开始时的值（zero-shot）；
DINOv2 上 LoRA 的增量 ΔW = scale·B A 以 0 为锚（起点就是 0）。weight 的取法见所在实验的 Notes。
"""
from __future__ import annotations

import torch

PARAMS = {"weight": 0.0}
MODELS = {"roma": {}}


def fill(p):
    return p


def check(cfg):
    pass


def data_kw(p):
    return {}


class Term:
    def __init__(self, p, model, ds):
        self.p, self.model = p, model
        lora = {id(x) for l in getattr(model, "lora", {}).values() for x in (l.A, l.B)}
        self.anchor = [(q, q.detach().clone()) for q in model.params if id(q) not in lora]

    def __call__(self, st):
        d = sum(((q.float() - q0.float()) ** 2).sum() for q, q0 in self.anchor)
        for l in getattr(self.model, "lora", {}).values():
            d = d + ((l.B @ l.A) * l.scale).pow(2).sum()
        return self.p["weight"] * d, {"l2sp_d2": round(float(d), 6)}, True
