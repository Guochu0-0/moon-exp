"""已知随机扰动（「【伪标签】首轮实验设计」#53，Noisy-Student 式「学生看扰动、老师看干净」）。实现见 finetune/augment.py。

- geo：SAR 施加已知随机仿射 T（绕中心旋转 ±rot 度、缩放 ±scale、平移 ±shift 原网格 px），离线伪标签随之精确变为 T∘A；
- photo：两侧各自随机 gamma、对比度、噪声（SAR 乘性散斑）、轻微模糊。
"""
from __future__ import annotations

from ..config import ConfigError

PARAMS = {"geo": True, "photo": True, "shift": 12.0, "rot": 3.0, "scale": 0.03}
MODELS = {"loftr": {}, "roma": {}}
Term = None


def fill(p):
    return p


def check(cfg):
    if not (cfg["aug"]["geo"] or cfg["aug"]["photo"]):
        raise ConfigError("[aug] geo、photo 至少开一个")


def data_kw(p):
    return {"aug": tuple(k for k in ("geo", "photo") if p[k]),
            "aug_shift": p["shift"], "aug_rot": p["rot"], "aug_scale": p["scale"]}
