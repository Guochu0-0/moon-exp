"""负样本对（#48 建议 ②、#49）：每对再配一张来自其他 ROI 的 SAR，与正样本对并入同一次前向
（前 B 个是正样本对，后 B 个是负样本对）。负样本对上怎么给分由打分成分决定（cexp：内点 −1、外点 0）。"""
from __future__ import annotations

from ..config import ConfigError

PARAMS = {}
MODELS = {"loftr": {}, "roma": {}}
Term = None


def fill(p):
    return p


def check(cfg):
    if "cexp" not in cfg:
        raise ConfigError("[neg] 只配合打分成分 [cexp] 使用")


def data_kw(p):
    return {"neg": True}
