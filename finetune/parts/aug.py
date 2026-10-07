"""已知随机扰动（「【伪标签】首轮实验设计」#53，Noisy-Student 式「学生看扰动、老师看干净」）。实现见 finetune/augment.py。

- geo：SAR 施加已知随机仿射 T（绕中心旋转 ±rot 度、缩放 ±scale、平移 ±shift 原网格 px），离线伪标签随之精确变为 T∘A；
- photo：两侧各自随机 gamma、对比度、噪声（SAR 乘性散斑）、轻微模糊；
- erase：SAR 上随机擦除 1–erase_n 块（填 patch 均值），共占 erase_min–erase_max 面积，另返回遮挡掩码 occ；
  [pseudo] 在被遮处不监督位置、certainty 目标为 0（#120；ARFlow 2003.13045、SMURF 2105.07014 §3.2.1、PDC-Net+ 2109.13912 §3.5）。
"""
from __future__ import annotations

from ..config import ConfigError

PARAMS = {"geo": True, "photo": True, "shift": 12.0, "rot": 3.0, "scale": 0.03,
          "erase": False, "erase_n": 3, "erase_min": 0.10, "erase_max": 0.25}
MODELS = {"loftr": {}, "roma": {}}
Term = None


def fill(p):
    return p


def check(cfg):
    a = cfg["aug"]
    if not (a["geo"] or a["photo"] or a["erase"]):
        raise ConfigError("[aug] geo、photo、erase 至少开一个")
    if a["erase"] and ("pseudo" not in cfg or cfg["model"]["name"] != "roma"):
        raise ConfigError("[aug] erase 目前只配合 RoMa 的 [pseudo]（遮挡掩码由它使用）")


def data_kw(p):
    kw = {"aug": tuple(k for k in ("geo", "photo", "erase") if p[k]),
          "aug_shift": p["shift"], "aug_rot": p["rot"], "aug_scale": p["scale"]}
    if p["erase"]:
        kw["aug_erase"] = (p["erase_n"], p["erase_min"], p["erase_max"])
    return kw
