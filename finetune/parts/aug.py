"""已知随机扰动（「【伪标签】首轮实验设计」#53，Noisy-Student 式「学生看扰动、老师看干净」）。实现见 finetune/augment.py。

- geo：SAR 施加已知随机仿射 T（绕中心旋转 ±rot 度、缩放 ±scale、平移 ±shift 原网格 px），离线伪标签随之精确变为 T∘A；
- photo：两侧各自随机 gamma、对比度、噪声（SAR 乘性散斑）、轻微模糊；
- erase：SAR 上随机擦除 1–erase_n 块（填 patch 均值），共占 erase_min–erase_max 面积，另返回遮挡掩码 occ；
  [pseudo] 在被遮处不监督位置、certainty 目标为 0（#120；ARFlow 2003.13045、SMURF 2105.07014 §3.2.1、PDC-Net+ 2109.13912 §3.5）；
- blockmask：光学或 SAR 随机一张按 mask_block（原网格 px）分块、每块以 mask_ratio 遮掉（填 patch 均值），被遮处照常监督
  （#127，MIC 2212.01322：遮住了也要从上下文推断出对应）。
"""
from __future__ import annotations

from ..config import ConfigError

PARAMS = {"geo": True, "photo": True, "shift": 12.0, "rot": 3.0, "scale": 0.03,
          "erase": False, "erase_n": 3, "erase_min": 0.10, "erase_max": 0.25,
          "blockmask": False, "mask_block": 32, "mask_ratio": 0.3}
MODELS = {"loftr": {}, "roma": {}}
Term = None


def fill(p):
    return p


def check(cfg):
    a = cfg["aug"]
    if not (a["geo"] or a["photo"] or a["erase"] or a["blockmask"]):
        raise ConfigError("[aug] geo、photo、erase、blockmask 至少开一个")
    if a["erase"] and ("pseudo" not in cfg or cfg["model"]["name"] != "roma"):
        raise ConfigError("[aug] erase 目前只配合 RoMa 的 [pseudo]（遮挡掩码由它使用）")


def data_kw(p):
    kw = {"aug": tuple(k for k in ("geo", "photo", "erase", "blockmask") if p[k]),
          "aug_shift": p["shift"], "aug_rot": p["rot"], "aug_scale": p["scale"]}
    if p["erase"]:
        kw["aug_erase"] = (p["erase_n"], p["erase_min"], p["erase_max"])
    if p["blockmask"]:
        kw["aug_mask"] = (p["mask_block"], p["mask_ratio"])
    return kw
