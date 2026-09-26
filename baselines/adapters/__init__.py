"""适配器登记表。按名字懒加载，避免在一个方法的环境里 import 另一个方法的依赖。

适配器契约：
    a = Adapter(repo=..., weights=..., device=..., **params)
    kp0, kp1, conf = a.match(opt, sar)
- opt、sar：H×W float32，值域 [0,1]（映射见 baselines/inputs.py）。适配器从张量层接入官方 pipeline，
  跳过官方的 uint8 loader；不得不量化的地方写进 a.notes。
- kp0（光学）、kp1（SAR）：N×2 float，**原 512 网格、0-based、整数 = 像素中心**。内部 resize 过的由适配器映射回来。
- conf：N，或 None。
- 输出的是 RANSAC 之前的原始匹配；统一的仿射估计在 baselines/fit.py。
"""
from __future__ import annotations

import importlib

REGISTRY = {
    "identity": "baselines.adapters.identity:Identity",
    "loftr": "baselines.adapters.loftr:LoFTRAdapter",
    "xoftr": "baselines.adapters.xoftr:XoFTRAdapter",
    "spsg": "baselines.adapters.spsg:SPSGAdapter",
    "matchanything": "baselines.adapters.matchanything:MatchAnythingAdapter",
    "romatch": "baselines.adapters.romatch:RomatchAdapter",
    "splg": "baselines.adapters.splg:SPLGAdapter",
    "eloftr": "baselines.adapters.eloftr:ELoFTRAdapter",
    "edm": "baselines.adapters.edm:EDMAdapter",
    "geoformer": "baselines.adapters.geoformer:GeoFormerAdapter",
}


def load(name: str):
    if name not in REGISTRY:
        raise KeyError(f"未知的适配器 {name!r}，可选 {sorted(REGISTRY)}")
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)
