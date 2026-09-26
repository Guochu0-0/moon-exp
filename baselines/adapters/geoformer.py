"""GeoFormer（ruc-aimc-lab/GeoFormer，ICCV 2023，third_party/GeoFormer，8b9506e）。

对照官方（调研 tmp/interfaces/geoformer.md，均带 file:line）：
- 模型：`model.full_model.GeoFormer(loftr_cfg, geo_cfg)`，两份配置取 `cvpr_ds_config.default_cfg` 与
  `geo_config.default_cfg`。官方 `dict(default_cfg)` 是浅拷贝会改全局，这里 deepcopy。粗匹配阈值 0.2
  （geo_cfg['coarse_thr'] 覆盖 LoFTR thr），精匹配 0.1 固定；无 reparam。
- 权重：唯一发布的 geoformer.ckpt，`['state_dict']`，无前缀；多出的 `geo_module.merge_desc.*` 在代码里已注释掉，
  官方 strict=False，这里核对无缺键。推断为 MegaDepth 训练版。
- 输入：`(1,1,H,W)` float [0,1]，forward 内无归一化；H、W 为 8 的倍数。官方测试分辨率：短边 480（只缩不放，
  向下取整到 8）→ 512² 缩到 480²，cv2 双线性。不传 scale0/1（会影响 GeoModule 的窗口计算），外面自己回映。
- 输出：`mkpts0_f/mkpts1_f/mconf`，缩放网格上的整数、整数 = 像素中心，按中心对齐公式回映。空时返回空，无假匹配。
- 依赖：import 链里 utils/common_utils.py 要 skimage，utils/homography.py 要 imgaug、scipy.stats、torchvision，
  推理用不到 skimage/imgaug，没装时打桩。仓库顶层包名是 `model`、`utils`，import 前清掉 sys.modules 里的同名项。
"""
from __future__ import annotations

import copy
import sys
import types

import numpy as np

from .base import load_ckpt, to_original


def _stub(name, **attrs):
    try:
        __import__(name)
        return False
    except Exception:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
        for i in range(1, len(parts)):
            setattr(sys.modules[".".join(parts[:i])], parts[i], sys.modules[".".join(parts[:i + 1])])
        for k, v in attrs.items():
            setattr(sys.modules[name], k, v)
        return True


class GeoFormerAdapter:
    def __init__(self, repo, weights, device="cuda", short_side=480, coarse_thr=0.2):
        sys.path.insert(0, str(repo))
        for k in [k for k in sys.modules if k in ("model", "utils") or k.startswith(("model.", "utils."))]:
            del sys.modules[k]
        stubbed = []
        if _stub("skimage.feature", peak_local_max=None):
            stubbed.append("skimage")
        if _stub("imgaug.augmenters"):
            stubbed.append("imgaug")
        from model.full_model import GeoFormer
        from model.geo_config import default_cfg as geo_cfg
        from model.loftr_src.loftr.utils.cvpr_ds_config import default_cfg as loftr_cfg

        lcfg, gcfg = copy.deepcopy(dict(loftr_cfg)), copy.deepcopy(dict(geo_cfg))
        lcfg["match_coarse"]["thr"] = coarse_thr
        gcfg["coarse_thr"] = coarse_thr
        self.model = GeoFormer(lcfg, gcfg)
        sd = load_ckpt(weights)
        sd = sd.get("state_dict", sd)
        res = self.model.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"GeoFormer 权重缺键 {len(res.missing_keys)} 个，例如 {res.missing_keys[:5]}")
        self.model = self.model.eval().to(device)
        self.device, self.short_side = device, short_side
        self.notes = (f"short_side={short_side} (shrink only), coarse_thr={coarse_thr}, fine 0.1, "
                      f"unexpected_keys={len(res.unexpected_keys)}, stubbed={stubbed}")

    def _tensor(self, img):
        import cv2
        import torch

        h, w = img.shape
        hn, wn = h, w
        if self.short_side and min(h, w) > self.short_side:
            s = self.short_side / min(h, w)
            hn, wn = int(round(h * s)), int(round(w * s))
        hn, wn = hn // 8 * 8, wn // 8 * 8
        if (hn, wn) != (h, w):
            img = cv2.resize(np.ascontiguousarray(img, np.float32), (wn, hn), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(img, np.float32))[None, None].to(self.device), (h, w, hn, wn)

    def match(self, opt, sar):
        import torch

        t0, g0 = self._tensor(opt)
        t1, g1 = self._tensor(sar)
        batch = {"image0": t0, "image1": t1}
        with torch.no_grad():
            self.model(batch)
        kp0 = to_original(batch["mkpts0_f"].float().cpu().numpy(), *g0)
        kp1 = to_original(batch["mkpts1_f"].float().cpu().numpy(), *g1)
        return kp0, kp1, batch["mconf"].float().cpu().numpy().reshape(-1)
