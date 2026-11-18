"""RoMa（romatch）系：MINIMA-RoMa（third_party/RoMa_minima，0d3fd22）与 AnyMatch-RoMa（AnyMatch 的 third_party/RoMa_AnyMatch，
与前者只差一行注释）。MatchAnything-RoMa 是另一份分叉，见 matchanything.py。

对照官方（docs/research/baselines/interfaces-b.md §7；AnyMatch 调研 tmp/interfaces/anymatch.md）：
- 构造：`roma_outdoor(device, weights=state_dict)`（MINIMA load_model.py:7-34）。MINIMA 的 .pth 就是 state_dict；
  AnyMatch 的是训练 ckpt，取 `['model']`。DINOv2 由 hub 缓存提供（TORCH_HOME），VGG 分支 pretrained=False。
  `roma_model` 内部 strict 加载。coarse 560、upsample 864、symmetric、threshold_balanced、sample_thresh 0.05。
- 输入：官方 wrapper（MINIMA/AnyMatch data_io_roma.py）BGR→RGB → cv2 双线性长边 640（df 8）→ /255 → ToPILImage
  → `model.match(pilA, pilB, batched=False)`，match 内部拉伸到 560/864 并做 ImageNet 归一化。
  `match()` 只收 PIL RGB（check_rgb 读 .mode），没有张量入口，所以**这里在 640 网格上量化到 uint8 不可避免**：
  float [0,1] 按 torchvision ToPILImage 的方式 `mul(255).byte()`（截断）转成 PIL，其余照官方。
- 输出：`sample(warp, cert)`（默认 num=10000，随机，runner 每 pair 重置种子）→ `to_pixel_coordinates` 在 640 网格，
  像素中心在 +0.5（matcher.py 网格 linspace(-1+1/h, 1-1/h)）→ 先 −0.5 再按中心对齐公式回映 512。
  官方 wrapper 直接乘 scale，不做 ±0.5。
"""
from __future__ import annotations

import sys
import inspect

import numpy as np

from .base import load_ckpt, long_side_size, to_original


class RomatchAdapter:
    def __init__(self, repo, weights, device="cuda", weights_key=None, long_side=640, coarse_res=None,
                 upsample_res=None, swap=False):
        """coarse_res / upsample_res：覆盖 roma_outdoor 默认的 560 / 864（须是 14 的倍数）；swap：把 SAR 当第一张图、
        光学当第二张图送进模型，输出仍按（光学, SAR）的顺序。两者都只用于检验多次打标是否一致（#131），默认不变。"""
        sys.path.insert(0, str(repo))
        from romatch import roma_outdoor

        ckpt = load_ckpt(weights)
        sd = ckpt[weights_key] if weights_key else ckpt
        res = {k: v for k, v in (("coarse_res", coarse_res), ("upsample_res", upsample_res)) if v is not None}
        self.model = roma_outdoor(device=device, weights=sd, **res)
        self.swap = swap
        # 合并进 DINOv2 的各份 LoRA（按合并顺序）；finetune/models/roma.py 从本存档起步时把它们记进新存档（#238）
        self.lora_merged = merge_saved(self.model.encoder.dinov2_vitl14[0], ckpt) if weights_key else []
        self.model.eval()
        self.device, self.long_side = device, long_side
        self.resolved_params = resolved_parameters(self.model, long_side, swap)
        self.notes = (f"long_side={long_side}, uint8 PIL at 640 grid (official match() only takes PIL RGB), "
                      f"weights_key={weights_key}, sample num=default(10000)"
                      + (f", coarse_res={coarse_res}, upsample_res={upsample_res}" if res else "")
                      + (", swap (SAR as first image)" if swap else ""))

    def _pil(self, img):
        import cv2
        from PIL import Image

        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side, df=8)
        if (hn, wn) != (h, w):
            img = cv2.resize(np.ascontiguousarray(img, np.float32), (wn, hn))
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)          # ToPILImage: mul(255).byte()
        return Image.fromarray(np.repeat(u8[..., None], 3, -1), mode="RGB"), (h, w, hn, wn)

    def match(self, opt, sar):
        import torch

        p0, g0 = self._pil(opt)
        p1, g1 = self._pil(sar)
        with torch.no_grad():
            if self.swap:
                warp, cert = self.model.match(p1, p0, batched=False, device=self.device)
                m, conf = self.model.sample(warp, cert)
                k1, k0 = self.model.to_pixel_coordinates(m, g1[2], g1[3], g0[2], g0[3])
            else:
                warp, cert = self.model.match(p0, p1, batched=False, device=self.device)
                m, conf = self.model.sample(warp, cert)
                k0, k1 = self.model.to_pixel_coordinates(m, g0[2], g0[3], g1[2], g1[3])
        kp0 = to_original(k0.float().cpu().numpy() - 0.5, *g0)
        kp1 = to_original(k1.float().cpu().numpy() - 0.5, *g1)
        return kp0, kp1, conf.float().cpu().numpy()


def resolved_parameters(model, long_side, swap):
    """从实际实例记录推理参数，不把构造默认值事后当成运行事实（#267）。"""
    import torch
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "coarse_resolution": [model.h_resized, model.w_resized],
        "upsample_resolution": list(model.upsample_res),
        "output_resolution": list(model.get_output_resolution()),
        **{k: getattr(model, k) for k in ("symmetric", "upsample_preds", "attenuate_cert", "sample_mode", "sample_thresh")},
        "sample_signature": str(inspect.signature(model.sample)),
        "sample_num": inspect.signature(model.sample).parameters["num"].default,
        "match_signature": str(inspect.signature(model.match)), "match_batched": False,
        "adapter_long_side": long_side, "adapter_swap": swap, "model_training": model.training,
        "encoder_amp_dtype": str(model.encoder.amp_dtype),
        "decoder_amp_dtype": str(model.decoder.amp_dtype),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def merge_saved(dinov2, ckpt: dict):
    """把训练存档里 DINOv2 的改动装进 dinov2（romatch 把 DINOv2 放在列表里，不在 state_dict 中，只能另存）：
    先按顺序合并 'dinov2_lora_base'（从别的存档起步时，起点已合并进来的各份 LoRA，#238），再合并本次训练的
    'dinov2_lora'（#122、#124），最后按原 dtype 覆盖 'dinov2_blocks'（全参数训练的块，#124）。合并后结构与原模型相同。
    返回合并过的各份 LoRA，按合并顺序。"""
    chain = list(ckpt.get("dinov2_lora_base") or []) + ([ckpt["dinov2_lora"]] if ckpt.get("dinov2_lora") else [])
    for lora in chain:
        merge_lora(dinov2, lora)
    blocks = ckpt.get("dinov2_blocks")
    if blocks:
        bad = dinov2.load_state_dict(blocks, strict=False).unexpected_keys
        if bad:
            raise KeyError(f"dinov2_blocks 里有 DINOv2 没有的键：{bad[:5]}")
    return chain


def merge_lora(dinov2, lora: dict):
    """W ← W + scale·B A（2106.09685 §4.1），在 fp32 里算再转回 W 的 dtype。lora：模块名 → {A, B, scale}。"""
    import torch

    mods = dict(dinov2.named_modules())
    with torch.no_grad():
        for name, d in lora.items():
            lin = mods[name]
            delta = d["scale"] * (d["B"].float() @ d["A"].float())
            lin.weight.copy_((lin.weight.float() + delta.to(lin.weight.device)).to(lin.weight.dtype))
