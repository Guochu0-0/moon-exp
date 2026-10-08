"""核对 DINOv2 LoRA / 全参数块的存档 → 适配器加载（#124）：把训练包装里的 LoRA B 和全参数块随机扰动，
算一次 DINOv2 特征；存档后用 RomatchAdapter 加载再算一次，两者应只差 fp16 舍入。

    MOON_WEIGHTS=… TORCH_HOME=… python runs/T14/code/check_merge.py runs/T14/configs/main__all.toml …
"""
import os
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from baselines.adapters.romatch import RomatchAdapter  # noqa: E402
from finetune.config import load  # noqa: E402
from finetune.models.roma import Model  # noqa: E402


def feats(adapter_or_dv, x):
    dv = adapter_or_dv
    with torch.no_grad():
        return dv.forward_features(x.half())["x_norm_patchtokens"].float()


def loaded_dino(m, weights, key):
    ad = RomatchAdapter(REPO / m.infer_cfg["repo"], weights, weights_key=key)
    return ad.model.encoder.dinov2_vitl14[0].to("cuda").half()


torch.manual_seed(0)
x = torch.randn(1, 3, 560, 560, device="cuda")
for path in sys.argv[1:]:
    cfg = load(path)
    m = Model(cfg, os.environ["MOON_WEIGHTS"])
    with torch.no_grad():
        for l in m.lora.values():
            l.B.normal_(0, 0.02)
        for blk in m.dino_blocks.values():
            for p in blk.parameters():
                p.add_(torch.randn_like(p) * 0.01 * p.abs().mean())
    a = feats(m.model.encoder.dinov2_vitl14[0], x)
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.pt"
        torch.save(m.state_dict(), ck)
        b = feats(loaded_dino(m, str(ck), "model"), x)
    base = feats(loaded_dino(m, m.weights, m.infer_cfg.get("params", {}).get("weights_key")), x)
    rel = ((a - b).norm() / a.norm()).item()
    moved = ((a - base).norm() / a.norm()).item()
    print(f"{Path(path).name}: LoRA {len(m.lora)} 层，全参数块 {len(m.dino_blocks)} 个；"
          f"训练包装 vs 存档加载后 相对差 {rel:.2e}，扰动造成的特征变化 {moved:.2e}", flush=True)
    del m
    torch.cuda.empty_cache()
