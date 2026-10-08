"""核对 DINOv2 LoRA / 全参数块的存档 → 适配器加载（#124）。把训练包装里的 LoRA B 和全参数块随机扰动后存档，
用 RomatchAdapter 加载，做两项比较：

1. 权重（精确）：按 merge_lora 的算法由训练包装的参数算出期望的 fp16 DINOv2 权重（LoRA 层 W + scale·B A 在 fp32 里算、
   转回 fp16；全参数块转 fp16），与加载后的 DINOv2 逐张量比较。应在 1 个 fp16 舍入单位内一致，且没有缺键。
2. 特征：训练包装与加载后各算一次 DINOv2 特征的相对差；另给不加扰动时训练包装与原权重的相对差，作为 fp16 误差的底线。

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
from finetune.models.roma import LoRA, Model  # noqa: E402


def feats(dv, x):
    with torch.no_grad():
        return dv.forward_features(x.half())["x_norm_patchtokens"].float()


def loaded_dino(m, weights, key):
    ad = RomatchAdapter(REPO / m.infer_cfg["repo"], weights, weights_key=key)
    return ad.model.encoder.dinov2_vitl14[0].to("cuda").half()


def expected(dv) -> dict:
    """训练包装的 DINOv2 → 合并后应有的 fp16 state_dict（键名与原模型相同）。"""
    out = {}
    lora = {n: mod for n, mod in dv.named_modules() if isinstance(mod, LoRA)}
    for k, v in dv.state_dict().items():
        if any(k.startswith(n + ".") for n in lora) and not k.split(".")[-1] in ("weight", "bias"):
            continue                                     # LoRA 的 A、B
        out[k.replace(".base.", ".")] = v.detach().half()
    with torch.no_grad():
        for n, l in lora.items():
            delta = l.scale * (l.B.float() @ l.A.float())
            out[n + ".weight"] = (l.base.weight.float() + delta).half()
    return out


torch.manual_seed(0)
x = torch.randn(1, 3, 560, 560, device="cuda")
for path in sys.argv[1:]:
    cfg = load(path)
    m = Model(cfg, os.environ["MOON_WEIGHTS"])
    dv = m.model.encoder.dinov2_vitl14[0]
    base_key = m.infer_cfg.get("params", {}).get("weights_key")
    floor = feats(dv, x)
    base = feats(loaded_dino(m, m.weights, base_key), x)
    floor_rel = ((floor - base).norm() / base.norm()).item()
    with torch.no_grad():
        for l in m.lora.values():
            l.B.normal_(0, 0.02)
        for blk in m.dino_blocks.values():
            for p in blk.parameters():
                p.add_(torch.randn_like(p) * 0.01 * p.abs().mean())
    a = feats(dv, x)
    exp = expected(dv)
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.pt"
        torch.save(m.state_dict(), ck)
        dv2 = loaded_dino(m, str(ck), "model")
        b = feats(dv2, x)
    got = {k: v.detach() for k, v in dv2.state_dict().items()}
    missing = sorted(set(exp) ^ set(got))
    bad = []
    for k in sorted(set(exp) & set(got)):
        e, g = exp[k].float(), got[k].float()
        ulp = (e.abs().clamp_min(6e-5) * 2 ** -10)      # fp16 的 1 个舍入单位（相对 2^-10）
        if ((e - g).abs() > ulp * 1.01).any():
            bad.append((k, ((e - g).abs() / ulp).max().item()))
    rel = ((a - b).norm() / a.norm()).item()
    moved = ((a - base).norm() / a.norm()).item()
    print(f"{Path(path).name}: LoRA {len(m.lora)} 层，全参数块 {len(m.dino_blocks)} 个；"
          f"权重 {len(exp)} 个张量，键不一致 {len(missing)} 个，超 1 ulp 的 {len(bad)} 个 {bad[:3]}；"
          f"特征相对差 扰动后 训练包装 vs 加载 {rel:.2e}，不扰动 训练包装 vs 原权重 {floor_rel:.2e}，"
          f"扰动造成的变化 {moved:.2e}", flush=True)
    del m, dv, dv2
    torch.cuda.empty_cache()
