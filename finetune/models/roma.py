"""AnyMatch-RoMa 的训练态包装（「【RoMa】微调代码接入与实测」#64；方案见 research/roma-finetune 分支的 README）。

- 构造与权重加载复用 baselines 的 RomatchAdapter，训练的网络就是评测的网络。DINOv2 不是参数、默认冻结
  （lora_r > 0 时在 qkv 或全部线性层上加 LoRA，#122、#124；dino_unfreeze > 0 时最后几块全参数训练，#124）；VGG（encoder.cnn）默认冻结（train_vgg 打开），只训 decoder。
  前向要 model.train() 才输出 gm_cls，但全部 BN 保持 eval（bs 1–2）。
- 输入：与适配器同口径（原网格 → cv2 双线性长边 640 → uint8 → PIL bicubic 560 → /255 → ImageNet 归一化），
  灰度复制成 3 通道。默认只训 560 一遍（同 RoMa 原训练），评测照旧 560 → 864、symmetric；
  train_res 可改为训推理时的 864 上采样那一遍（#122）。
- decoder 内部按 romatch 的 amp_dtype（fp16）autocast，训练用 GradScaler。

坐标约定：RoMa 归一化坐标 x_n ∈ (−1, 1)，像素中心 linspace(−1+1/n, 1−1/n)；原网格角点约定 u = 256·(x_n + 1)。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image

from baselines.adapters.base import long_side_size

from ..geom import HW

REPO = Path(__file__).resolve().parents[2]
PARAMS = {"config": "configs/baselines/anymatch_roma__minmax.json",
          "init": "",            # 起点 ckpt（{'model': ...}）；空 = 推理配置里的底座权重
          "train_vgg": False,    # 解冻 VGG（encoder.cnn）
          "train_decoder": True, # 训 decoder；false 时只训 VGG（#120）
          "vgg_lr": 1.0,         # VGG 的学习率 = 这一倍数 × [optim] lr（#120；RoMa 原训练为 1/20）
          "vgg_dropout": 0.0,    # 训练前向时对 VGG 各尺度特征做 channel dropout 的概率（#120，UniMatch V2 式特征扰动）
          "train_res": "560",    # 训哪一遍（#122）：560 只训 560 一遍（原训练）/ 864 560 一遍 no_grad、只训推理时的
                                 # 864 上采样那一遍 / both 两遍都训、损失相加
          "lora_r": 0,           # DINOv2 每块 attn.qkv 上的 LoRA 秩（#122，2106.09685）；0 = DINOv2 冻结
          "lora_alpha": 16.0,    # LoRA 缩放 = alpha / r
          "lora_lr": 1e-4,       # LoRA 参数的学习率（绝对值，不随 [optim] lr 缩放）
          "lora_targets": "qkv", # LoRA 挂在哪些层（#124）：qkv 只挂 attn.qkv / all 每块全部线性层（qkv、attn.proj、mlp.fc1、mlp.fc2）
          "dino_unfreeze": 0,    # DINOv2 最后几块全参数训练（#124），lr 同 [optim] lr；这几块不挂 LoRA
          "dino_ckpt": False,    # DINOv2 各块用梯度检查点（省显存，lora_r > 0 时才有意义）
          "unc_head": False}     # 只在训练时用的不确定度头（#127）：每个细化层输出卷积的输入上接 1×1 卷积，出 2 通道
                                 # （混合权重 logit、β），放进 corresps[s]["unc"]；不进 'model' 权重，推理结构不变
OPTIM = {"wd": 0.01}             # romatch 原配置
RUN = {"save_zero": False}
TORCH_HOME = "/remote-home/xufang/YGC/weights/torch_home"   # DINOv2 缓存
RES = 560
RES_HI = 864                     # 推理的上采样分辨率（romatch roma_outdoor 的 upsample_res）
TRAIN_RES = ("560", "864", "both")
LORA_TARGETS = {"qkv": ("attn.qkv",), "all": ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _norm(x):
    return (x.expand(-1, 3, -1, -1) - MEAN.to(x)) / STD.to(x)


class LoRA(torch.nn.Module):
    """y = W x + b + (α/r)·B A x（2106.09685 §4.1）。A 按 kaiming 均匀初始化、B 为 0，起点与原层相同；
    A、B 用 fp32 存，参与 fp16 的前向时临时转型。合并见 baselines.adapters.romatch.merge_lora。"""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale


class Model:
    amp = True

    def __init__(self, cfg: dict, weights_root: str, device="cuda"):
        from baselines.adapters.romatch import RomatchAdapter

        m = cfg["model"]
        self.infer_cfg = json.loads((REPO / m["config"]).read_text(encoding="utf-8"))
        self.weights = m["init"] or str(Path(weights_root) / self.infer_cfg["weights"])
        prm = dict(self.infer_cfg.get("params", {}))
        if m["init"]:
            prm["weights_key"] = "model"
        self.long_side = prm.get("long_side", 640)
        self.adapter = RomatchAdapter(REPO / self.infer_cfg["repo"], self.weights, device=device,
                                      weights_key=prm.get("weights_key"), long_side=self.long_side)
        self.model, self.device = self.adapter.model, device
        self.input = self.infer_cfg["input"]
        self.s = HW / RES
        self.train_res = m["train_res"]
        if self.train_res not in TRAIN_RES:
            raise ValueError(f"[model] train_res 可选 {TRAIN_RES}，得到 {self.train_res!r}")
        self.resize_hi = (lambda img: self.resize(img, RES_HI)) if self.train_res != "560" else None
        self.model.train()
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()
        dec, vgg = [], []
        for name, p in self.model.named_parameters():
            on_dec = m["train_decoder"] and name.startswith("decoder.")
            on_vgg = m["train_vgg"] and name.startswith("encoder.cnn")
            p.requires_grad_(on_dec or on_vgg)
            (dec if on_dec else vgg if on_vgg else []).append(p)
        self.lora, self.dino_blocks = ({}, {}) if m["lora_r"] <= 0 and m["dino_unfreeze"] <= 0 else self._open_dino(m)
        lora = [p for l in self.lora.values() for p in (l.A, l.B)]
        dino = [p for b in self.dino_blocks.values() for p in b.parameters()]
        self.unc_heads, self._unc = None, {}
        if m["unc_head"]:
            self.unc_heads = torch.nn.ModuleDict()
            for k, ref in self.model.decoder.conv_refiner.items():
                head = torch.nn.Conv2d(ref.out_conv.in_channels, 2, 1).to(device)
                torch.nn.init.zeros_(head.weight)
                torch.nn.init.zeros_(head.bias)      # 起点：两分量各半、β = 0（两个都是 1 px 的拉普拉斯）
                self.unc_heads[k] = head
                ref.out_conv.register_forward_hook(
                    lambda mod, inp, out, k=k: self._unc.__setitem__(int(k), self.unc_heads[k](inp[0])))
            dec = dec + list(self.unc_heads.parameters())
        self.params = dec + vgg + lora + dino
        self.groups = [g for g in ({"params": dec, "lr_scale": 1.0}, {"params": vgg, "lr_scale": m["vgg_lr"]},
                                   {"params": lora, "lr_scale": m["lora_lr"] / cfg["optim"]["lr"]},
                                   {"params": dino, "lr_scale": 1.0})
                       if g["params"]]
        if m["vgg_dropout"] > 0:   # 只在训练用的这个包装里挂钩子，存下的权重和评测链路不受影响
            p_drop = m["vgg_dropout"]
            self.model.encoder.cnn.register_forward_hook(
                lambda mod, inp, out: {k: torch.nn.functional.dropout2d(v, p_drop, training=True) for k, v in out.items()})

    def _open_dino(self, m) -> tuple[dict, dict]:
        """让 DINOv2 参与训练，并替换 encoder.forward：原实现在 no_grad 里算 DINOv2，这里放开梯度。
        DINOv2 先按原实现搬到 GPU、转 fp16。前面各块在 lora_targets 指定的层上挂 LoRA（之后不再整体 .to()，
        A、B 保持 fp32）；最后 dino_unfreeze 块转成 fp32 主权重、全部参数可训，前向在 fp16 autocast 里算、
        输出转回 fp16（与其余块衔接）。返回 (LoRA 模块名 → LoRA, 块名 → 全参数训练的块)。"""
        enc = self.model.encoder
        dv = enc.dinov2_vitl14[0].to(self.device).to(enc.amp_dtype)
        enc.dinov2_vitl14[0] = dv
        dv.requires_grad_(False)   # 原权重不求梯度（#124 起；之前它们也不在优化器里，只是白算一份梯度）
        if m["lora_targets"] not in LORA_TARGETS:
            raise ValueError(f"[model] lora_targets 可选 {tuple(LORA_TARGETS)}，得到 {m['lora_targets']!r}")
        n_full = m["dino_unfreeze"]
        if not 0 <= n_full <= len(dv.blocks):
            raise ValueError(f"[model] dino_unfreeze 应在 0…{len(dv.blocks)}，得到 {n_full}")
        lora, full = {}, {}
        for i, blk in enumerate(dv.blocks):
            if i >= len(dv.blocks) - n_full:
                blk.float().requires_grad_(True)
                full[f"blocks.{i}"] = blk
                f = blk.forward

                def fwd(x, f=f, dt=enc.amp_dtype):
                    with torch.autocast("cuda", dtype=dt):
                        return f(x).to(dt)
                blk.forward = fwd
            elif m["lora_r"] > 0:
                for name in LORA_TARGETS[m["lora_targets"]]:
                    parent, attr = blk.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
                    setattr(parent, attr, LoRA(getattr(parent, attr), m["lora_r"], m["lora_alpha"]))
                    lora[f"blocks.{i}.{name}"] = getattr(parent, attr)
            if m["dino_ckpt"]:
                f = blk.forward
                blk.forward = lambda x, f=f: torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)

        def forward(x, upsample=False):     # 同 romatch CNNandDinov2.forward，去掉 no_grad
            B, C, H, W = x.shape
            fp = enc.cnn(x)
            if not upsample:
                f = dv.forward_features(x.to(enc.amp_dtype))["x_norm_patchtokens"]
                fp[16] = f.permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
            return fp

        enc.forward = forward
        return lora, full

    def resize(self, img: np.ndarray, res: int = RES) -> np.ndarray:
        """原网格 float [0,1] → res² float [0,1]，逐步照适配器 + romatch match() 的预处理（含 uint8 量化）。
        res = 864 即推理上采样那一遍的输入（match() 从同一张 640 网格的 PIL 图 bicubic 缩放）。"""
        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side, df=8)
        if (hn, wn) != (h, w):
            img = cv2.resize(np.ascontiguousarray(img, np.float32), (wn, hn))
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(u8).resize((res, res), Image.BICUBIC)
        return np.asarray(pil, np.float32) / 255.0

    def forward(self, image0, image1):
        """image0/1: (B,1,560,560) float [0,1]。返回 romatch 的 corresps（训练态：含 gm_cls、gm_certainty）。"""
        return self._attach(self.model({"im_A": _norm(image0), "im_B": _norm(image1)}, batched=True))

    def _attach(self, corresps):
        """unc_head 打开时把这次前向各尺度的不确定度输出放进 corresps[s]["unc"]。"""
        if self.unc_heads is not None:
            for s, u in self._unc.items():
                if s in corresps:
                    corresps[s]["unc"] = u
            self._unc = {}
        return corresps

    def forward_hi(self, image0, image1, corresps):
        """推理上采样那一遍（#122）：image0/1 (B,1,864,864)；corresps = 560 一遍的输出。照 romatch match()：
        以 560 一遍尺度 1 的 flow、certainty 为初值，只跑 VGG 与尺度 8…1 的细化，scale_factor = 864/560。
        初值 detach（推理时这一遍本来就不回传到 560 那一遍）。"""
        c1 = corresps[1]
        init = {"flow": c1["flow"].detach(), "certainty": c1["certainty"].detach()}
        self._unc = {}
        return self._attach(self.model({"im_A": _norm(image0), "im_B": _norm(image1), "corresps": init},
                                       batched=True, upsample=True, scale_factor=RES_HI / RES))

    def state_dict(self):
        """{'model': ...}，RomatchAdapter(weights_key='model') 直接能读；有 LoRA 时另存 'dinov2_lora'
        （每层 A、B、scale），适配器加载时合并进 DINOv2 对应层的权重；有全参数训练的块时另存 'dinov2_blocks'
        （块内参数，fp32），适配器加载时覆盖进 DINOv2。推理结构不变。"""
        sd = {"model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}
        if self.lora:
            sd["dinov2_lora"] = {k: {"A": l.A.detach().cpu(), "B": l.B.detach().cpu(), "scale": l.scale}
                                 for k, l in self.lora.items()}
        if self.unc_heads is not None:   # 只作记录，评测适配器不读
            sd["unc_head"] = {k: v.detach().cpu() for k, v in self.unc_heads.state_dict().items()}
        if self.dino_blocks:
            sd["dinov2_blocks"] = {f"{b}.{k}": v.detach().cpu() for b, blk in self.dino_blocks.items()
                                   for k, v in blk.state_dict().items()}
        return sd
