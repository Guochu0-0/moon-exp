"""AnyMatch-RoMa 的训练态包装（「【RoMa】微调代码接入与实测」#64；方案见 research/roma-finetune 分支的 README）。

- 构造与权重加载复用 baselines 的 RomatchAdapter，训练的网络就是评测的网络。DINOv2 不是参数、始终冻结；
  VGG（encoder.cnn）默认冻结（train_vgg 打开），只训 decoder。前向要 model.train() 才输出 gm_cls，但全部 BN 保持 eval（bs 1–2）。
- 输入：与适配器同口径（原网格 → cv2 双线性长边 640 → uint8 → PIL bicubic 560 → /255 → ImageNet 归一化），
  灰度复制成 3 通道。训练只跑 560 一遍（同 RoMa 原训练），评测照旧 560 → 864、symmetric。
- decoder 内部按 romatch 的 amp_dtype（fp16）autocast，训练用 GradScaler。

坐标约定：RoMa 归一化坐标 x_n ∈ (−1, 1)，像素中心 linspace(−1+1/n, 1−1/n)；原网格角点约定 u = 256·(x_n + 1)。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from baselines.adapters.base import long_side_size

from ..geom import HW

REPO = Path(__file__).resolve().parents[2]
PARAMS = {"config": "configs/baselines/anymatch_roma__minmax.json",
          "init": "",            # 起点 ckpt（{'model': ...}）；空 = 推理配置里的底座权重
          "train_vgg": False,    # 解冻 VGG（encoder.cnn）
          "train_decoder": True, # 训 decoder；false 时只训 VGG（#120）
          "vgg_lr": 1.0,         # VGG 的学习率 = 这一倍数 × [optim] lr（#120；RoMa 原训练为 1/20）
          "vgg_dropout": 0.0}    # 训练前向时对 VGG 各尺度特征做 channel dropout 的概率（#120，UniMatch V2 式特征扰动）
OPTIM = {"wd": 0.01}             # romatch 原配置
RUN = {"save_zero": False}
TORCH_HOME = "/remote-home/xufang/YGC/weights/torch_home"   # DINOv2 缓存
RES = 560
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
        self.params = dec + vgg
        self.groups = [g for g in ({"params": dec, "lr_scale": 1.0}, {"params": vgg, "lr_scale": m["vgg_lr"]})
                       if g["params"]]
        if m["vgg_dropout"] > 0:   # 只在训练用的这个包装里挂钩子，存下的权重和评测链路不受影响
            p_drop = m["vgg_dropout"]
            self.model.encoder.cnn.register_forward_hook(
                lambda mod, inp, out: {k: torch.nn.functional.dropout2d(v, p_drop, training=True) for k, v in out.items()})

    def resize(self, img: np.ndarray) -> np.ndarray:
        """原网格 float [0,1] → 560² float [0,1]，逐步照适配器 + romatch match() 的预处理（含 uint8 量化）。"""
        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side, df=8)
        if (hn, wn) != (h, w):
            img = cv2.resize(np.ascontiguousarray(img, np.float32), (wn, hn))
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(u8).resize((RES, RES), Image.BICUBIC)
        return np.asarray(pil, np.float32) / 255.0

    def forward(self, image0, image1):
        """image0/1: (B,1,560,560) float [0,1]。返回 romatch 的 corresps（训练态：含 gm_cls、gm_certainty）。"""
        norm = lambda x: (x.expand(-1, 3, -1, -1) - MEAN.to(x)) / STD.to(x)
        return self.model({"im_A": norm(image0), "im_B": norm(image1)}, batched=True)

    def state_dict(self):
        """{'model': ...}，RomatchAdapter(weights_key='model') 直接能读。"""
        return {"model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}
