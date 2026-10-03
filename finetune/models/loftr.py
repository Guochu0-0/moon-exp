"""AnyMatch-LoFTR 的训练态包装。

模型构造、权重加载、分辨率全部复用 baselines.adapters.loftr.LoFTRAdapter，保证训练的网络与 baseline 推理的
是同一个东西（配置见 configs/baselines/anymatch_loftr.json）。

训练时模块**保持 eval()**，只打开梯度，原因有两个（对照上游 LoFTR df7ca80）：
- `CoarseMatching.get_coarse_match` 在 `self.training` 为真时会用 GT 粗匹配（`spv_b_ids` 等）补齐训练样本
  （`coarse_matching.py:200-236`），我们没有 GT，走这条路会直接报 KeyError；
- backbone 里是 BatchNorm，batch 只有 1–2 对，训练态会用很偏的 batch 统计量并改写 running stats。
这样前向与推理完全相同，梯度照常流过 `conf_matrix`（粗级 dual-softmax）和 `expec_f`（细级 soft-argmax 与 std）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from baselines.adapters.base import long_side_size

from ..geom import HW

REPO = Path(__file__).resolve().parents[2]
PARAMS = {"config": "configs/baselines/anymatch_loftr.json",
          "init": ""}        # 起点 ckpt；空 = 推理配置里的底座权重
OPTIM = {}
RUN = {}
TORCH_HOME = "/opt/torch_home"


class Model:
    amp = False

    def __init__(self, cfg: dict, weights_root: str, device="cuda"):
        from baselines.adapters.loftr import LoFTRAdapter

        m = cfg["model"]
        self.infer_cfg = json.loads((REPO / m["config"]).read_text(encoding="utf-8"))
        self.weights = m["init"] or str(Path(weights_root) / self.infer_cfg["weights"])
        prm = self.infer_cfg.get("params", {})
        self.adapter = LoFTRAdapter(REPO / self.infer_cfg["repo"], self.weights, device=device,
                                    long_side=prm.get("long_side", 640), temp_bug_fix=prm.get("temp_bug_fix", True),
                                    coarse_thr=prm.get("coarse_thr"))
        self.model, self.device, self.long_side = self.adapter.model, device, prm.get("long_side", 640)
        self.input = self.infer_cfg["input"]
        self.s = HW / self.long_side   # 原网格 / 输入网格（正方形 patch）
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def resize(self, img: np.ndarray) -> np.ndarray:
        """原网格 float32 → 网络输入网格（与适配器 _tensor 相同的 resize）。"""
        h, w = img.shape
        hn, wn = long_side_size(h, w, self.long_side, df=8)
        return img if (hn, wn) == (h, w) else cv2.resize(img, (wn, hn))

    def forward(self, image0, image1) -> dict:
        """image0/1: (B,1,H,W) float [0,1]，已在设备上。返回 LoFTR 的 data 字典，带梯度的键：
        conf_matrix (B, L, S)、expec_f (M, 3) = 细级 soft-argmax 的归一化坐标 + std。
        b_ids/i_ids/j_ids 是推理同款的粗匹配（阈值 + MNN），mkpts0_f/mkpts1_f 在输入网格上。"""
        data = {"image0": image0, "image1": image1}
        self.model(data)
        return data

    def state_dict(self):
        """存成 baselines 适配器能直接读的格式：{'state_dict': ...}。"""
        return {"state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}
