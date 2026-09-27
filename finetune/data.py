"""无标注训练集的 patch 对加载。影像映射与 baseline 主表同口径（baselines/inputs.py）。"""
from __future__ import annotations

import numpy as np
import torch

from baselines import inputs
from baselines.data import Data


class PairSet(torch.utils.data.Dataset):
    """一个 split 的全部 patch 对（train 没有标注，所以不筛 Label）。返回网络输入网格上的 (1,H,W) 张量。"""

    def __init__(self, root, split, resize, optical="div255", sar="p2p98", limit=0):
        self.data, self.split, self.resize = Data(root), split, resize
        self.pairs = self.data.pairs(split, labelled_only=False)
        if limit:
            self.pairs = self.pairs[:limit]
        self.map_opt, self.map_sar = inputs.get("optical", optical), inputs.get("sar", sar)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        opt = self.resize(self.map_opt(self.data.optical(self.split, pair)))
        sar = self.resize(self.map_sar(self.data.sar(self.split, pair)))
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        return {"pair": pair, "image0": t(opt), "image1": t(sar)}
