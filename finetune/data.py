"""无标注训练集的 patch 对加载。影像映射与 baseline 主表同口径（baselines/inputs.py）。"""
from __future__ import annotations

import numpy as np
import torch

from baselines import inputs
from baselines.data import Data


class PairSet(torch.utils.data.Dataset):
    """一个 split 的全部 patch 对（train 没有标注，所以不筛 Label）。返回网络输入网格上的 (1,H,W) 张量。"""

    def __init__(self, root, split, resize, optical="div255", sar="p2p98", limit=0, neg=False, seed=0):
        """neg：每对额外返回一张来自其他 ROI 的 SAR（image1_neg，负样本对，#49）。"""
        self.data, self.split, self.resize, self.neg = Data(root), split, resize, neg
        self.seed = seed
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
        out = {"pair": pair, "image0": t(opt), "image1": t(sar)}
        if self.neg:
            other = neg_partner(self.pairs, pair, np.random.default_rng([self.seed, i]))   # 按索引播种，与 worker 无关
            out["pair_neg"] = other
            out["image1_neg"] = t(self.resize(self.map_sar(self.data.sar(self.split, other))))
        return out


def neg_partner(pairs, pair, rng):
    """随机取一个与 pair 不同 ROI 的 pair（负样本对的 SAR 来源）。"""
    roi = pair.split("/")[0]
    while True:
        other = pairs[int(rng.integers(len(pairs)))]
        if other.split("/")[0] != roi:
            return other
