"""无标注训练集的 patch 对加载。影像映射与 baseline 主表同口径（moonlib/inputs.py）。"""
from __future__ import annotations

import numpy as np
import torch

from baselines.data import Data
from moonlib import inputs

from . import augment


class PairSet(torch.utils.data.Dataset):
    """一个 split 的全部 patch 对（train 没有标注，所以不筛 Label）。返回网络输入网格上的 (1,H,W) 张量。"""

    def __init__(self, root, split, resize, optical="div255", sar="p2p98", limit=0, neg=False, seed=0, aug=(),
                 aug_shift=12.0, aug_rot=3.0, aug_scale=0.03):
        """neg：每对额外返回一张来自其他 ROI 的 SAR（image1_neg，负样本对，#49）。
        aug：学生侧扰动（finetune/augment.py，#53）。含 "geo" 时 SAR 施加已知随机仿射，另返回 T（原网格角点约定 2×3）；
        含 "photo" 时两侧加光度扰动。"""
        self.data, self.split, self.resize, self.neg = Data(root), split, resize, neg
        self.seed, self.aug = seed, tuple(aug)
        self.aug_kw = {"shift": aug_shift, "rot": aug_rot, "scale": aug_scale}
        self.pairs = self.data.pairs(split, labelled_only=False)
        if limit:
            self.pairs = self.pairs[:limit]
        self.map_opt, self.map_sar = inputs.get("optical", optical), inputs.get("sar", sar)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        opt = self.map_opt(self.data.optical(self.split, pair)).astype(np.float32)
        sar = self.map_sar(self.data.sar(self.split, pair)).astype(np.float32)
        extra = {}
        if self.aug:   # 每个 epoch、每个 worker 不同（worker 的 torch 种子逐 epoch 变化）
            rng = np.random.default_rng([self.seed, i, torch.initial_seed() % 2 ** 32])
            if "geo" in self.aug:
                T = augment.sample_T(rng, **self.aug_kw)
                sar = augment.warp(sar, T)
                extra["T"] = torch.from_numpy(T)
            if "photo" in self.aug:
                opt, sar = augment.photometric(opt, rng, sar=False), augment.photometric(sar, rng, sar=True)
        opt, sar = self.resize(opt), self.resize(sar)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        out = {"pair": pair, "image0": t(opt), "image1": t(sar), **extra}
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
