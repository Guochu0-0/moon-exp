"""无标注训练集的 patch 对加载。影像映射与 baseline 主表同口径（moonlib/inputs.py）。"""
from __future__ import annotations

import numpy as np
import torch

from baselines.data import Data
from moonlib import inputs

from . import augment


class PairSet(torch.utils.data.Dataset):
    """一个 split 的全部 patch 对（train 没有标注，所以不筛 Label）。返回网络输入网格上的 (1,H,W) 张量。
    split 可用 + 连接多个（如 "val+test"，过拟合上限参考 #96），pair 标识在各 split 之间不能重名。"""

    def __init__(self, root, split, resize, optical="div255", sar="p2p98", limit=0, neg=False, seed=0, aug=(),
                 aug_shift=12.0, aug_rot=3.0, aug_scale=0.03, aug_erase=(3, 0.10, 0.25), resize_hi=None):
        """neg：每对额外返回一张来自其他 ROI 的 SAR（image1_neg，负样本对，#49）。
        aug：学生侧扰动（finetune/augment.py，#53）。含 "geo" 时 SAR 施加已知随机仿射，另返回 T（原网格角点约定 2×3）；
        含 "photo" 时两侧加光度扰动；含 "erase" 时 SAR 随机擦除（aug_erase = 块数上限、面积下限、上限），
        另返回 occ（原网格 (1,512,512) uint8，被擦为 1）。
        resize_hi：给出时另返回 image0_hi / image1_hi（推理上采样那一遍的输入，#122），与 image0/1 同一份扰动后的影像。"""
        self.resize_hi = resize_hi
        self.data, self.split, self.resize, self.neg = Data(root), split, resize, neg
        self.seed, self.aug = seed, tuple(aug)
        self.aug_kw = {"shift": aug_shift, "rot": aug_rot, "scale": aug_scale}
        self.erase_kw = dict(zip(("n_max", "a_min", "a_max"), aug_erase))
        self.src = {}                                    # pair → 所在 split
        for sp in split.split("+"):
            for q in self.data.pairs(sp, labelled_only=False):
                if q in self.src:
                    raise ValueError(f"{q} 同时出现在 {self.src[q]} 和 {sp}")
                self.src[q] = sp
        self.pairs = list(self.src)
        if limit:
            self.pairs = self.pairs[:limit]
        self.map_opt, self.map_sar = inputs.get("optical", optical), inputs.get("sar", sar)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        opt = self.map_opt(self.data.optical(self.src[pair], pair)).astype(np.float32)
        sar = self.map_sar(self.data.sar(self.src[pair], pair)).astype(np.float32)
        extra = {}
        if self.aug:   # 每个 epoch、每个 worker 不同（worker 的 torch 种子逐 epoch 变化）
            rng = np.random.default_rng([self.seed, i, torch.initial_seed() % 2 ** 32])
            if "geo" in self.aug:
                T = augment.sample_T(rng, **self.aug_kw)
                sar = augment.warp(sar, T)
                extra["T"] = torch.from_numpy(T)
            if "photo" in self.aug:
                opt, sar = augment.photometric(opt, rng, sar=False), augment.photometric(sar, rng, sar=True)
            if "erase" in self.aug:
                sar, occ = augment.erase(sar, rng, **self.erase_kw)
                extra["occ"] = torch.from_numpy(occ)[None]
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        if self.resize_hi is not None:
            extra["image0_hi"], extra["image1_hi"] = t(self.resize_hi(opt)), t(self.resize_hi(sar))
        opt, sar = self.resize(opt), self.resize(sar)
        out = {"pair": pair, "image0": t(opt), "image1": t(sar), **extra}
        if self.neg:
            other = neg_partner(self.pairs, pair, np.random.default_rng([self.seed, i]))   # 按索引播种，与 worker 无关
            out["pair_neg"] = other
            out["image1_neg"] = t(self.resize(self.map_sar(self.data.sar(self.src[other], other))))
        return out


def neg_partner(pairs, pair, rng):
    """随机取一个与 pair 不同 ROI 的 pair（负样本对的 SAR 来源）。"""
    roi = pair.split("/")[0]
    while True:
        other = pairs[int(rng.integers(len(pairs)))]
        if other.split("/")[0] != roi:
            return other
