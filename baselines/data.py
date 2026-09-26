"""在 matcher 环境里枚举 pair、读影像。与 workbench/dataset.py 同口径（pair 标识、顺序），但兼容 py3.8、不依赖 tifffile。"""
from __future__ import annotations

import re
from pathlib import Path

from .tif import read_tif

SPLIT_DIRS = {"val": "Val", "test": "Test"}


class Data:
    def __init__(self, root):
        root = Path(root)
        if (root / "Processed_Data").is_dir():
            root = root / "Processed_Data"
        self.root = root

    def pairs(self, split: str, labelled_only: bool = True) -> list:
        base = self.root / SPLIT_DIRS[split]
        out = []
        for roi in sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("ROI_")):
            nums = []
            for f in (base / roi / "SAR").glob("patch_*.tif"):
                m = re.fullmatch(r"patch_(\d+)\.tif", f.name)
                if m:
                    nums.append(int(m.group(1)))
            for n in sorted(nums):
                pair = f"{roi}/patch_{n}"
                if not labelled_only or self.path(split, pair, "Label").exists():
                    out.append(pair)
        return out

    def path(self, split: str, pair: str, kind: str) -> Path:
        roi, patch = pair.split("/")
        ext = "txt" if kind == "Label" else "tif"
        return self.root / SPLIT_DIRS[split] / roi / kind / f"{patch}.{ext}"

    def optical(self, split: str, pair: str):
        return read_tif(self.path(split, pair, "Optical"))

    def sar(self, split: str, pair: str):
        return read_tif(self.path(split, pair, "SAR"))
