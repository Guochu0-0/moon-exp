"""本地数据集：枚举 pair、读检查点、读影像。

布局（Processed_Data 下）：

    <split>/ROI_xxx/{Optical,SAR}/patch_<n>.tif
    <split>/ROI_xxx/Label/patch_<n>.txt     没有该文件 = 无标注 pair

pair 标识写作 ``ROI_037/patch_1``（split 另记），与文件路径一一对应。
split 内另有一个「编号」：ROI 名升序、patch 序号按数值升序，从 0 起。它只为按编号检索（Test #10），
由数据集文件决定，含无标注 pair。
"""
from __future__ import annotations

import re
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

SPLIT_DIRS = {"val": "Val", "test": "Test"}
SAR_DISPLAY_BAND = 1  # 旧 benchmark 取第 2 通道；只影响显示


class Dataset:
    def __init__(self, root):
        root = Path(root)
        if (root / "Processed_Data").is_dir():
            root = root / "Processed_Data"
        self.root = root

    @lru_cache(maxsize=None)
    def pairs(self, split: str) -> list[str]:
        """split 内全部 pair，按编号顺序。"""
        base = self.root / SPLIT_DIRS[split]
        out = []
        for roi in sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("ROI_")):
            nums = sorted(int(m.group(1)) for f in (base / roi / "SAR").glob("patch_*.tif")
                          if (m := re.fullmatch(r"patch_(\d+)\.tif", f.name)))
            out += [f"{roi}/patch_{n}" for n in nums]
        return out

    def _path(self, split: str, pair: str, kind: str) -> Path:
        roi, patch = pair.split("/")
        ext = "txt" if kind == "Label" else "tif"
        return self.root / SPLIT_DIRS[split] / roi / kind / f"{patch}.{ext}"

    @lru_cache(maxsize=None)
    def checkpoints(self, split: str, pair: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(光学 N×2, SAR N×2)，像素坐标，y 向下。无标注返回 None。"""
        p = self._path(split, pair, "Label")
        if not p.exists():
            return None
        a = np.loadtxt(p, dtype=np.float64, ndmin=2)
        a[:, [1, 3]] *= -1  # Label 里 y 存为负值
        return a[:, :2], a[:, 2:4]

    def labelled(self, split: str) -> list[str]:
        return [p for p in self.pairs(split) if self._path(split, p, "Label").exists()]

    @lru_cache(maxsize=None)
    def numbers(self, split: str) -> dict[str, int]:
        """pair → 编号。"""
        return {p: i for i, p in enumerate(self.pairs(split))}

    @lru_cache(maxsize=4096)
    def size(self, split: str, pair: str, kind: str) -> tuple[int, int]:
        """影像宽、高（px），只读 TIFF 头。kind 为 Optical / SAR。"""
        import tifffile

        with tifffile.TiffFile(self._path(split, pair, kind)) as t:
            h, w = t.pages[0].shape[:2]
        return w, h

    def optical(self, split: str, pair: str) -> np.ndarray:
        return read_tif(self._path(split, pair, "Optical"))

    def sar(self, split: str, pair: str) -> np.ndarray:
        return read_tif(self._path(split, pair, "SAR"))


def read_tif(path) -> np.ndarray:
    """逐 strip 用标准库 zlib 解压。装了 imagecodecs 时 tifffile 走 libdeflate，在本机上会报 BAD_DATA（连它自己写的文件也是），
    所以不依赖它。"""
    import tifffile

    with tifffile.TiffFile(path) as t, open(path, "rb") as f:
        pg = t.pages[0]
        chunks = []
        for off, cnt in zip(pg.dataoffsets, pg.databytecounts):
            f.seek(off)
            raw = f.read(cnt)
            chunks.append(zlib.decompress(raw) if pg.compression == 8 else raw)
        dtype = pg.dtype.newbyteorder(t.byteorder)
        n = int(np.prod(pg.shape))
        return np.frombuffer(b"".join(chunks), dtype=dtype, count=n).reshape(pg.shape).copy()


def to_display(img: np.ndarray, band: int | None = None, lo=1.0, hi=99.0) -> np.ndarray:
    """按分位数拉伸到 uint8，只用于工作台显示；不是评价协议的一部分。"""
    if img.ndim == 3:
        img = img[..., SAR_DISPLAY_BAND if band is None else band]
    x = img.astype(np.float32)
    a, b = np.percentile(x, [lo, hi])
    return (np.clip((x - a) / max(b - a, 1e-12), 0, 1) * 255).astype(np.uint8)
