"""只用 numpy + zlib 的最小 TIFF 读取器。

baseline 各自的环境（旧 torch、py3.8）里不一定有 tifffile，装了 imagecodecs 的 tifffile 读本数据集的 SAR
还会报 LIBDEFLATE_BAD_DATA（见 workbench/dataset.py），所以这里直接解析 IFD、逐 strip/tile 用 zlib 解压。
只覆盖本数据集用到的情形：经典 TIFF（非 BigTIFF）、无压缩或 deflate、predictor=1、单页。
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

_TYPES = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i", 10: "ii", 11: "f", 12: "d"}
_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_SAMPLE_KIND = {1: "u", 2: "i", 3: "f"}


def _tags(buf: bytes) -> tuple[str, dict]:
    bo = {b"II": "<", b"MM": ">"}.get(buf[:2])
    if bo is None:
        raise ValueError("不是 TIFF")
    magic, ifd = struct.unpack(bo + "HI", buf[2:8])
    if magic != 42:
        raise ValueError(f"不支持的 TIFF 变体（magic={magic}，BigTIFF？）")
    (n,) = struct.unpack(bo + "H", buf[ifd:ifd + 2])
    tags = {}
    for i in range(n):
        e = ifd + 2 + 12 * i
        tag, typ, count = struct.unpack(bo + "HHI", buf[e:e + 8])
        if typ not in _TYPES:
            continue
        size = _SIZES[typ] * count
        data = buf[e + 8:e + 12] if size <= 4 else buf[struct.unpack(bo + "I", buf[e + 8:e + 12])[0]:][:size]
        if typ == 2:
            tags[tag] = data[:count].rstrip(b"\0").decode("latin-1")
            continue
        fmt = _TYPES[typ]
        vals = struct.unpack(bo + fmt * count, data[:size])
        if typ in (5, 10):
            vals = tuple(vals[k] / vals[k + 1] if vals[k + 1] else 0.0 for k in range(0, len(vals), 2))
        tags[tag] = vals
    return bo, tags


def read_tif(path) -> np.ndarray:
    """返回 H×W 或 H×W×C（按样本交错排列）的数组，dtype 按文件。"""
    with open(path, "rb") as f:
        buf = f.read()
    bo, t = _tags(buf)
    w, h = t[256][0], t[257][0]
    spp = t.get(277, (1,))[0]
    bits = t.get(258, (1,))
    kind = _SAMPLE_KIND[t.get(339, (1,))[0]]
    comp = t.get(259, (1,))[0]
    planar = t.get(284, (1,))[0]
    if len(set(bits)) != 1:
        raise NotImplementedError(f"{path}: 各样本位深不同 {bits}")
    if t.get(317, (1,))[0] != 1:
        raise NotImplementedError(f"{path}: predictor={t[317][0]}")
    if comp not in (1, 8, 32946):
        raise NotImplementedError(f"{path}: compression={comp}")
    dtype = np.dtype(f"{bo}{kind}{bits[0] // 8}")

    def chunk(off, cnt):
        raw = buf[off:off + cnt]
        return zlib.decompress(raw) if comp != 1 else raw

    nplanes = spp if planar == 2 else 1
    cps = 1 if planar == 2 else spp          # 每个 chunk 里交错的样本数
    if 324 in t:                              # tiled
        tw, th = t[322][0], t[323][0]
        offs, cnts = t[324], t[325]
        ntx, nty = -(-w // tw), -(-h // th)
        planes = []
        for p in range(nplanes):
            img = np.empty((nty * th, ntx * tw, cps), dtype)
            for k in range(ntx * nty):
                i = p * ntx * nty + k
                tile = np.frombuffer(chunk(offs[i], cnts[i]), dtype, count=th * tw * cps).reshape(th, tw, cps)
                ty, tx = divmod(k, ntx)
                img[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw] = tile
            planes.append(img[:h, :w])
    else:                                     # stripped
        rps = t.get(278, (h,))[0]
        offs, cnts = t[273], t[279]
        nstrip = -(-h // rps)
        planes = []
        for p in range(nplanes):
            data = b"".join(chunk(offs[p * nstrip + s], cnts[p * nstrip + s]) for s in range(nstrip))
            planes.append(np.frombuffer(data, dtype, count=h * w * cps).reshape(h, w, cps))
    arr = np.concatenate(planes, axis=2) if nplanes > 1 else planes[0]
    arr = arr.astype(dtype.newbyteorder("="), copy=True)
    return arr[..., 0] if arr.shape[2] == 1 else arr
