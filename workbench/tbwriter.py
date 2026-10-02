"""只用标准库写 TensorBoard 标量（scalars），给训练环境用：服务器的 loftr 环境没装 tensorboard（#76）。

    w = ScalarWriter("runs/E3/tb/main/scalars")
    w.add("loss", 0.31, step)              # 只写 simple_value
    w.close()

文件格式同官方：events.out.tfevents.<秒>.<host>，TFRecord 帧（uint64 长度 | 掩码 crc32c | 数据 | 掩码 crc32c），
数据是手工编码的 Event protobuf，第一条是 file_version。workbench.tb 和 TensorBoard 都能读。
兼容 Python 3.10，不依赖 workbench 的其他模块。
"""
from __future__ import annotations

import socket
import struct
import time
from pathlib import Path

_POLY = 0x82F63B78   # crc32c（Castagnoli），反射形式
_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ _POLY if _c & 1 else _c >> 1
    _TABLE.append(_c)


def crc32c(data: bytes) -> int:
    c = 0xFFFFFFFF
    for b in data:
        c = _TABLE[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _masked(data: bytes) -> int:
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _field(num: int, payload: bytes) -> bytes:   # length-delimited（wire type 2）
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _event(wall_time: float, step: int, *, file_version: str | None = None, tag: str | None = None,
           value: float | None = None) -> bytes:
    msg = b"\x09" + struct.pack("<d", wall_time) + b"\x10" + _varint(step)
    if file_version is not None:
        msg += _field(3, file_version.encode())
    if tag is not None:
        v = _field(1, tag.encode()) + b"\x15" + struct.pack("<f", value)
        msg += _field(5, _field(1, v))   # Event.summary → Summary.value → Value
    return msg


def _record(data: bytes) -> bytes:
    head = struct.pack("<Q", len(data))
    return head + struct.pack("<I", _masked(head)) + data + struct.pack("<I", _masked(data))


class ScalarWriter:
    def __init__(self, logdir, flush_every: int = 100):
        d = Path(logdir)
        d.mkdir(parents=True, exist_ok=True)
        now = time.time()
        self.path = d / f"events.out.tfevents.{int(now)}.{socket.gethostname()}"
        self._f = open(self.path, "ab")
        self._f.write(_record(_event(now, 0, file_version="brain.Event:2")))
        self._n, self._every = 0, flush_every

    def add(self, tag: str, value, step: int):
        self._f.write(_record(_event(time.time(), int(step), tag=tag, value=float(value))))
        self._n += 1
        if self._n % self._every == 0:
            self._f.flush()

    def add_dict(self, rec: dict, step: int, prefix: str = ""):
        """rec 里的数值项各写一条（bool、非数值、step 本身跳过），训练循环的 log 行直接传进来。"""
        for k, v in rec.items():
            if k != "step" and isinstance(v, (int, float)) and not isinstance(v, bool):
                self.add(prefix + k, v, step)

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
