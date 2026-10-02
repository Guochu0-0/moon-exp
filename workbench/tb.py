"""TensorBoard 日志：流式读 scalars，以及「在 TensorBoard 中打开」用的子进程。调研见 docs/research/tensorboard-logs.md。

读取：自写 TFRecord 帧解析（uint64 长度 | uint32 crc | 数据 | uint32 crc，不校验 crc），只依赖 tensorboard 包里的
event_pb2，只取 simple_value。比官方读取器快一个数量级，也不会像 EventAccumulator 那样悄悄蓄水池采样。

- 一个目录（不递归）是一个 run：其中的 events 文件按文件名里的时间戳排序，逐个 tag 丢掉旧文件中
  step ≥ 新文件里该 tag 起始 step 的点（续训重叠）。必须按 tag 做：同一文件里有的 tag 用 global_step、有的用 epoch。
- 一个 logdir（tb/<method>/）下每个含 events 文件的目录各是一个 run，按相对路径命名：不同的 version_N 分开，
  add_scalars 建的子目录（loss_train/、loss_val/）也各是一个 run，tag 相同。media/ 按约定只放图片，不读。

子进程：`python -m tensorboard.main --logdir <d> --host 127.0.0.1 --port <空闲端口>`，每个 logdir 复用一个实例，
工作台退出时统一 terminate（atexit）；Windows 上另把子进程放进「关闭即杀」的 job，Linux 上设 PDEATHSIG，
工作台被强杀时 TensorBoard 也随之结束。不在进程内调 program.TensorBoard().launch()：它没有停止接口，还会和工作台抢 GIL。
"""
from __future__ import annotations

import atexit
import math
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

EVENTS = "events.out.tfevents."
SKIP_DIRS = ("media", "checkpoints")
MAX_POINTS = 2000          # 每条曲线最多下发的点数；超过时按序号分桶，保留每桶最小、最大值
START_TIMEOUT = 90         # 秒，等 TensorBoard 能响应 /data/runs


def read_file(path) -> dict[str, list[tuple[int, float]]]:
    """一个 events 文件里的 simple_value：{tag: [(step, value)]}，按写入顺序。尾部写到一半或坏掉的记录处停止。"""
    from google.protobuf.message import DecodeError
    from tensorboard.compat.proto import event_pb2

    out: dict[str, list[tuple[int, float]]] = {}
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                break
            (n,) = struct.unpack("<Q", head[:8])
            body = f.read(n + 4)
            if len(body) < n + 4:
                break
            try:
                ev = event_pb2.Event.FromString(body[:n])
            except DecodeError:
                break
            for v in ev.summary.value:
                if v.HasField("simple_value"):
                    out.setdefault(v.tag, []).append((ev.step, v.simple_value))
    return out


def _timestamp(f: Path) -> tuple[int, str]:
    m = re.match(re.escape(EVENTS) + r"(\d+)", f.name)
    return (int(m.group(1)) if m else 0, f.name)


def event_files(d: Path) -> list[Path]:
    return sorted((f for f in d.iterdir() if f.is_file() and f.name.startswith(EVENTS)), key=_timestamp)


def read_dir(d) -> dict[str, dict]:
    """一个 run：{tag: {"step": [...], "value": [...]}}，续训重叠已按 tag 截断。"""
    tags: dict[str, list[tuple[int, float]]] = {}
    for f in event_files(Path(d)):
        for tag, pts in read_file(f).items():
            start = min(s for s, _ in pts)
            tags[tag] = [p for p in tags.get(tag, []) if p[0] < start] + pts
    return {t: {"step": [s for s, _ in p], "value": [v for _, v in p]} for t, p in sorted(tags.items())}


def run_dirs(root) -> list[Path]:
    """logdir 下含 events 文件的目录（含 root 自己），跳过 media/ 与 checkpoints/。"""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for d, subs, files in os.walk(root):
        subs[:] = sorted(s for s in subs if s not in SKIP_DIRS)
        if any(f.startswith(EVENTS) for f in files):
            out.append(Path(d))
    return out


def read_logdir(root) -> dict[str, dict]:
    """{run 的相对路径: read_dir 的结果}；没有 scalar 的 run（比如 add_scalars 留下的空父目录）不列。"""
    root = Path(root)
    out = {}
    for d in run_dirs(root):
        tags = read_dir(d)
        if tags:
            out[d.relative_to(root).as_posix() or "."] = tags
    return dict(sorted(out.items()))


def thin(series: dict, limit: int = MAX_POINTS) -> dict:
    """点数超过 limit 时按序号分桶，每桶保留最小、最大值所在的点，另保留首尾点；n 为原始点数。"""
    s, v = series["step"], series["value"]
    n = len(s)
    if n <= limit:
        return {"step": s, "value": v, "n": n}
    buckets = (limit - 2) // 2
    keep = {0, n - 1}
    for b in range(buckets):
        lo, hi = b * n // buckets, (b + 1) * n // buckets
        idx = [i for i in range(lo, hi) if not math.isnan(v[i])]
        if idx:
            keep.add(min(idx, key=v.__getitem__))
            keep.add(max(idx, key=v.__getitem__))
        else:
            keep.add(lo)
    keep = sorted(keep)
    return {"step": [s[i] for i in keep], "value": [v[i] for i in keep], "n": n}


# ---------------- 「在 TensorBoard 中打开」 ----------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TensorBoards:
    """每个 logdir 一个 TensorBoard 子进程，复用；close（以及 atexit）时统一 terminate。"""

    def __init__(self, python: str = sys.executable):
        self.python = python
        self.procs: dict[str, tuple[subprocess.Popen, str]] = {}
        self._lock = threading.Lock()
        self._job = _kill_on_close_job()
        atexit.register(self.close)

    def open(self, logdir) -> str:
        key = str(Path(logdir).resolve())
        with self._lock:
            hit = self.procs.get(key)
            if hit and hit[0].poll() is None:
                return hit[1]
            port = free_port()
            url = f"http://127.0.0.1:{port}/"
            log = tempfile.TemporaryFile()
            kw = {"preexec_fn": _die_with_parent} if sys.platform.startswith("linux") else {}
            p = subprocess.Popen([self.python, "-m", "tensorboard.main", "--logdir", key, "--host", "127.0.0.1",
                                  "--port", str(port)], stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kw)
            if self._job:
                _assign(self._job, p)
            t0 = time.time()
            while True:
                if p.poll() is not None:
                    log.seek(0)
                    tail = log.read().decode("utf-8", "replace").strip().splitlines()[-5:]
                    raise RuntimeError(f"TensorBoard 启动失败（退出码 {p.returncode}）：" + " / ".join(tail))
                try:
                    with urllib.request.urlopen(url + "data/runs", timeout=2):
                        break
                except OSError:
                    if time.time() - t0 > START_TIMEOUT:
                        p.terminate()
                        raise RuntimeError(f"TensorBoard {START_TIMEOUT} 秒内没有就绪") from None
                    time.sleep(0.3)
            self.procs[key] = (p, url)
            return url

    def close(self):
        with self._lock:
            for p, _ in self.procs.values():
                if p.poll() is None:
                    p.terminate()
            for p, _ in self.procs.values():
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
            self.procs.clear()


def _die_with_parent():   # Linux：父进程（工作台）死掉时内核给 TensorBoard 发 SIGTERM
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG


def _kill_on_close_job():
    """Windows：建一个 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE 的 job。工作台进程一结束（包括被强杀），句柄关闭，job 里的进程全被杀。"""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not job or not k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        return None                                          # 9 = JobObjectExtendedLimitInformation；失败就只靠 atexit
    return job


def _assign(job, p: subprocess.Popen):
    import ctypes
    from ctypes import wintypes

    ctypes.WinDLL("kernel32").AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(p._handle)))
