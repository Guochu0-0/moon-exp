"""按实验把服务器 runs/<id>/ 里被 gitignore 的部分拉回本地。

    python -m workbench sync B0 B0m [--extra certainty]... [--tb all] [--host xufang154外网]

拉什么（路径相对 runs/）：
- 点对 `<id>/preds/<m>/<split>_matches.npz`：默认拉。
- 中间结果 `<id>/inter/<m>/<name>/` 下的数据：只拉 `--extra <name>` 点名的（meta.json 在 git 里，不拉）。
- TB 日志 `<id>/tb/<m>/`：默认只拉自写的 `scalars/`，以及上游格式（Lightning）的整个 `version_N/` 目录；
  `--tb all` 拉整个 tb/。两种情况都排除 `checkpoints/` 和 `*.ckpt`。
- ckpt `<id>/ckpt/`：永远不拉。

传输：`ssh <host> "cd <root>/runs && find …"` 拿远端清单（大小、mtime），本地已有且大小与 mtime（秒）都相同的跳过；
其余用 `ssh <host> "tar cf - -T -" | tar xf -` 流回来，tar 保留 mtime，所以下次就能跳过。只依赖 ssh 与 tar，
不需要 rsync。远程命令都加 timeout（见 docs/agents/servers.md）。测试时传 shell= 把 ["ssh", host] 换成本地 shell。
主机取参数、环境变量 MOON_SYNC_HOST、DEFAULT_HOST；远端根目录取参数、MOON_SYNC_ROOT、DEFAULT_ROOT。
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "xufang154外网"
DEFAULT_ROOT = "/remote-home/xufang/YGC/moon-exp"
HOST_ENV = "MOON_SYNC_HOST"
ROOT_ENV = "MOON_SYNC_ROOT"
LIST_TIMEOUT = 300      # 秒，远端 find
FETCH_TIMEOUT = 3600    # 秒，远端 tar
_VERSION = re.compile(r"version_\d+")


@dataclass
class Result:
    fetched: list[str]    # 相对 runs/ 的路径
    skipped: list[str]    # 本地已有且未变
    missing: list[str]    # 远端没有这个实验目录
    fetched_bytes: int


def should_pull(rel: str, extras=(), tb_all: bool = False) -> bool:
    """rel 是相对 runs/ 的路径，如 B0/preds/loftr/val_matches.npz。"""
    parts = rel.split("/")
    if len(parts) < 3:
        return False
    kind, rest = parts[1], parts[2:]
    if kind == "preds":
        return len(rest) == 2 and rest[1].endswith("_matches.npz")
    if kind == "inter":        # inter/<m>/<name>/…
        return len(rest) >= 3 and rest[1] in extras and rest[2:] != ["meta.json"]
    if kind == "tb":           # tb/<m>/…
        sub = rest[1:]
        if not sub or "checkpoints" in sub or sub[-1].endswith(".ckpt"):
            return False
        if tb_all or sub[0] == "scalars":
            return True
        return sub[0] != "media" and any(_VERSION.fullmatch(p) for p in sub[:-1])   # 上游格式的 version_N/
    return False               # ckpt/ 与进 git 的文件


def _cd_runs(remote_root: str) -> str:
    return f"cd {shlex.quote(remote_root + '/runs')} && "


def list_remote(ids, remote_root: str, shell: list[str]) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """{相对 runs/ 的路径: (大小, mtime 秒)}，以及远端缺的实验。ckpt/ 和 checkpoints/ 不进清单。"""
    script = (_cd_runs(remote_root) + f"for d in {' '.join(shlex.quote(i) for i in ids)}; do "
              f"if [ -d \"$d\" ]; then timeout {LIST_TIMEOUT} find \"$d\" \\( -path \"$d/ckpt\" -o -name checkpoints \\) "
              f"-prune -o -type f -printf '%s\\t%T@\\t%p\\n'; else printf '?\\t%s\\n' \"$d\"; fi; done")
    r = subprocess.run([*shell, script], capture_output=True, timeout=LIST_TIMEOUT + 60)
    if r.returncode != 0:
        raise RuntimeError(f"列远端文件失败（{r.returncode}）：{r.stderr.decode('utf-8', 'replace').strip()}")
    files, missing = {}, []
    for line in r.stdout.decode("utf-8").splitlines():
        cols = line.split("\t")
        if cols[0] == "?":
            missing.append(cols[1])
        elif len(cols) == 3:
            files[cols[2]] = (int(cols[0]), int(float(cols[1])))
    return files, missing


def fetch(paths: list[str], runs: Path, remote_root: str, shell: list[str]):
    """把 paths（相对 runs/）从远端流进本地 runs/。"""
    runs.mkdir(parents=True, exist_ok=True)
    cmd = _cd_runs(remote_root) + f"timeout {FETCH_TIMEOUT} tar cf - -T -"
    src = subprocess.Popen([*shell, cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 不用 -C：Git 自带的 GNU tar 会把 D:\… 里的冒号当成远程主机
    dst = subprocess.Popen(["tar", "xf", "-"], cwd=runs, stdin=src.stdout, stderr=subprocess.PIPE)
    src.stdout.close()
    src.stdin.write("".join(p + "\n" for p in paths).encode("utf-8"))
    src.stdin.close()
    dst_err = dst.communicate(timeout=FETCH_TIMEOUT + 60)[1]
    src_err = src.stderr.read()
    src.wait()
    if src.returncode or dst.returncode:
        err = (src_err + dst_err).decode("utf-8", "replace").strip()
        raise RuntimeError(f"传输失败（远端 {src.returncode}，本地 {dst.returncode}）：{err}")


def sync(ids, runs, *, host: str | None = None, remote_root: str | None = None, extras=(), tb_all: bool = False,
         shell: list[str] | None = None) -> Result:
    runs = Path(runs)
    remote_root = remote_root or os.environ.get(ROOT_ENV) or DEFAULT_ROOT
    shell = shell or ["ssh", host or os.environ.get(HOST_ENV) or DEFAULT_HOST]
    files, missing = list_remote(ids, remote_root, shell)
    fetched, skipped = [], []
    for rel, (size, mtime) in sorted(files.items()):
        if not should_pull(rel, extras, tb_all):
            continue
        local = runs / rel
        if local.is_file() and (st := local.stat()).st_size == size and int(st.st_mtime) == mtime:
            skipped.append(rel)
        else:
            fetched.append(rel)
    if fetched:
        fetch(fetched, runs, remote_root, shell)
    return Result(fetched, skipped, missing, sum(files[p][0] for p in fetched))

