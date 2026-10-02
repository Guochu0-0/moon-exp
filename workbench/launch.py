"""启动检查与运行记录（「管理规范落地：文档与工具」#76）。规矩见 docs/agents/experiments.md。

    python -m workbench.launch check                          # 代码不干净就退出码 2
    python -m workbench.launch begin runs/P P17 -- <命令...>   # 检查 + 在 launch/P17.json 追加一次启动
    python -m workbench.launch end runs/P P17 ok|fail          # 给最近一次启动补上结束时间与结果

「干净」= 有 git、已跟踪的代码没有改动、没有未跟踪的代码文件。runs/ 下的产物不算，但 runs/<id>/code/ 算（一次性代码也要先提交）。
被 .gitignore 排除的文件不算。临时调试可设 MOON_ALLOW_DIRTY=1 放行，launch 记录里会留下 dirty 和放行标记。
兼容 Python 3.10（训练环境也调用 require_clean），不依赖 workbench 的其他模块。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOW_ENV = "MOON_ALLOW_DIRTY"
_CODE = (".", ":(exclude,top)runs")
_RUN_CODE = (":(glob,top)runs/*/code/**",)


def _git(repo: Path, *args, timeout=120) -> str | None:
    # --no-optional-locks：不刷新索引、不拿 index.lock（gpfs 上 status 慢，被超时杀掉会留残锁）
    r = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True,
                       encoding="utf-8", timeout=timeout)
    return r.stdout if r.returncode == 0 else None


def repo_state(repo=REPO) -> dict:
    """commit、dirty，以及不干净的文件（git status --porcelain 的行）。不在 git 里时 commit 为 None。"""
    repo = Path(repo)
    commit = _git(repo, "rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "dirty": None, "changes": []}
    lines = []
    for spec in (_CODE, _RUN_CODE):
        out = _git(repo, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=untracked", "--", *spec)
        lines += [ln for ln in (out or "").splitlines() if ln.strip()]
    return {"commit": commit.strip(), "dirty": bool(lines), "changes": lines}


def require_clean(repo=REPO) -> dict:
    """不干净就打印原因并退出（码 2）；设了 MOON_ALLOW_DIRTY=1 时只警告。返回 repo_state 加 allow_dirty。"""
    st = repo_state(repo)
    allow = os.environ.get(ALLOW_ENV) == "1"
    if st["commit"] is None:
        msg = f"{repo} 不是 git 仓库（导出副本不能跑实验，见 docs/agents/experiments.md）"
    elif st["dirty"]:
        msg = "代码有未提交的改动，先提交再跑（docs/agents/experiments.md）：\n  " + "\n  ".join(st["changes"][:30])
    else:
        return {**st, "allow_dirty": False}
    if not allow:
        print(f"launch: {msg}\n临时调试可设 {ALLOW_ENV}=1。", file=sys.stderr)
        raise SystemExit(2)
    print(f"launch: 警告（{ALLOW_ENV}=1 放行）：{msg}", file=sys.stderr)
    return {**st, "allow_dirty": True}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _path(run_dir, method) -> Path:
    return Path(run_dir) / "launch" / f"{method}.json"


def _load(p: Path) -> list:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _save(p: Path, items: list):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def begin(run_dir, method: str, cmd: list[str], repo=REPO) -> dict:
    """检查代码，然后在 runs/<id>/launch/<method>.json 追加一次启动（续训、重跑各算一次）。"""
    st = require_clean(repo)
    rec = {"start": _now(), "end": None, "status": None, "commit": st["commit"], "dirty": st["dirty"],
           "allow_dirty": st["allow_dirty"], "host": socket.gethostname(), "cwd": os.getcwd(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"), "cmd": cmd}
    if st["dirty"]:
        rec["changes"] = st["changes"]
    p = _path(run_dir, method)
    _save(p, _load(p) + [rec])
    return rec


def end(run_dir, method: str, status: str):
    p = _path(run_dir, method)
    items = _load(p)
    if not items:
        raise SystemExit(f"launch: {p} 没有启动记录")
    items[-1].update(end=_now(), status=status)
    _save(p, items)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check"]:
        st = require_clean()
        print(st["commit"])
    elif argv[:1] == ["begin"] and len(argv) >= 3:
        cmd = argv[argv.index("--") + 1:] if "--" in argv else []
        begin(argv[1], argv[2], cmd)
    elif argv[:1] == ["end"] and len(argv) == 4:
        end(argv[1], argv[2], argv[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
