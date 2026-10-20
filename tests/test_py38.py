"""py3.8 编译检查：moonlib 与 baselines 要在各 baseline 的 matcher 环境（py3.8）里 import，新语法会把它们弄坏。

用服务器上的 py3.8 解释器逐个编译两个包的 .py（只在内存里 compile，不写 __pycache__）。
解释器按顺序找：环境变量 MOON_PY38、PATH 上的 python3.8、下面列出的服务器路径；都找不到就跳过。
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("moonlib", "baselines")
CANDIDATES = ("/opt/envs/matcher/bin/python",
              "/root/anaconda3/envs/roma/bin/python3.8")     # 126 上的 conda 环境（2026-10-20 核实为 3.8.12）

COMPILE = """
import sys
bad = []
for p in sys.argv[1:]:
    try:
        compile(open(p, encoding="utf-8").read(), p, "exec")
    except SyntaxError as e:
        bad.append("%s:%s: %s" % (p, e.lineno, e.msg))
print("\\n".join(bad))
sys.exit(1 if bad else 0)
"""


def _is_py38(exe) -> bool:
    try:
        r = subprocess.run([exe, "-c", "import sys; print(sys.version_info[:2] == (3, 8))"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.stdout.strip() == "True"


def find_py38():
    for exe in (os.environ.get("MOON_PY38"), shutil.which("python3.8"), *CANDIDATES):
        if exe and Path(exe).exists() and _is_py38(exe):
            return exe
    return None


def sources():
    return sorted(str(p) for pkg in PACKAGES for p in (REPO / pkg).rglob("*.py") if "__pycache__" not in p.parts)


def test_packages_have_sources():
    assert {Path(p).relative_to(REPO).parts[0] for p in sources()} == set(PACKAGES)


def test_packages_compile_on_py38():
    exe = find_py38()
    if exe is None:
        pytest.skip("找不到 py3.8 解释器（可设 MOON_PY38）")
    r = subprocess.run([exe, "-c", COMPILE, *sources()], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"{exe} 编译不过：\n{r.stdout}{r.stderr}"
