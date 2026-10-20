"""py3.8 检查：moonlib 与 baselines 要在各 baseline 的 matcher 环境（py3.8）里 import，新语法会把它们弄坏。

两层：
- 编译：用服务器上的 py3.8 解释器逐个编译两个包的 .py（只在内存里 compile，不写 __pycache__），抓语法。
  解释器按顺序找：环境变量 MOON_PY38、PATH 上的 python3.8、下面列出的服务器路径；都找不到就跳过。
- 注解：编译过得去、import 时才报错的注解（定义时求值的 list[...]、X | Y），用当前解释器的 ast 扫，不需要 py3.8。
"""
import ast
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


PEP585 = {"list", "dict", "tuple", "set", "frozenset", "type"}


def runtime_annotation_errors(src: str, name: str) -> list:
    """编译过得去、import 时才炸的注解：def 的参数/返回值注解、模块与类层的变量注解在定义时求值，
    其中的 list[...] 等（PEP 585）和 X | Y（PEP 604）在 py3.8 上报 TypeError。文件有
    from __future__ import annotations 时注解不求值，不算。"""
    tree = ast.parse(src)
    if any(isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names)
           for n in tree.body):
        return []
    anns = []

    def visit(body, in_func):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                anns.extend(x.annotation for x in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg)
                            if x is not None and x.annotation is not None)
                if n.returns is not None:
                    anns.append(n.returns)
                visit(n.body, True)
            elif isinstance(n, ast.ClassDef):
                visit(n.body, in_func)
            elif isinstance(n, ast.AnnAssign):
                if not in_func:
                    anns.append(n.annotation)
            else:
                for field in ("body", "orelse", "finalbody", "handlers"):
                    visit(getattr(n, field, []) or [], in_func)

    visit(tree.body, False)
    out = []
    for ann in anns:
        for x in ast.walk(ann):
            if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) and x.value.id in PEP585:
                out.append(f"{name}:{x.lineno}: {x.value.id}[...]")
            elif isinstance(x, ast.BinOp) and isinstance(x.op, ast.BitOr):
                out.append(f"{name}:{x.lineno}: X | Y")
    return out


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


RUNTIME_ANNOTATION_CASES = [
    ("def f(x: list[int]): pass", ["<s>:1: list[...]"]),
    ("def f() -> dict[str, int]: pass", ["<s>:1: dict[...]"]),
    ("x: int | None = None", ["<s>:1: X | Y"]),
    ("class A:\n    y: tuple[int, ...] = ()", ["<s>:2: tuple[...]"]),
    ("def f():\n    y: list[int] = []", []),                  # 函数体内的变量注解不求值
    ("from __future__ import annotations\ndef f(x: list[int] | None): pass", []),
    ("def f(x: 'list[int]'): pass", []),
    ("from typing import List\ndef f(x: List[int]): pass", []),
]


@pytest.mark.parametrize("src,want", RUNTIME_ANNOTATION_CASES)
def test_runtime_annotation_scan(src, want):
    assert runtime_annotation_errors(src, "<s>") == want


def test_packages_have_no_runtime_py39_annotations():
    bad = [e for p in sources() for e in runtime_annotation_errors(Path(p).read_text(encoding="utf-8"), p)]
    assert not bad, "py3.8 上 import 时会报错的注解（加 from __future__ import annotations 或改用 typing）：\n" + "\n".join(bad)


def test_packages_compile_on_py38():
    exe = find_py38()
    if exe is None:
        pytest.skip("找不到 py3.8 解释器（可设 MOON_PY38）")
    r = subprocess.run([exe, "-c", COMPILE, *sources()], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"{exe} 编译不过：\n{r.stdout}{r.stderr}"
