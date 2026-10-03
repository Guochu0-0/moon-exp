"""Notes 的 Markdown 渲染与写回（前端 notes.js），用 node 跑 workbench_notes.test.mjs。"""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="没有 node")
def test_notes_markdown():
    r = subprocess.run(["node", "--test", str(Path(__file__).with_name("workbench_notes.test.mjs"))],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
