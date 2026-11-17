"""复用完整预测完成 E50 的 CPU 汇总、工作台评价和记录检查。"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from workbench import launch

R = ROOT / "runs/E50"
WB = "/opt/envs/wb/bin/python"


if __name__ == "__main__":
    os.environ["MOON_DATA"] = "/remote-home/xufang/YGC/dataset/Moon"
    os.environ.setdefault("MOON_GIT_TIMEOUT", "600")
    launch.begin(R, "finalize", [sys.executable, *sys.argv], entry="runs/E50/code/finalize.py")
    try:
        for cmd in ([WB, "runs/E50/code/summary.py"], [WB, "-m", "workbench", "eval", "E50"],
                    [WB, "-m", "workbench", "check"]):
            subprocess.run(cmd, cwd=ROOT, check=True)
    except BaseException:
        launch.end(R, "finalize", "fail")
        raise
    launch.end(R, "finalize", "ok")
