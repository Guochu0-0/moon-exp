"""#267 的单次无人值守执行：老师分片→固定标签→原生训练驱动→汇总与检查。"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from workbench import launch

R = ROOT / "runs/E50"
PY = "/opt/envs/loftr/bin/python"
WB = "/opt/envs/wb/bin/python"


def run(cmd, env=None):
    print("RUN", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, check=True)


def main():
    probe = json.loads((R / "extra/probe.json").read_text())
    assert all(probe["finite_grad"].values()) and probe["loss_output_grad_max_diff"] < 1e-5
    assert probe["resolved_inference"] is not None
    children, streams = [], []
    for worker, gpu in ((0, "0"), (1, "1")):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        log = open(R / f"ckpt/teacher_worker{worker}.log", "a")
        streams.append(log)
        children.append(subprocess.Popen([PY, "runs/E50/code/prepare.py", "--worker", str(worker)], cwd=ROOT,
                        env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
    try:
        codes = [p.wait() for p in children]
    finally:
        for f in streams:
            f.close()
    assert codes == [0, 0], codes
    run([PY, "runs/E50/code/prepare.py", "--fit"])
    # 标签及身份先提交，正式训练的commit就包含实际固定标签。
    run(["git", "add", "runs/E50/extra", "runs/E50/launch"])
    run(["git", "commit", "-m", "experiment: 固定 E50 全量伪仿射与启动依据"])
    run(["timeout", "120", "git", "push"])
    env = {**os.environ, "GPU": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    run([PY, "scripts/finetune/run.py", "runs/E50"], env=env)
    completed = json.loads((R / "launch/main.json").read_text())[-1]
    assert completed["status"] == "ok" and completed["end"], completed
    run([WB, "runs/E50/code/summary.py"])
    run([WB, "-m", "workbench", "eval", "E50"])
    run([WB, "-m", "workbench", "check"])
    print("E50 execution complete; ready for record commit and PR", flush=True)


if __name__ == "__main__":
    os.environ["TORCH_HOME"] = "/remote-home/xufang/YGC/weights/torch_home"
    os.environ.setdefault("MOON_GIT_TIMEOUT", "600")
    launch.begin(R, "pipeline", [sys.executable, *sys.argv], entry="runs/E50/code/run_all.py",
                 args={"teacher_gpus": [0, 1], "training_gpu": 0, "seed": 0, "experiment": "E50",
                       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                       "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                       "MOON_GIT_TIMEOUT": os.environ["MOON_GIT_TIMEOUT"]})
    try:
        main()
    except BaseException:
        launch.end(R, "pipeline", "fail")
        raise
    launch.end(R, "pipeline", "ok")
