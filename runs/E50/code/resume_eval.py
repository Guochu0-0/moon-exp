"""资源让卡后的评测续跑：保留训练及完整Val结果，不重建标签、不重新训练。"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from finetune import config as C
from workbench import launch
from baselines.match import sha256

R = ROOT / "runs/E50"
PY, WB = "/opt/envs/loftr/bin/python", "/opt/envs/wb/bin/python"


def main():
    assert (R / "ckpt/main/train.done").is_file()
    assert (R / "ckpt/main/ckpt_7907.pt").is_file()
    last = json.loads((R / "ckpt/main/log.jsonl").read_text().splitlines()[-1])
    assert last["step"] == 7907
    labels = json.loads((R / "extra/labels_meta.json").read_text())
    assert sha256(R / "extra/labels.jsonl") == labels["labels_sha256"]
    cfg = C.load(R / "configs/main.toml")
    cfg["run"]["eval_gpus"] = [int(g) for g in os.environ.get("EVAL_GPUS", "0,1").split(",")]
    cfg = C.expand(cfg)
    gpu = cfg["run"]["eval_gpus"][0]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    complete = len(list((R / "sweep/main/S/preds").glob("step*/val.meta.json")))
    spec = importlib.util.spec_from_file_location("finetune_driver267", ROOT / "scripts/finetune/run.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    launch.begin(R, "main", [sys.executable, *sys.argv], entry="runs/E50/code/resume_eval.py",
                 config_file="runs/E50/configs/main.toml", args=cfg,
                 config=json.loads((ROOT / cfg["model"]["config"]).read_text()),
                 reason=os.environ.get("EVAL_RESUME_REASON", "Val resumed after withdrawing owned processes on shared GPUs"),
                 completed_val_checkpoints=complete, training_complete_step=7907)
    try:
        driver.Job(R, "main", cfg, gpu)()
    except BaseException:
        launch.end(R, "main", "fail")
        raise
    launch.end(R, "main", "ok")
    for cmd in ([WB, "runs/E50/code/summary.py"], [WB, "-m", "workbench", "eval", "E50"],
                [WB, "-m", "workbench", "check"]):
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    os.environ.setdefault("MOON_GIT_TIMEOUT", "600")
    os.environ["TORCH_HOME"] = "/remote-home/xufang/YGC/weights/torch_home"
    launch.begin(R, "pipeline_resume", [sys.executable, *sys.argv], entry="runs/E50/code/resume_eval.py")
    try:
        main()
    except BaseException:
        launch.end(R, "pipeline_resume", "fail")
        raise
    launch.end(R, "pipeline_resume", "ok")
