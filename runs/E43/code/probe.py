"""E43 定系数（#250）：照 configs/_probe.toml 从 zero-shot 起训前 50 次更新（新项只量梯度、不进更新），然后汇总系数。
在 gpfs worktree 里、4 张空卡上跑：

    GPU=0,1,2,3 nohup /opt/envs/loftr/bin/python runs/E43/code/probe.py > runs/E43/ckpt/probe_<host>.log 2>&1 < /dev/null &
    /opt/envs/loftr/bin/python runs/E43/code/probe.py --summary      # 只重算汇总

训练产物在 ckpt/probe/（不进 git）。汇总写 extra/data/probe.json：每次更新两项在共用参数上的梯度范数（日志 occ_g_old、
occ_g_new 为 4 张卡的平均）与比值 occ_g_ratio（各卡比值的平均），系数 = 50 次更新比值的中位数，取 2 位有效数字。
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("driver", REPO / "scripts" / "finetune" / "run.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
from baselines.match import git_head  # noqa: E402
from workbench import launch  # noqa: E402

CK = R / "ckpt" / "probe"


def summary():
    rows = [json.loads(x) for x in (CK / "log.jsonl").read_text(encoding="utf-8").splitlines() if x.startswith("{")]
    keys = ("occ_g_old", "occ_g_new", "occ_g_ratio", "occ_g_old_all", "occ_g_new_all")
    ratio = np.array([r.get("occ_g_ratio", np.nan) for r in rows], float)
    med = float(np.nanmedian(ratio))
    args = json.loads((CK / "args.json").read_text(encoding="utf-8"))
    out = {"commit": args["commit"], "updates": len(rows),
           "ratio_median": round(med, 4),
           "ratio_quartiles": [round(float(x), 4) for x in np.nanpercentile(ratio, [25, 75])],
           "coef": float(f"{med:.2g}"),
           "rows": [{"step": r["step"], **{k: r.get(k) for k in keys}} for r in rows]}
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    (R / "extra" / "data" / "probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                                     encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"}, flush=True)


def main():
    if "--summary" in sys.argv:
        return summary()
    gpu = os.environ["GPU"]
    launch.require_clean(REPO)
    CK.mkdir(parents=True, exist_ok=True)
    print("commit", git_head(REPO), flush=True)
    from finetune.models import MODELS
    env = {"TORCH_HOME": MODELS["roma"].TORCH_HOME}
    n = len(gpu.split(","))
    D.run([D.PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}", "-m", "finetune.train",
           R / "configs" / "_probe.toml", "--out", CK],
          CK / "train.log", env, gpu)
    summary()


if __name__ == "__main__":
    main()
