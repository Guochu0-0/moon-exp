#!/bin/bash
# 开跑前的核对（#202），在 worktree 根目录：GPU=<卡> nohup bash runs/E25/code/smoke.sh > runs/E25/ckpt/smoke.log 2>&1 < /dev/null &
# 用 configs/fr1_s0.toml 训练前 30 步：看第 1 步显存（多一份冻结模型），以及第 1 步冻结参照与当前模型的参照是否重合
# （此时两者权重相同，ref_gap 应为 0、d_cur 应等于 d_ref）。
set -euo pipefail
R=runs/E25
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $R/ckpt/_smoke
rm -f $R/ckpt/_smoke/log.jsonl
/opt/envs/loftr/bin/python -m finetune.train $R/configs/fr1_s0.toml --out $R/ckpt/_smoke --stop-at 30
/opt/envs/loftr/bin/python - <<'PY'
import json
rs = [json.loads(l) for l in open("runs/E25/ckpt/_smoke/log.jsonl") if l.strip()]
for r in rs[:3] + rs[-2:]:
    print({k: r.get(k) for k in ("step", "mem_gb", "sec", "d_ref", "d_cur", "ref_gap", "ref_ok", "cur_ok", "rew", "loss")})
PY
