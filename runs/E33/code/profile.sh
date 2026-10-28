#!/bin/bash
# E33 只用（#226）：在一张卡上依次跑 plain、detail 两种耗时测量（profile_step.py）。CFG 为配置，TAG 为输出子目录名。
#   GPU=<卡> CFG=runs/E32/configs/ema.toml TAG=base nohup bash runs/E33/code/profile.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
source runs/E33/code/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
out=runs/E33/ckpt/profile/$TAG
mkdir -p "$out"
for mode in ${MODES:-plain detail}; do
  $PY runs/E33/code/profile_step.py "$CFG" --out "$out/$mode" --mode $mode --steps ${STEPS:-60} >> "$out/run.log" 2>&1
done
echo DONE >> "$out/run.log"
