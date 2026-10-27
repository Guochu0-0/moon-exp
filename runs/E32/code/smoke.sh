#!/bin/bash
# E32 只用（#221）：正式启动前的冒烟运行。每组只训前几步（--stop-at），产物写到 YGC/tmp221/smoke/<组>（不进实验记录），
# 只看能不能跑通、显存与每步耗时、日志里有没有 d_ref、in_frame 与细化各级的量。
#   GPU=<卡> nohup bash runs/E32/code/smoke.sh <组>... > <日志> 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
Y=/remote-home/xufang/YGC
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
for m in "$@"; do
  n=30
  [ "$m" = acc4 ] && n=40
  echo "== $m"
  nice -n 10 /opt/envs/loftr/bin/python -m finetune.train runs/E32/configs/$m.toml --out $Y/tmp221/smoke/$m --stop-at $n 2>&1 | tail -3
  echo "exit $m ${PIPESTATUS[0]}"
done
