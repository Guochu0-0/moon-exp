#!/bin/bash
# E32 只用（#221）：一张卡上先训练（train_queue.py，领到没有可训的为止），再转去评测（eval_queue.py）。
# EVAL_ONLY=1 时只评测。日志写 runs/E32/ckpt/card_<host>_g<GPU>.log。
#   GPU=<卡> [METHODS=acc4] [EVAL_ONLY=1] nohup bash runs/E32/code/card.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
PY=/opt/envs/loftr/bin/python
log=runs/E32/ckpt/card_$(hostname)_g$GPU.log
mkdir -p runs/E32/ckpt
if [ "${EVAL_ONLY:-0}" != 1 ]; then
  $PY runs/E32/code/train_queue.py >> "$log" 2>&1
fi
$PY runs/E32/code/eval_queue.py >> "$log" 2>&1
