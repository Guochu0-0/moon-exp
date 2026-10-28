#!/bin/bash
# E35 只用（#230）：一张卡上同时起训练（train_queue.py，领到没有可训的为止）和评测（eval_queue.py，存档一出就评 Val），
# 两者共用这张卡（训练约 11 GB，评测另占几 GB）。EVAL_ONLY=1 时只评测；TRAIN_ONLY=1 时只训练。
# 日志写 runs/E35/ckpt/{train,eval}_<host>_g<GPU>.log。
#   GPU=<卡> [METHODS=tau0.999] [EVAL_ONLY=1|TRAIN_ONLY=1] nohup bash runs/E35/code/card.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
PY=/opt/envs/loftr/bin/python
h=$(hostname)
mkdir -p runs/E35/ckpt
if [ "${EVAL_ONLY:-0}" != 1 ]; then
  $PY runs/E35/code/train_queue.py >> "runs/E35/ckpt/train_${h}_g$GPU.log" 2>&1 &
fi
if [ "${TRAIN_ONLY:-0}" != 1 ]; then
  sleep 30
  $PY runs/E35/code/eval_queue.py >> "runs/E35/ckpt/eval_${h}_g$GPU.log" 2>&1 &
fi
wait
