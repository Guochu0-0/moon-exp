#!/usr/bin/env bash
# T3：AdaBN（1603.04779）：在 Train 上重估参照配方 VGG 的 BN 统计。参照取 M15（种子 0，第 7000 步，Val AUC@5 0.3156）。
# Train 均匀取 2000 对，光学、SAR 各按训练口径缩放到 560 过一遍 VGG（推理还有 864 那一遍，这里不单独估它）。
#   GPU=<空卡> bash runs/T3/code/run.sh      （在 worktree 根目录）
set -euo pipefail
R=runs/T3; CK=$R/ckpt; MC=/remote-home/xufang/YGC/moon-exp/runs/M/ckpt
Y=/remote-home/xufang/YGC; PY=/opt/envs/loftr/bin/python
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $CK
[ -f $CK/main.pt ] || $PY scripts/finetune/merge.py adabn $CK/main.pt --init $MC/M15/ckpt_7000.pt --n 2000
bash scripts/finetune/eval_ckpt.sh $R main $CK/main.pt
