#!/bin/bash
# E49 只用（#261）：定新项系数的 100 次更新（configs/_calib.toml：E41 门槛 0.8 + 图像相似度约束奖励、系数 190 真正施加、另记 4 卡平均后的梯度范数），4 卡，不存档。
# 产物 ckpt/_calib/（log.jsonl 等，不进 git），之后 code/calib.py 取系数。在 worktree 根目录运行：
#     GPU=0,1,2,3 nohup bash runs/E49/code/calib.sh > runs/E49/ckpt/calib_<host>.log 2>&1 < /dev/null &
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID
export MOON_DATA=${MOON_DATA:-/remote-home/xufang/YGC/dataset/Moon} MOON_WEIGHTS=${MOON_WEIGHTS:-/remote-home/xufang/YGC/weights}
mkdir -p runs/E49/ckpt/_calib
exec /opt/envs/loftr/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 -m finetune.train \
    runs/E49/configs/_calib.toml --out runs/E49/ckpt/_calib
