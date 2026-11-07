#!/bin/bash
# E47 只用（#255）：定新项系数的 50 次更新（configs/_calib.toml：E41 门槛 0.8 + 正反向互逆约束奖励、系数 0、只量不学），4 卡，不存档。
# 产物 ckpt/_calib/（log.jsonl 等，不进 git），之后 code/calib.py 取系数。在 worktree 根目录运行：
#     GPU=0,1,2,3 nohup bash runs/E47/code/calib.sh > runs/E47/ckpt/calib_<host>.log 2>&1 < /dev/null &
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID
export MOON_DATA=${MOON_DATA:-/remote-home/xufang/YGC/dataset/Moon} MOON_WEIGHTS=${MOON_WEIGHTS:-/remote-home/xufang/YGC/weights}
mkdir -p runs/E47/ckpt/_calib
exec /opt/envs/loftr/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 -m finetune.train \
    runs/E47/configs/_calib.toml --out runs/E47/ckpt/_calib
