#!/bin/bash
# 开跑前的小探针（#201），在 worktree 根目录：GPU=<卡> nohup bash runs/E24/code/probe.sh > runs/E24/ckpt/probe.log 2>&1 < /dev/null &
set -euo pipefail
R=runs/E24
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $R/extra/data
/opt/envs/loftr/bin/python $R/code/probe.py --out $R/extra/data/probe.json "$@"
