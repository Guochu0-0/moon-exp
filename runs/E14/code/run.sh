#!/bin/bash
# 跑全部各组并写启动记录（#179）。在 worktree 根目录：GPU=<卡> nohup bash runs/E14/code/run.sh > runs/E14/.claims/run.log 2>&1 < /dev/null &
set -uo pipefail
mkdir -p runs/E14/.claims; mkdir runs/E14/.claims/main || { echo "已在跑"; exit 1; }
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
PY=/opt/envs/loftr/bin/python
CMD=($PY runs/E14/code/probe.py --n 200)
$PY -m workbench.launch begin runs/E14 main -- "${CMD[@]}" || exit 2
if "${CMD[@]}"; then st=ok; else st=fail; fi
$PY -m workbench.launch end runs/E14 main $st
