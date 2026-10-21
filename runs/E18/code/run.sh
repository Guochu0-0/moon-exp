#!/bin/bash
# 跑 E18 的两项并写一次启动记录（#189）：诊断探针 probe.py 在第一张卡，开销测试 cost.py 在第二张卡（同时跑）。
# 在 worktree 根目录：GPUS="0 1" nohup bash runs/E18/code/run.sh > runs/E18/.claims/run.log 2>&1 < /dev/null &
set -uo pipefail
R=runs/E18
mkdir -p $R/.claims
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/opt/envs/loftr/bin/python
set -- $GPUS
CUDA_VISIBLE_DEVICES=$1,$2 $PY -m workbench.launch begin $R main -- bash $R/code/run.sh || exit 2
CUDA_VISIBLE_DEVICES=$1 $PY $R/code/probe.py --n 200 > $R/.claims/probe.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=$2 $PY $R/code/cost.py > $R/.claims/cost.log 2>&1 &
p2=$!
st=ok
wait $p1 || { echo "probe FAIL"; st=fail; }
wait $p2 || { echo "cost FAIL"; st=fail; }
$PY -m workbench.launch end $R main $st
