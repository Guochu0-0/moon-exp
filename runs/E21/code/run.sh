#!/bin/bash
# 跑 E21 并写一次启动记录（#195）：先核对闭式与画相图（CPU），再把 200 对分两片在两张卡上跑 probe.py，最后汇总。
# 在 worktree 根目录：GPUS="1 2" nohup bash runs/E21/code/run.sh > runs/E21/.claims/run.log 2>&1 < /dev/null &
set -uo pipefail
R=runs/E21
mkdir -p $R/.claims
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/opt/envs/loftr/bin/python
set -- $GPUS
CUDA_VISIBLE_DEVICES=$1,$2 $PY -m workbench.launch begin $R main -- bash $R/code/run.sh || exit 2
st=ok
$PY $R/code/check_geom.py > $R/.claims/check.log 2>&1 || { echo "check FAIL"; st=fail; }
$PY $R/code/phase.py > $R/.claims/phase.log 2>&1 || { echo "phase FAIL"; st=fail; }
CUDA_VISIBLE_DEVICES=$1 $PY $R/code/probe.py --shard 0 --nshard 2 > $R/.claims/probe0.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=$2 $PY $R/code/probe.py --shard 1 --nshard 2 > $R/.claims/probe1.log 2>&1 &
p2=$!
wait $p1 || { echo "probe0 FAIL"; st=fail; }
wait $p2 || { echo "probe1 FAIL"; st=fail; }
[ $st = ok ] && { $PY $R/code/report.py > $R/.claims/report.log 2>&1 || { echo "report FAIL"; st=fail; }; }
$PY -m workbench.launch end $R main $st
