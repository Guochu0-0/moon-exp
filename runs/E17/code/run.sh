#!/bin/bash
# 跑一批任务并写一次启动记录（#187）。每张卡一个 worker，按顺序用 mkdir 领 probe.py 的任务，产物已在的跳过。
# 在 worktree 根目录：GPUS="1 2" nohup bash runs/E17/code/run.sh <任务…> > runs/E17/.claims/run_<批>.log 2>&1 < /dev/null &
set -uo pipefail
R=runs/E17
mkdir -p $R/.claims $R/raw
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/opt/envs/loftr/bin/python
CUDA_VISIBLE_DEVICES=${GPUS// /,} $PY -m workbench.launch begin $R main -- bash $R/code/run.sh "$@" || exit 2
worker() {
  local g=$1; shift
  for t in "$@"; do
    ls $R/raw/$t.* > /dev/null 2>&1 && continue
    mkdir $R/.claims/$t 2> /dev/null || continue
    echo "[$(date +%T)] g$g $t"
    if CUDA_VISIBLE_DEVICES=$g $PY $R/code/probe.py $t > $R/.claims/$t/log 2>&1; then echo "[$(date +%T)] g$g $t ok"
    else echo "[$(date +%T)] g$g $t FAIL"; touch $R/.claims/fail; fi
  done
}
for g in $GPUS; do worker $g "$@" & done
wait
st=ok; [ -e $R/.claims/fail ] && st=fail
$PY -m workbench.launch end $R main $st
