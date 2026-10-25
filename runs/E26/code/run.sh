#!/bin/bash
# 跑一批方法（#206）。每张卡一个 worker，按顺序用 mkdir 领方法；每个方法依次跑 probe.py 的 morph_、learn_ 两项，
# 两项产物都在的跳过；每个方法一条启动记录（launch/<方法>.json）。
# 在 worktree 根目录：GPUS="2 3" nohup bash runs/E26/code/run.sh <方法…> > runs/E26/.claims/run_<主机>.log 2>&1 < /dev/null &
set -uo pipefail
R=runs/E26
mkdir -p $R/.claims $R/raw
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/opt/envs/loftr/bin/python
worker() {
  local g=$1; shift
  for m in "$@"; do
    [ -e $R/raw/morph_$m.npz ] && [ -e $R/raw/learn_$m.npz ] && continue
    mkdir $R/.claims/$m 2> /dev/null || continue
    echo "[$(date +%T)] g$g $m"
    CUDA_VISIBLE_DEVICES=$g $PY -m workbench.launch begin $R $m -- $PY $R/code/probe.py "{morph,learn}_$m" \
      || { echo "[$(date +%T)] g$g $m 启动记录失败"; continue; }
    st=ok
    for t in morph learn; do
      [ -e $R/raw/${t}_$m.npz ] && continue
      CUDA_VISIBLE_DEVICES=$g $PY $R/code/probe.py ${t}_$m >> $R/.claims/$m/log 2>&1 || st=fail
    done
    $PY -m workbench.launch end $R $m $st
    echo "[$(date +%T)] g$g $m $st"
  done
}
for g in $GPUS; do worker $g "$@" & done
wait
