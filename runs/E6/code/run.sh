#!/bin/bash
# 一张卡一个进程，按顺序领取 6 个推理任务（mkdir 占位在 runs/E6/.claims/，跨 gpfs 各机共享）：
#   cd $WT/<分支> && GPU=1 nohup bash runs/E6/code/run.sh > runs/E6/raw/queue_$(hostname)_g1.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
PY=/opt/envs/loftr/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:?GPU}
mkdir -p runs/E6/.claims runs/E6/raw
for m in roma_zs roma_m4 roma_e1 loftr_zs loftr_q4 loftr_e1; do
  claim=runs/E6/.claims/infer_$m
  mkdir "$claim" 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) $(hostname) g$GPU start $m" | tee -a "$claim/host"
  (cd runs/E6/code && $PY infer.py $m) && echo "$(date -u +%FT%TZ) done $m" || echo "$(date -u +%FT%TZ) FAIL $m"
done
