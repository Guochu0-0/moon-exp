#!/bin/bash
# 一张卡一个进程，按顺序领取任务（mkdir 占位在 runs/E4/.claims/，跨 gpfs 各机共享）：
#   cd $WT/<分支> && GPU=1 nohup bash runs/E4/code/run.sh > runs/E4/raw/queue_$(hostname)_g1.log 2>&1 < /dev/null &
# 先跑 6 个全量推理（每个约 5–10 分钟），再跑 6 个遮挡（每个约 2 小时）。
set -u
cd "$(dirname "$0")/../../.."
PY=/opt/envs/loftr/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:?GPU}
mkdir -p runs/E4/.claims runs/E4/raw
for job in infer:loftr_zs infer:roma_zs infer:loftr_q4 infer:roma_m4 infer:loftr_e1 infer:roma_e1 \
           occlude:loftr_zs occlude:roma_zs occlude:loftr_q4 occlude:roma_m4 occlude:loftr_e1 occlude:roma_e1; do
  step=${job%%:*}; m=${job#*:}
  mkdir runs/E4/.claims/${step}_$m 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) $(hostname) g$GPU start $job" | tee -a runs/E4/.claims/${step}_$m/host
  (cd runs/E4/code && $PY $step.py $m) && echo "$(date -u +%FT%TZ) done $job" || echo "$(date -u +%FT%TZ) FAIL $job"
done
