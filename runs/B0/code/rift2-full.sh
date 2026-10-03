#!/bin/bash
# RIFT2 full run on 154: prep -> N sharded MATLAB processes per split -> collect -> fit
N=${N:-8}
cd /remote-home/xufang/YGC/moon-exp
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon
IO=/opt/tmp/rift2
R=/remote-home/xufang/YGC/results/baselines
for s in val test; do
  nice -n 10 /opt/envs/wb/bin/python -m baselines.rift2 prep configs/baselines/rift2.json --split $s --io $IO
done
for s in val test; do
  echo "$(date +%T) matlab $s x$N"
  for k in $(seq 1 $N); do
    nice -n 10 /root/matlab-batch.sh "addpath('baselines/matlab'); addpath('third_party/RIFT2'); rift2_split('$IO/$s/in','$IO/$s/out',0,2,0,$k,$N)" > /root/scripts/rift2-$s-$k.log 2>&1 &
    sleep 3
  done
  wait
  echo "$(date +%T) matlab $s done: $(ls $IO/$s/out | wc -l) outputs"
  nice -n 10 /opt/envs/wb/bin/python -m baselines.rift2 collect configs/baselines/rift2.json --split $s --io $IO --out $R
  nice -n 10 /opt/envs/wb/bin/python -m baselines.fit $R/rift2 --split $s --run runs/B0
done
echo "RIFT2 DONE"
