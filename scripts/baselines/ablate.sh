#!/bin/bash
# SAR input-mapping ablation on Val.  usage: GPU=<id> ablate.sh <method>__<map> ...
# Derives a config from configs/baselines/<method>.json with input.sar=<map>, method=<method>__<map>;
# claims via mkdir (safe across hosts on gpfs), matches Val, fits into runs/B0m.
cd /remote-home/xufang/YGC/moon-exp
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/opt/torch_home
R=/remote-home/xufang/YGC/results/baselines_ablation
mkdir -p $R/_claims $R/_cfg
for job in "$@"; do
  m=${job%__*}; map=${job#*__}
  if ! mkdir $R/_claims/$job 2>/dev/null; then echo "$(date +%T) SKIP $job"; continue; fi
  echo "$(hostname) gpu$GPU $(date +%T)" > $R/_claims/$job/host
  /opt/envs/wb/bin/python -c "
import json,sys
c=json.load(open('configs/baselines/$m.json')); c['input']['sar']='$map'; c['method']='$job'
json.dump(c,open('$R/_cfg/$job.json','w'),ensure_ascii=False,indent=1)"
  echo "$(date +%T) START $job"
  if CUDA_VISIBLE_DEVICES=$GPU nice -n 10 ionice -c3 /opt/envs/loftr/bin/python -m baselines.match $R/_cfg/$job.json \
       --split val --out $R --resume > $R/_claims/$job/log 2>&1 \
     && nice -n 10 /opt/envs/wb/bin/python -m baselines.fit $R/$job --split val --run runs/B0m >> $R/_claims/$job/log 2>&1; then
    echo "$(date +%T) OK $job"
  else
    echo "$(date +%T) FAIL $job"
  fi
done
echo "$(date +%T) ABLATE DONE"
