#!/bin/bash
# 用法: nohup run.sh <gpu> > $MOON_RESULTS/coarse70/run.log 2>&1 &
. /remote-home/xufang/YGC/moon-exp-cf70/cf70/common.sh
LOCK=$MOON_RESULTS/coarse70/run.lock
mkdir $LOCK || { echo "lock exists: $LOCK"; exit 1; }
echo $$ > $LOCK/pid
cd $D
for j in $JOBS; do
  name=${j%%:*}; ck=${j#*:}
  for s in val test; do
    if [ -f $MOON_RESULTS/coarse70/$name-coarse/$s.meta.json ]; then echo "skip $name/$s"; continue; fi
    W=""; [ -n "$ck" ] && W="--weights $ck"
    echo "=== $(date '+%F %T') $name $s $W"
    CUDA_VISIBLE_DEVICES=$1 nice -n 10 $PY -m baselines.match $CFG --split $s $W --name $name-coarse --out $MOON_RESULTS/coarse70
    echo "=== exit $? $(date '+%F %T')"
  done
done
echo ALLDONE $(date '+%F %T')
