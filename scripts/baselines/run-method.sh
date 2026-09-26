#!/bin/bash
# usage: run-method.sh <method> <env> <gpu>   — claim, match val+test, fit into runs/B0 (eval is run once at the end)
set -e
M=$1; ENV=$2; GPU=$3
R=/remote-home/xufang/YGC/results/baselines
mkdir -p $R/_claims
if ! mkdir $R/_claims/$M 2>/dev/null; then echo "SKIP $M (claimed by $(cat $R/_claims/$M/host 2>/dev/null))"; exit 0; fi
echo "$(hostname) gpu$GPU $(date +%T)" > $R/_claims/$M/host
cd /remote-home/xufang/YGC/moon-exp
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/opt/torch_home
for s in val test; do
  CUDA_VISIBLE_DEVICES=$GPU nice -n 10 ionice -c3 /opt/envs/$ENV/bin/python -m baselines.match configs/baselines/$M.json --split $s --out $R --resume
  nice -n 10 /opt/envs/wb/bin/python -m baselines.fit $R/$M --split $s --run runs/B0
done
echo "RUN $M DONE"
