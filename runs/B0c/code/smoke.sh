#!/bin/bash
# 用法: smoke.sh <gpu>
. /remote-home/xufang/YGC/moon-exp-cf70/cf70/common.sh
cd $D
CUDA_VISIBLE_DEVICES=$1 timeout 600 nice -n 10 $PY -m baselines.match $CFG --split val --limit 5 --weights $F/S1/ckpt_6000.pt --name smoke-coarse --out $MOON_RESULTS/coarse70/_smoke
