#!/bin/bash
# 开跑前的两项实测（#197），在 worktree 根目录：GPU=<卡> nohup bash runs/E22/code/anchor.sh > runs/E22/ckpt/anchor.log 2>&1 < /dev/null &
# 1. dist.py：zero-shot 下尺度 16、8、4 抽到位置离参照的距离分布，以及 gauss / cauchy 两种细化奖励在单精度下的表现；
# 2. 用 configs/_anchor.toml（细化部分学习率 = 粗级 decoder）训练前 100 步，记各部分参数每步的相对改变量。
set -euo pipefail
R=runs/E22
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
PY=/opt/envs/loftr/bin/python
mkdir -p $R/extra/data $R/ckpt/_anchor
$PY $R/code/dist.py --out $R/extra/data/dist.json
rm -f $R/ckpt/_anchor/log.jsonl
$PY -m finetune.train $R/configs/_anchor.toml --out $R/ckpt/_anchor --stop-at 100
cp $R/ckpt/_anchor/log.jsonl $R/extra/data/anchor_log.jsonl
$PY $R/code/anchor.py $R/extra/data/anchor_log.jsonl
