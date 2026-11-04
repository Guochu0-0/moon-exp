#!/bin/bash
# E40（#242）：推理一个分片。在本票 worktree 根目录运行：
#   GPU=<空卡> CK=<E37 all 的 ckpt_7907.pt> nohup bash runs/E40/code/run.sh <k> <n> > runs/E40/raw/log_s<k>.txt 2>&1 < /dev/null &
# 全部分片完成后：bash runs/E40/code/post.sh
# 抽样噪声对照：前面加 MODE=resample，启动记录记为 resample_s<k>，完成后：bash runs/E40/code/post_resample.sh
set -euo pipefail
K=$1; N=$2; MODE=${MODE:-constraints}
TAG=infer_s$K; [ "$MODE" = resample ] && TAG=resample_s$K
Y=/remote-home/xufang/YGC
PY=/opt/envs/loftr/bin/python; WB=/opt/envs/wb/bin/python
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home PYTHONPATH=.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
ok=fail
$WB -m workbench.launch begin runs/E40 $TAG -- env MODE=$MODE bash runs/E40/code/run.sh "$@"
trap '$WB -m workbench.launch end runs/E40 $TAG $ok' EXIT
nice -n 10 $PY runs/E40/code/infer.py --ckpt "$CK" --shard "$K/$N" --out runs/E40/raw --mode "$MODE"
ok=ok
