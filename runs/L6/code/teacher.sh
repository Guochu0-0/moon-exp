#!/bin/bash
# L6 的老师 warp：零样本 AnyMatch-RoMa（与伪标签同一个老师）按评测口径在参与训练的 3954 对（前 50%）上推理，
# 存两个方向的落点与 certainty（288 网格，float16）到 runs/L6/raw/teacher0/，不进 git。
# 用法（在本票 worktree 根目录）：GPU=<空卡> bash runs/L6/code/teacher.sh
set -euo pipefail
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:?GPU 未设}
/opt/envs/loftr/bin/python -m finetune.dense_label /remote-home/xufang/YGC/weights/anymatch/RoMa_AnyMatch.pth \
    --labels /remote-home/xufang/YGC/results/finetune/labels_roma0.jsonl --top 0.5 --out runs/L6/raw/teacher0
