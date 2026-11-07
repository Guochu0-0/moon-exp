#!/bin/bash
# E46 只用（#254，照抄 E42 的 calib.sh）：定新项系数的 50 次更新（configs/_calib.toml：E41 门槛 0.8 + 同模态自配对约束奖励、
# 系数 0、只量不学），4 卡，不存档。产物 ckpt/_calib/（log.jsonl 等，不进 git），之后 code/calib.py 取系数。启动记录 launch/calib.json。
# 在 worktree 根目录运行：
#     GPU=0,1,2,3 nohup bash runs/E46/code/calib.sh > runs/E46/ckpt/calib_<host>.log 2>&1 < /dev/null &
set -euo pipefail
Y=/remote-home/xufang/YGC
WB=/opt/envs/wb/bin/python
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID
export MOON_DATA=${MOON_DATA:-$Y/dataset/Moon} MOON_WEIGHTS=${MOON_WEIGHTS:-$Y/weights}
mkdir -p runs/E46/ckpt/_calib
ok=fail
$WB -m workbench.launch begin runs/E46 calib -- bash runs/E46/code/calib.sh
trap '$WB -m workbench.launch end runs/E46 calib $ok' EXIT
/opt/envs/loftr/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 -m finetune.train \
    runs/E46/configs/_calib.toml --out runs/E46/ckpt/_calib
ok=ok
