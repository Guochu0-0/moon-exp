#!/bin/bash
# E39（#240）：交换光学和 SAR 的输入顺序再推理一次，点对放 runs/E39/raw/（不进 git），由 swap_fit.py 估反向仿射并求逆。
# 在本票 worktree 根目录运行：GPU=<空卡> CK=<E37 all 的 ckpt_7907.pt> bash runs/E39/code/run_swap.sh zero_shot|e37
set -euo pipefail
M=$1
Y=/remote-home/xufang/YGC
PY=/opt/envs/loftr/bin/python; WB=/opt/envs/wb/bin/python
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home PYTHONPATH=.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
ok=fail
$WB -m workbench.launch begin runs/E39 swap_$M -- bash runs/E39/code/run_swap.sh "$@"
trap '$WB -m workbench.launch end runs/E39 swap_$M $ok' EXIT
EXTRA=()
[ "$M" = e37 ] && EXTRA+=(--weights "$CK")
nice -n 10 $PY -m baselines.match configs/baselines/anymatch_roma__minmax_swap.json --split val "${EXTRA[@]}" --name "swap_$M" --out runs/E39/raw
$WB runs/E39/code/swap_fit.py "$M"
ok=ok
