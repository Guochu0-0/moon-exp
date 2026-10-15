#!/usr/bin/env bash
# E9 的一个方法：Val 上 zero-shot 推理 → RANSAC 仿射 → preds，写启动记录（#150）。在本票 worktree 根目录运行。
#
#   GPU=<空卡> bash runs/E9/code/one.sh <方法名>
#
# 方法名为 code/cfg/<方法名>.json；<模型>__fuse_s1_cpr 不推理，合并 <模型>__s1 与 <模型>__cpr 的点对后估仿射（fuse.py）。
# 原始点对写在 runs/E9/raw/<方法名>/（不进 git）。
set -euo pipefail
R=runs/E9; M=$1
Y=/remote-home/xufang/YGC
PY=/opt/envs/loftr/bin/python; WB=/opt/envs/wb/bin/python
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-}
$WB -m workbench.launch begin "$R" "$M" -- bash runs/E9/code/one.sh "$@"
ok=fail
trap '$WB -m workbench.launch end "$R" "$M" $ok' EXIT
case $M in
  *__fuse_s1_cpr)
    m=${M%__fuse_s1_cpr}
    $WB $R/code/fuse.py $R/raw/${m}__s1 $R/raw/${m}__cpr --out $R/raw/$M --split val ;;
  *)
    nice -n 10 $PY -m baselines.match $R/code/cfg/$M.json --split val --out $R/raw ;;
esac
$WB -m baselines.fit $R/raw/$M --split val --run $R
ok=ok
