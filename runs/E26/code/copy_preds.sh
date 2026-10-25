#!/bin/bash
# 把各存档在验证集上的正式评测（训练驱动逐存档评测时写的 sweep 记录）复制成 E26 的 preds/<方法>/val.*，供工作台评测、
# 逐对查看点对与中间结果（#206）。zs 取 B0m 的 anymatch_roma__minmax（同一套推理配置与 SAR 拉伸）。不重跑推理。
# 在 worktree 根目录：bash runs/E26/code/copy_preds.sh
set -euo pipefail
MAIN=/remote-home/xufang/YGC/moon-exp/runs
R=runs/E26
methods=$(/opt/envs/loftr/bin/python -c "import sys; sys.path.insert(0, '$R/code'); import probe; print(' '.join(probe.METHODS))")
for m in $methods; do
  d=$R/preds/$m
  mkdir -p $d
  if [ $m = zs ]; then
    cp $MAIN/B0m/preds/anymatch_roma__minmax/val.jsonl $MAIN/B0m/preds/anymatch_roma__minmax/val.meta.json $d/
    cp $MAIN/B0m/preds/anymatch_roma__minmax/val_matches.npz $d/
    continue
  fi
  e=${m%%_*}; e=${e^^}; rest=${m#*_}; step=${rest##*_}; run=${rest%_*}
  cp $MAIN/$e/sweep/$run/S/preds/step$step/val.jsonl $MAIN/$e/sweep/$run/S/preds/step$step/val.meta.json $d/
  cp $MAIN/$e/sweep/$run/match/step$step/val_matches.npz $d/
done
ls $R/preds | wc -l
