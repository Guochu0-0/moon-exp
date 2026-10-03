#!/bin/bash
# 已停用（#76）：产物写到 results/finetune/，不符合 docs/agents/experiments.md。新实验用 scripts/finetune/run.py；本文件留待「历史代码与结果整理」（#77）归位。
# RoMa 系微调 + 逐 ckpt 在 Val 上评测（「【RoMa】微调代码接入与实测」#64），照 scenes.sh。
# usage: GPU=<id> roma.sh <name> [finetune.train_roma 的参数...]
# 产物在 $MOON_RESULTS/finetune/<name>/：训练日志、ckpt、match/、sweep/S/metrics.json（各 step 的 Val AUC@5）。
# 评测走 baseline 原链路（560 → 864、symmetric、采 10000 点），配置 anymatch_roma__minmax。
set -e
cd "${REPO:-/remote-home/xufang/YGC/moon-exp-roma}"
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights \
       MOON_RESULTS=/remote-home/xufang/YGC/results TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home \
       CUDA_DEVICE_ORDER=PCI_BUS_ID
name=$1; shift
R=$MOON_RESULTS/finetune/$name
CFG=${CFG:-configs/baselines/anymatch_roma__minmax.json}
PY=/opt/envs/loftr/bin/python WB=/opt/envs/wb/bin/python
run() { CUDA_VISIBLE_DEVICES=$GPU nice -n 10 ionice -c3 "$@"; }
mkdir -p $R

if [ ! -f $R/train.done ]; then
  echo "$(date +%T) TRAIN $*"
  run $PY -m finetune.train_roma $CFG --out $R "$@" > $R/train.log 2>&1 && touch $R/train.done
fi
echo "$(date +%T) SWEEP"
[ -d $R/sweep/S ] || $WB -m workbench --runs $R/sweep new S --title "$name ckpt sweep"
for c in $(ls $R/ckpt_*.pt | sort -t_ -k2 -n); do
  st=$(basename $c .pt); st=${st#ckpt_}
  [ -f $R/sweep/S/preds/step$st/val.jsonl ] && continue
  run $PY -m baselines.match $CFG --split val --weights $c --name step$st --out $R/match >> $R/sweep.log 2>&1
  nice -n 10 $WB -m baselines.fit $R/match/step$st --split val --run $R/sweep/S >> $R/sweep.log 2>&1
done
nice -n 10 $WB -m workbench --runs $R/sweep eval S >> $R/sweep.log 2>&1
echo "$(date +%T) DONE $name"
