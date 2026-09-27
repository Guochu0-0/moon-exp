#!/bin/bash
# SCENES 式伪标签微调 + 逐 ckpt 在 Val 上评测（「复现 SCENES 式伪标签微调 baseline」#26）。
# usage: GPU=<id> scenes.sh <name> [finetune.train 的参数...]
#   离线伪标签（SCENES 做法）：scenes.sh S1 --labels /remote-home/xufang/YGC/results/finetune/labels_b0.jsonl
#   在线伪标签：              scenes.sh S2
# 产物在 $MOON_RESULTS/finetune/<name>/：训练日志、ckpt、match/（原始点对）、sweep/（逐 ckpt 的工作台记录与 metrics.json）。
# 选模看 sweep/S/metrics.json 里各 step 的 Val AUC@5；选中的再按 baseline 流程写进 runs/。
set -e
cd "${REPO:-/remote-home/xufang/YGC/moon-exp-ft26}"
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights \
       MOON_RESULTS=/remote-home/xufang/YGC/results TORCH_HOME=/opt/torch_home
name=$1; shift
R=$MOON_RESULTS/finetune/$name
L=$MOON_RESULTS/finetune/labels_b0.jsonl
CFG=configs/baselines/anymatch_loftr.json
PY=/opt/envs/loftr/bin/python WB=/opt/envs/wb/bin/python
run() { CUDA_VISIBLE_DEVICES=$GPU nice -n 10 ionice -c3 "$@"; }
mkdir -p $R

if [[ " $* " == *" --labels "* ]] && [ ! -f $L ]; then
  echo "$(date +%T) LABEL"
  run $PY -m finetune.label $CFG --out $L.tmp > $R/label.log 2>&1 && mv $L.tmp $L
fi
if [ ! -f $R/train.done ]; then
  echo "$(date +%T) TRAIN $*"
  run $PY -m finetune.train $CFG --out $R "$@" > $R/train.log 2>&1 && touch $R/train.done
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
