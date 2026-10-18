#!/usr/bin/env bash
# E9 难版评测的推理队列（#157，照 runs/E7/code/queue.sh）：每条线最终的方法 × 9 个扰动变体，各跑 Val、Test，只推理不训练。
# ckpt 取驱动按原始 Val AUC@5 选出的峰值（runs/E9/sweep/<m>/peak.json）；原始档就是驱动写的 preds/<m>/。
# 一个任务 = 一个方法名 <m>__<变体>，由 scripts/finetune/eval_ckpt.sh 推理、估仿射、写 preds 与启动记录。
#
#   FINAL="lns_r2 rns_r2 q4g" GPU=<空卡> KIND=loftr|roma nohup bash runs/E9/code/queue.sh > runs/E9/ckpt/hq_<host>_g<GPU>.log 2>&1 < /dev/null &
#
# 在本票 worktree 的根目录运行，每张卡起一个；靠 mkdir runs/E9/.claims/<方法名> 占位，跨 gpfs 各机共享。
# KIND 只领 LoFTR 或 RoMa 的任务：同 E7，RoMa 全放 V100，LoFTR 放 RTX 3090。
# 让卡：touch runs/E9/.stop_g<GPU>（或 runs/E9/.stop 停全部），当前任务跑完后不再领新的。
set -uo pipefail
R=runs/E9
: "${FINAL:?FINAL 未设：各条线最终的方法名，空格分隔}"
VARIANTS=(shift12 shift64 shift128 rot3 rot10 rot20 scale3 scale10 scale20)

mkdir -p $R/.claims $R/ckpt
for v in "${VARIANTS[@]}"; do
  for m in $FINAL; do
    case $m in
      rns*) cfg=configs/baselines/anymatch_roma__minmax.json; kind=roma ;;
      *)    cfg=configs/baselines/anymatch_loftr.json; kind=loftr ;;
    esac
    [ "${KIND:-all}" = all ] || [ "$KIND" = $kind ] || continue
    step=$(/opt/envs/wb/bin/python -c "import json; print(json.load(open('$R/sweep/$m/peak.json'))['step'])") || continue
    name=${m}__$v
    [ -e $R/.stop ] || [ -e $R/.stop_g$GPU ] && { echo "$(date +%T) stop file, exit"; exit 0; }
    mkdir $R/.claims/$name 2>/dev/null || continue
    echo "$(hostname) gpu$GPU $(date '+%F %T %z')" > $R/.claims/$name/host
    echo "$(date +%T) START $name step$step"
    if bash scripts/finetune/eval_ckpt.sh $R "$name" "$R/ckpt/$m/ckpt_$step.pt" "$cfg" "$v" >> $R/ckpt/$name.log 2>&1; then
      echo "$(date +%T) OK $name"
    else
      echo "$(date +%T) FAIL $name（见 $R/ckpt/$name.log）"
    fi
  done
done
echo "$(date +%T) queue empty"
