#!/usr/bin/env bash
# E7 难版评测的推理队列（#138）：每个方法 × （原始 + 9 个扰动变体），各跑 Val、Test，只推理不训练。
# 一个任务 = 一个方法名（<方法> 或 <方法>__<变体>），由 scripts/finetune/eval_ckpt.sh 推理、估仿射、写 preds 与启动记录。
#
#   GPU=<空卡> KIND=loftr|roma nohup bash runs/E7/code/queue.sh > runs/E7/ckpt/queue_<host>_g<GPU>.log 2>&1 < /dev/null &
#
# 在本票 worktree 的根目录运行，每张卡起一个；靠 mkdir runs/E7/.claims/<方法名> 占位，跨 gpfs 各机共享。
# KIND 只领 LoFTR 或 RoMa 的任务：RoMa 的结果随显卡型号变（同一 ckpt 在 V100 与 3090 上差约 3 px，#100），所以 RoMa 全放 V100，LoFTR 放 3090。
# 让卡：touch runs/E7/.stop_g<GPU>（或 runs/E7/.stop 停全部），当前任务跑完后不再领新的。
set -uo pipefail
R=runs/E7
M=/remote-home/xufang/YGC/moon-exp/runs        # ckpt 在 gpfs 主 checkout（结票时由 wt.sh close 挪过去的）
LOFTR=configs/baselines/anymatch_loftr.json
# 方法  ckpt（- 为原始权重）  推理配置（与各方法原来评测时相同）
JOBS=(
  "anymatch_loftr  -                                     $LOFTR"
  "anymatch_roma   -                                     configs/baselines/anymatch_roma.json"
  "loftr_lr5e5     $M/E1/ckpt/loftr_lr5e5/ckpt_14000.pt  $LOFTR"
  "roma_vgg        $M/E1/ckpt/roma_vgg/ckpt_16000.pt     configs/baselines/anymatch_roma__minmax.json"
  "P10             $M/P/ckpt/P10/ckpt_7000.pt            $LOFTR"
  "Q4              $M/Q/ckpt/Q4/ckpt_1500.pt             $LOFTR"
  "Q11p            $M/Q/ckpt/Q11p/ckpt_4000.pt           $LOFTR"
  "erase_geo__r2   $M/N/ckpt/erase_geo__r2/ckpt_7000.pt  configs/baselines/anymatch_roma__minmax.json"
)
VARIANTS=("" shift12 shift64 shift128 rot3 rot10 rot20 scale3 scale10 scale20)

mkdir -p $R/.claims $R/ckpt
for v in "${VARIANTS[@]}"; do
  for job in "${JOBS[@]}"; do
    read -r m ck cfg <<< "$job"
    case "${KIND:-all}:$cfg" in loftr:*roma*|roma:*loftr*) continue;; esac
    name=$m${v:+__$v}
    [ -e $R/.stop ] || [ -e $R/.stop_g$GPU ] && { echo "$(date +%T) stop file, exit"; exit 0; }
    mkdir $R/.claims/$name 2>/dev/null || continue
    echo "$(hostname) gpu$GPU $(date '+%F %T %z')" > $R/.claims/$name/host
    echo "$(date +%T) START $name"
    if bash scripts/finetune/eval_ckpt.sh $R "$name" "$ck" "$cfg" "$v" >> $R/ckpt/$name.log 2>&1; then
      echo "$(date +%T) OK $name"
    else
      echo "$(date +%T) FAIL $name（见 $R/ckpt/$name.log）"
    fi
  done
done
echo "$(date +%T) queue empty"
