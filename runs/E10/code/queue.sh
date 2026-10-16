#!/usr/bin/env bash
# E10 的推理队列（#150）：每张卡起一个，靠 mkdir runs/E10/.claims/<方法名> 占位，跨 gpfs 各机共享。在本票 worktree 根目录运行。
#
#   GPU=<空卡> KIND=loftr|roma nohup bash runs/E10/code/queue.sh > runs/E10/ckpt/queue_<host>_g<GPU>.log 2>&1 < /dev/null &
#
# RoMa 的结果随显卡型号变（同一权重在 V100 与 3090 上差约 3 px，#100），所以 RoMa 全放 V100；参照 s1 也在同一批卡上重跑。
# 融合方法排在最后，等它的两个来源都有 preds 才领。让卡：touch runs/E10/.stop_g<GPU>（或 runs/E10/.stop）。
set -uo pipefail
R=runs/E10
mkdir -p $R/.claims $R/ckpt
jobs=()
for f in $R/code/cfg/*${KIND:-}*.json; do jobs+=("$(basename "$f" .json)"); done
case ${KIND:-} in loftr) jobs+=(anymatch_loftr__fuse_s1_cpr);; roma) jobs+=(anymatch_roma__fuse_s1_cpr);; esac
for name in "${jobs[@]}"; do
  [ -e $R/.stop ] || [ -e $R/.stop_g$GPU ] && { echo "$(date +%T) stop file, exit"; exit 0; }
  if [[ $name == *__fuse_s1_cpr ]]; then
    m=${name%__fuse_s1_cpr}
    [ -f $R/preds/${m}__s1/val.jsonl ] && [ -f $R/preds/${m}__cpr/val.jsonl ] || { echo "$(date +%T) WAIT $name（来源未完成，跳过）"; continue; }
  fi
  mkdir $R/.claims/$name 2>/dev/null || continue
  echo "$(hostname) gpu$GPU $(date '+%F %T %z')" > $R/.claims/$name/host
  echo "$(date +%T) START $name"
  if bash $R/code/one.sh "$name" >> $R/ckpt/$name.log 2>&1; then
    echo "$(date +%T) OK $name"
  else
    echo "$(date +%T) FAIL $name（见 $R/ckpt/$name.log）"
  fi
done
echo "$(date +%T) queue empty"
