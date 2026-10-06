#!/usr/bin/env bash
# 第一批（#120）的排队：一张卡一个进程，按给定顺序依次跑实验；训练实验交给 scripts/finetune/run.py（靠 .claims 占位，
# 多张卡跑同一串也不会重复），T1–T3 是零训练脚本（runs/T*/code/run.sh，产物已在就跳过）。
#   GPU=<空卡> [STOP_HOUR=5] nohup bash runs/T1/code/queue.sh T1 T2 T3 T9 ... > <log> 2>&1 < /dev/null &
# STOP_HOUR：给出时，本机时间到了这个钟点（0–23）之后不再开始新的实验（夜间多占的卡要在 08:00 前让出）。
set -uo pipefail
cd "$(dirname "$0")/../../.."
for id in "$@"; do
  if [ -n "${STOP_HOUR:-}" ] && [ "$(date +%H)" -ge "$STOP_HOUR" ] && [ "$(date +%H)" -lt 12 ]; then
    echo "$(date +%T) 到 $STOP_HOUR 点，不再开始 $id 及之后的实验"; break
  fi
  echo "$(date +%T) BEGIN $id gpu$GPU"
  if [ -f "runs/$id/code/run.sh" ]; then
    bash "runs/$id/code/run.sh" || echo "$(date +%T) FAIL $id"
  else
    /opt/envs/loftr/bin/python scripts/finetune/run.py "runs/$id" || echo "$(date +%T) FAIL $id"
  fi
done
echo "$(date +%T) queue done"
