#!/bin/bash
# runs/O 的开跑入口（#131），在本票 worktree 根目录、代码已提交后，在 150 上跑一次：
#   bash runs/O/code/start.sh
# 做三件事：给要等老师的学生建占位（held）；生成 cur__r2 的选对结果；起无人值守调度 code/auto.sh。
# 三条链：stpp__r2（一步放开，不用等）；cl__r1 → cl__r2 → cl__r3（逐轮放宽）；cur__r2 → cur__r3（训完清理）。
set -euo pipefail
R=runs/O
N=/remote-home/xufang/YGC/moon-exp/runs/N/raw
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "工作区有未提交的改动" >&2; exit 1; }
mkdir -p $R/raw $R/ckpt $R/.claims
mkdir $R/raw/start.lock || { echo "已经起过（$R/raw/start.lock 存在）" >&2; exit 1; }
for s in cl__r2 cl__r3 cur__r3; do
    mkdir $R/.claims/$s
    echo "等老师的标签，由 code/auto.sh 放行" > $R/.claims/$s/held
done
bash $R/code/prep.sh erase__r1 | tee $R/raw/prep_erase__r1.out
SHARDS=${SHARDS:-4} nohup bash $R/code/auto.sh main \
    cl__r1:cl__r2 cl__r2:cl__r3:$R/raw/labels_cl__r1.jsonl cur__r2:cur__r3:$N/labels_erase__r1.jsonl \
    > $R/raw/auto.out 2>&1 < /dev/null &
echo "auto.sh pid $!"
