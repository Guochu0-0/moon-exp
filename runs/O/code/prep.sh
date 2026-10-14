#!/bin/bash
# 老师 <t> 的标签写出后，生成以它为老师的学生要用的选对结果（只用 CPU）。只按内点数取前若干比例的学生不需要这一步
# （配置里的 top 直接完成）。用法（在本票 worktree 根目录）：bash runs/O/code/prep.sh <老师>
#   erase__r1 是 runs/N 里底座的第 1 轮，它的标签早已打好；开跑前手工跑一次 prep.sh erase__r1。
set -euo pipefail
t=${1:?老师}
Y=/remote-home/xufang/YGC
N=$Y/moon-exp/runs/N/raw
R=runs/O
PY=/opt/envs/loftr/bin/python
mkdir -p $R/raw
case $t in
erase__r1)   # 训完清理的第 2 轮：新标签（erase__r1 打的）与它训练时用的标签（发布权重打的）角点位置相差超过 3 px 的对去掉
    $PY $R/code/pick_pairs.py $N/labels_erase__r1.jsonl --ref $Y/results/finetune/labels_roma0.jsonl --max-diff 3 \
        --out $R/raw/sel_cur__r2.jsonl ;;
cur__r2)     # 训完清理的第 3 轮：新标签（cur__r2 打的）与上一轮的标签（erase__r1 打的）相比
    $PY $R/code/pick_pairs.py $R/raw/labels_cur__r2.jsonl --ref $N/labels_erase__r1.jsonl --max-diff 3 \
        --out $R/raw/sel_cur__r3.jsonl ;;
*) echo "prep: $t 没有要生成的选对结果" ;;
esac
