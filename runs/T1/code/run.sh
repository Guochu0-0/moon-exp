#!/usr/bin/env bash
# T1：参照配方（M15）三个种子在 Val 峰值处的 ckpt 逐参数平均（Model soups 2203.05482 的均匀汤与贪心汤）。
# 贪心顺序按各种子的 Val AUC@5 峰值：M15s2（0.3188）> M15s1（0.3160）> M15（0.3156）。
#   main__top2：前两个的平均；main：三个的平均。贪心汤取两者中 Val 不降的那个（见 notes）。
#   GPU=<空卡> bash runs/T1/code/run.sh      （在 worktree 根目录）
set -euo pipefail
R=runs/T1; CK=$R/ckpt; MC=/remote-home/xufang/YGC/moon-exp/runs/M/ckpt
PY=/opt/envs/loftr/bin/python
mkdir -p $CK
[ -f $CK/main__top2.pt ] || $PY scripts/finetune/merge.py avg $CK/main__top2.pt $MC/M15s2/ckpt_4000.pt $MC/M15s1/ckpt_8000.pt
[ -f $CK/main.pt ] || $PY scripts/finetune/merge.py avg $CK/main.pt $MC/M15s2/ckpt_4000.pt $MC/M15s1/ckpt_8000.pt $MC/M15/ckpt_7000.pt
for m in main__top2 main; do bash scripts/finetune/eval_ckpt.sh $R $m $CK/$m.pt; done
