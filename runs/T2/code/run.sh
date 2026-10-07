#!/usr/bin/env bash
# T2：WiSE-FT（2109.01903）：θ = (1−α)·zero-shot + α·参照配方，参照取 M15（种子 0，Val 峰值第 7000 步，Val AUC@5 0.3156）。
#   main：α = 0.5（原文推荐）；main__a025、main__a075 作参考，不用来挑。
#   GPU=<空卡> bash runs/T2/code/run.sh      （在 worktree 根目录）
set -euo pipefail
R=runs/T2; CK=$R/ckpt; MC=/remote-home/xufang/YGC/moon-exp/runs/M/ckpt
ZERO=/remote-home/xufang/YGC/weights/anymatch/RoMa_AnyMatch.pth
PY=/opt/envs/loftr/bin/python
mkdir -p $CK
for x in "main 0.5" "main__a025 0.25" "main__a075 0.75"; do
  set -- $x
  [ -f $CK/$1.pt ] || $PY scripts/finetune/merge.py wise $CK/$1.pt --zero $ZERO --ft $MC/M15/ckpt_7000.pt --alpha $2
done
for m in main main__a025 main__a075; do bash scripts/finetune/eval_ckpt.sh $R $m $CK/$m.pt; done
