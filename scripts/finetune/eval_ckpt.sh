#!/usr/bin/env bash
# 把一个现成的 RoMa / LoFTR ckpt 当作实验 <run> 的方法 <method> 评测（Val、Test），写启动记录与 preds（#120）。
# 用于不训练、只合并或重估权重的方法（scripts/finetune/merge.py）；训练出来的方法走 scripts/finetune/run.py。
#
#   GPU=<空卡> bash scripts/finetune/eval_ckpt.sh runs/<id> <method> <ckpt> [推理配置]
#
# 在本票 worktree 的根目录运行。原始点对写在 runs/<id>/sweep/match/<method>/（不进 git）。
set -euo pipefail
R=$1; M=$2; CK=$3; CFG=${4:-configs/baselines/anymatch_roma__minmax.json}
Y=/remote-home/xufang/YGC
PY=/opt/envs/loftr/bin/python; WB=/opt/envs/wb/bin/python
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
$WB -m workbench.launch begin "$R" "$M" -- bash scripts/finetune/eval_ckpt.sh "$@"
ok=fail
trap '$WB -m workbench.launch end "$R" "$M" $ok' EXIT
for split in val test; do
  if [ ! -f "$R/preds/$M/$split.jsonl" ]; then
    nice -n 10 $PY -m baselines.match "$CFG" --split $split --weights "$CK" --name "$M" --out "$R/sweep/match"
    $WB -m baselines.fit "$R/sweep/match/$M" --split $split --run "$R"
  fi
done
ok=ok
