#!/bin/bash
# 第 2 轮的老师：用第 1 轮学生 <m>（run.py 按 Val AUC@5 峰值选出的那一步，同 M 第三批的做法）在干净的 Train 上推理，
# 按 finetune.label_from_raw 拟合伪仿射（RANSAC 3 px，匹配 ≥ 100、内点 ≥ 20），写到 runs/N/raw/labels_<m>.jsonl（不进 git），
# 再统计新旧标签前 50% 的重合比例，写到 runs/N/extra/overlap_<m>.json。
# 用法（在本票 worktree 根目录，代码已提交）：GPUS="1 2" bash runs/N/code/relabel.sh geo__r1 [旧标签，默认 zero-shot]
# GPUS 给几张卡就把 Train 分几份并行推理。
set -euo pipefail
m=${1:?方法名}
old=${2:-/remote-home/xufang/YGC/results/finetune/labels_roma0.jsonl}
gpus=(${GPUS:?GPUS 未设})
n=${#gpus[@]}
R=runs/N
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "工作区有未提交的改动" >&2; exit 1; }
step=$(/opt/envs/wb/bin/python -c "import json; print(json.load(open('$R/sweep/$m/peak.json'))['step'])")
ckpt=$R/ckpt/$m/ckpt_$step.pt
out=$R/raw/relabel/$m
mkdir -p $out $R/extra
export TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home MOON_DATA=/remote-home/xufang/YGC/dataset/Moon
export MOON_WEIGHTS=/remote-home/xufang/YGC/weights CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "$(date '+%F %T %z') $m step $step commit $(git rev-parse HEAD) host $(hostname) gpus ${gpus[*]}" >> $out/relabel.log
for k in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${gpus[$k]} nice -n 10 /opt/envs/loftr/bin/python -m baselines.match \
        configs/baselines/anymatch_roma__minmax.json --split train --weights $ckpt --name s$k --shard $k/$n \
        --resume --out $out >> $out/s$k.log 2>&1 &
done
wait
for k in $(seq 0 $((n - 1))); do
    /opt/envs/wb/bin/python -m finetune.label_from_raw $out/s$k/train.npz --out $out/s$k.labels.jsonl >> $out/relabel.log
done
cat $(for k in $(seq 0 $((n - 1))); do echo $out/s$k.labels.jsonl; done) > $R/raw/labels_$m.jsonl.tmp
mv $R/raw/labels_$m.jsonl.tmp $R/raw/labels_$m.jsonl
/opt/envs/wb/bin/python $R/code/overlap.py $old $R/raw/labels_$m.jsonl --teacher "$m step $step" \
    --json $R/extra/overlap_$m.json | tee -a $out/relabel.log
