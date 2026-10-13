#!/bin/bash
# 检验表里需要重新推理的部分（占一张卡）。在本票 worktree 根目录、代码已提交后跑：
#   GPU=<卡> nohup bash runs/O0/code/infer.sh val   > runs/O0/raw/infer_val.log   2>&1 < /dev/null &
#   GPU=<卡> SHARDS=4 nohup bash runs/O0/code/infer.sh train > runs/O0/raw/infer_train.log 2>&1 < /dev/null &
# val：两个老师（t0 发布权重、t1 runs/N 的 erase__r1 第 7000 步）在 Val 上各多打两次标：把 SAR 当第一张图（swap）、
#      推理分辨率由 560 / 864 改成 448 / 672（res448）。用来算「多次打标是否一致」这两个度量。
# train：发布权重在 Train 上重新推理一遍（当时的原始点对没有留下，只留了标签），同卡分 SHARDS 份并行。
# 原始点对写到 runs/O0/raw/match/，每对的度量写到 runs/O0/raw/。
set -euo pipefail
mode=${1:?val 或 train}
Y=/remote-home/xufang/YGC
N=$Y/moon-exp/runs/N
R=runs/O0
PY=/opt/envs/loftr/bin/python
export TORCH_HOME=$Y/weights/torch_home MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:?GPU 未设}
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "工作区有未提交的改动" >&2; exit 1; }
mkdir -p $R/raw/match
mkdir $R/raw/infer_$mode.lock || { echo "已经在跑或跑过（$R/raw/infer_$mode.lock 存在）" >&2; exit 1; }
echo "$(date '+%F %T %z') $mode commit $(git rev-parse HEAD) host $(hostname) gpu $GPU"
B=configs/baselines/anymatch_roma__minmax
if [ $mode = val ]; then
    for v in swap res448; do
        $PY -m baselines.match ${B}_$v.json --split val --name t0_$v --out $R/raw/match
        $PY -m baselines.match ${B}_$v.json --split val --weights $N/ckpt/erase__r1/ckpt_7000.pt --name t1_$v --out $R/raw/match
        for t in t0 t1; do
            CUDA_VISIBLE_DEVICES= $PY $R/code/pair_metrics.py $R/raw/match/${t}_$v/val.npz --out $R/raw/val_${t}_$v.jsonl
        done
    done
else
    n=${SHARDS:-4}
    for k in $(seq 0 $((n - 1))); do
        $PY -m baselines.match $B.json --split train --name t0_s$k --shard $k/$n --resume --out $R/raw/match \
            > $R/raw/match_t0_s$k.log 2>&1 &
    done
    wait
    CUDA_VISIBLE_DEVICES= $PY $R/code/pair_metrics.py $(for k in $(seq 0 $((n - 1))); do echo $R/raw/match/t0_s$k/train.npz; done) \
        --out $R/raw/train_t0.jsonl
fi
echo ALL DONE
