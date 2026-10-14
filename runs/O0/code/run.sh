#!/bin/bash
# 检验表的全部离线计算（只用 CPU）。在本票 worktree 根目录、代码已提交后跑一次：
#   nohup bash runs/O0/code/run.sh > runs/O0/raw/run.log 2>&1 < /dev/null &
# 老师 t0 = 发布的 AnyMatch-RoMa 权重；t1 = 上一批「SAR 随机擦除，第 1 轮」第 7000 步的学生（runs/N 的 erase__r1）。
# 每对的度量写到 runs/O0/raw/（不进 git），表写到 runs/O0/extra/data/。
set -euo pipefail
PY=/opt/envs/wb/bin/python
Y=/remote-home/xufang/YGC
N=$Y/moon-exp/runs/N
R=runs/O0
export MOON_DATA=$Y/dataset/Moon
mkdir -p $R/raw $R/extra/data
mkdir $R/raw/run.lock || { echo "已经在跑或跑过（$R/raw/run.lock 存在）" >&2; exit 1; }
$PY -m workbench.launch begin $R main -- bash $R/code/run.sh
trap '$PY -m workbench.launch end $R main fail' ERR

$PY $R/code/pair_metrics.py $Y/results/baselines_ablation/anymatch_roma__minmax/val.npz --out $R/raw/val_t0.jsonl
$PY $R/code/pair_metrics.py $N/sweep/erase__r1/match/step7000/val.npz --out $R/raw/val_t1.jsonl
$PY $R/code/table.py val $R/raw/val_t0.jsonl --data $MOON_DATA --alt gen=$R/raw/val_t1.jsonl --out $R/extra/data/val_t0.csv
$PY $R/code/table.py val $R/raw/val_t1.jsonl --data $MOON_DATA --alt gen=$R/raw/val_t0.jsonl --out $R/extra/data/val_t1.csv

$PY $R/code/pair_metrics.py $N/raw/relabel/erase__r1/s0/train.npz --out $R/raw/train_t1.jsonl
$PY $R/code/table.py train $R/raw/train_t1.jsonl --alt gen=$Y/results/finetune/labels_roma0.jsonl --out $R/extra/data/train_t1

$PY -m workbench.launch end $R main ok
echo ALL DONE
