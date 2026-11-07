#!/bin/bash
# E46（#254）：同模态自配对的准确度（code/selfpair_eval.py），依次评 zero-shot、E41 门槛 0.8 最后存档、本次训练的各存档。
# 已评完的（jsonl 已有全部行）续跑时自动跳过。启动记录 launch/selfpair.json。在 worktree 根目录运行：
#     GPU=<空卡> nohup bash runs/E46/code/selfpair.sh > runs/E46/ckpt/selfpair_<host>_g<GPU>.log 2>&1 < /dev/null &
set -euo pipefail
Y=/remote-home/xufang/YGC
PY=/opt/envs/loftr/bin/python; WB=/opt/envs/wb/bin/python
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home PYTHONPATH=.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
OUT=runs/E46/extra/data/selfpair
ok=fail
$WB -m workbench.launch begin runs/E46 selfpair -- bash runs/E46/code/selfpair.sh
trap '$WB -m workbench.launch end runs/E46 selfpair $ok' EXIT
$PY runs/E46/code/selfpair_eval.py --name zeroshot --out $OUT
$PY runs/E46/code/selfpair_eval.py --name e41s08_7907 --ckpt $Y/moon-exp/runs/E41/ckpt/main__screen0.8/ckpt_7907.pt --out $OUT
for c in $(ls runs/E46/ckpt/main/ckpt_*.pt | sort -t_ -k2 -n); do
    s=$(basename $c .pt); s=${s#ckpt_}
    $PY runs/E46/code/selfpair_eval.py --name e46_$s --ckpt $c --out $OUT
done
ok=ok
