#!/bin/bash
# E33 只用（#226）：多卡短跑，看各卡参数是否一致（ddp_sync.jsonl）与每次更新的耗时。GPUS 给出要用的卡（逗号分隔），
# 卡数即进程数，配置取 runs/E33/code/cfg/ddp<卡数>.toml，训 STEPS 次更新（默认 100）。
#   GPUS=0,1 nohup bash runs/E33/code/ddp_test.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
source runs/E33/code/env.sh
export CUDA_VISIBLE_DEVICES=$GPUS
n=$(echo $GPUS | tr ',' '\n' | wc -l)
d=runs/E33/ckpt/ddp/n$n
mkdir -p $d
$PY -m torch.distributed.run --standalone --nproc_per_node=$n -m finetune.train runs/E33/code/cfg/ddp$n.toml --out $d \
    --stop-at ${STEPS:-100} > $d/train.log 2>&1
echo "DDP_DONE n$n exit $?" >> runs/E33/ckpt/ddp/run.log
