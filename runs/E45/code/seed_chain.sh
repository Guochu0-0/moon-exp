#!/bin/bash
# E45 / E41 门槛 0.8 补种子（#264）：在 160 的 4 张卡上按顺序接力训练，种子 1 两边都训完再训种子 2。
# 先等已手工起的 E45 main__seed1 训练进程退出，再依次起 E41 main__screen0.8__seed1、E45 main__seed2、E41 main__screen0.8__seed2；
# 每次起之前等 4 张卡上没有任何进程。某个方法已被别处（如 150）领走时，train_queue.py 见到 .claims 直接退出，接着下一个。
# 评测由各实验的 eval_queue.py 在别的卡上接手。
#   nohup bash runs/E45/code/seed_chain.sh > runs/E45/ckpt/seed_chain_160.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.."
PY=/opt/envs/loftr/bin/python
log() { echo "$(date -u '+%F %T UTC') $*"; }
while pgrep -f "runs/E45/code/train_queu[e].py" > /dev/null; do sleep 300; done
for job in E41:main__screen0.8__seed1 E45:main__seed2 E41:main__screen0.8__seed2; do
    E=${job%%:*}; M=${job#*:}
    while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; do sleep 300; done
    log "start $E/$M"
    GPU=0,1,2,3 METHODS=$M $PY runs/$E/code/train_queue.py >> runs/$E/ckpt/train_160_g0123_$M.log 2>&1 < /dev/null
    log "exit $E/$M rc=$?"
done
log "chain done"
