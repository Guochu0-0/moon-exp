# 等 126 GPU3 上的 round2-126.txt 队列退出，再在 GPU3 上跑 round2-126only.txt。mkdir 锁防重复。
mkdir /remote-home/xufang/YGC/results/finetune/_chain126.lock 2>/dev/null || { echo "already started"; exit 0; }
cd /remote-home/xufang/YGC/moon-exp-r2
nohup setsid bash -c 'while pgrep -f "[f]irst_round.py scripts/finetune/jobs/round2-126.txt" >/dev/null; do sleep 60; done; GPU=3 /opt/envs/loftr/bin/python scripts/finetune/first_round.py scripts/finetune/jobs/round2-126only.txt' >> /remote-home/xufang/YGC/results/finetune/_queue_r2_126only_g3.log 2>&1 < /dev/null &
echo chained
