#!/bin/bash
cd /remote-home/xufang/YGC/moon-exp-ft26
F=/remote-home/xufang/YGC/results/finetune
GPU=1 nohup setsid bash scripts/finetune/scenes.sh S1 --labels $F/labels_b0.jsonl --steps 8000 --lr 1e-5 --save-every 1000 > $F/S1.driver.log 2>&1 < /dev/null &
echo "S1 pid $!"
GPU=2 nohup setsid bash scripts/finetune/scenes.sh S2 --steps 8000 --lr 1e-5 --save-every 1000 > $F/S2.driver.log 2>&1 < /dev/null &
echo "S2 pid $!"
