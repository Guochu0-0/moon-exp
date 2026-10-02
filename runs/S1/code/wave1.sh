cd /remote-home/xufang/YGC/moon-exp-ft26
F=/remote-home/xufang/YGC/results/finetune
S1=$F/S1/ckpt_6000.pt
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/opt/torch_home
GPU=2 nohup setsid bash scripts/finetune/scenes.sh S1c --init $S1 --labels $F/labels_b0.jsonl --steps 4000 --lr 1e-5 --save-every 1000 > $F/S1c.driver.log 2>&1 < /dev/null &
GPU=3 nohup setsid bash scripts/finetune/scenes.sh S1lr3 --labels $F/labels_b0.jsonl --steps 4000 --lr 3e-5 --save-every 1000 > $F/S1lr3.driver.log 2>&1 < /dev/null &
nohup setsid bash -c "CUDA_VISIBLE_DEVICES=1 nice -n 10 ionice -c3 /opt/envs/loftr/bin/python -m finetune.label configs/baselines/anymatch_loftr.json --init $S1 --out $F/labels_s1.jsonl > $F/labels_s1.log 2>&1 && GPU=1 bash scripts/finetune/scenes.sh S1r2 --init $S1 --labels $F/labels_s1.jsonl --steps 4000 --lr 1e-5 --save-every 1000" > $F/S1r2.driver.log 2>&1 < /dev/null &
