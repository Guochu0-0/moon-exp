cd /remote-home/xufang/YGC/moon-exp-roma
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID
F=/remote-home/xufang/YGC/results/finetune
CFG=configs/baselines/anymatch_roma__minmax.json
PY=/opt/envs/loftr/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m finetune.train_roma $CFG --out $F/_smoke_roma_ps --labels $F/labels_p2.jsonl --label-top 0.5 --aug geo,photo --steps 60 --save-every 0 --save0 --workers 2 > $F/_smoke_roma_ps.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -m finetune.train_roma $CFG --out $F/_smoke_roma_rl --w-pseudo 0 --w-ripe 1 --neg --steps 60 --save-every 0 --workers 2 > $F/_smoke_roma_rl.log 2>&1 &
wait
