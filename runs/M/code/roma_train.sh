export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_VISIBLE_DEVICES=3
cd /remote-home/xufang/YGC/moon-exp-roma
for sp in test train; do /opt/envs/loftr/bin/python -m baselines.match configs/baselines/anymatch_roma__minmax.json --split $sp --resume --out /remote-home/xufang/YGC/results/finetune/_roma_train; done
