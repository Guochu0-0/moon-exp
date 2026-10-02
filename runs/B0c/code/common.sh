Y=/remote-home/xufang/YGC
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights MOON_RESULTS=$Y/results
export CUDA_DEVICE_ORDER=PCI_BUS_ID
D=$Y/moon-exp-cf70
PY=/opt/envs/loftr/bin/python
CFG=configs/baselines/anymatch_loftr_coarse.json
F=$MOON_RESULTS/finetune
# 方法名 ckpt（空=配置里的 zero-shot 权重）
JOBS="anymatch_loftr: main:$F/S1/ckpt_6000.pt C2:$F/C2/ckpt_500.pt Q4:$F/Q4/ckpt_1500.pt Q4s1:$F/Q4s1/ckpt_1500.pt P8:$F/P8/ckpt_8000.pt"
