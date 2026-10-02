cd /remote-home/xufang/YGC/moon-exp-ft26
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/opt/torch_home
R=/remote-home/xufang/YGC/results/finetune/S1
CUDA_VISIBLE_DEVICES=1 nice -n 10 ionice -c3 /opt/envs/loftr/bin/python -m baselines.match configs/baselines/anymatch_loftr.json --split test --weights $R/ckpt_6000.pt --name step6000 --out $R/match > $R/test.log 2>&1
for sp in val test; do nice -n 10 /opt/envs/wb/bin/python -m baselines.fit $R/match/step6000 --split $sp --run runs/S1 --name main >> $R/test.log 2>&1; done
echo DONE >> $R/test.log
