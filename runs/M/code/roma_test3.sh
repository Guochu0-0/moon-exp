cd /remote-home/xufang/YGC/moon-exp-roma
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
F=/remote-home/xufang/YGC/results/finetune; CFG=configs/baselines/anymatch_roma__minmax.json
for p in M13:8000 M14:8000 M4s1:4000; do x=${p%:*}; st=${p#*:}; R=$F/$x
  /opt/envs/loftr/bin/python -m baselines.match $CFG --split test --weights $R/ckpt_$st.pt --name step$st --out $R/match --resume >> $R/test.log 2>&1
  /opt/envs/wb/bin/python -m baselines.fit $R/match/step$st --split test --run $R/sweep/S >> $R/test.log 2>&1
  /opt/envs/wb/bin/python -m workbench --runs $R/sweep eval S >> $R/test.log 2>&1
done
