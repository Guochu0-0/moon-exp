cd /remote-home/xufang/YGC/moon-exp-roma
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/remote-home/xufang/YGC/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID
F=/remote-home/xufang/YGC/results/finetune
CFG=configs/baselines/anymatch_roma__minmax.json
PY=/opt/envs/loftr/bin/python WB=/opt/envs/wb/bin/python
testeval() {  # gpu name step
  R=$F/$2
  CUDA_VISIBLE_DEVICES=$1 $PY -m baselines.match $CFG --split test --weights $R/ckpt_$3.pt --name step$3 --out $R/match --resume >> $R/test.log 2>&1
  $WB -m baselines.fit $R/match/step$3 --split test --run $R/sweep/S >> $R/test.log 2>&1
}
shard() {  # gpu k
  CUDA_VISIBLE_DEVICES=$1 $PY -m baselines.match $CFG --split train --weights $F/M3/ckpt_2000.pt --name s$2 --shard $2/3 --resume --out $F/_relabel_M3 > $F/_relabel_M3_s$2.log 2>&1
}
(testeval 1 M1 5000; shard 1 0) &
(testeval 2 M3 2000; shard 2 1) &
(shard 3 2) &
wait
for x in M1 M3; do $WB -m workbench --runs $F/$x/sweep eval S >> $F/$x/test.log 2>&1; done
echo done > $F/_roma_126a.done
