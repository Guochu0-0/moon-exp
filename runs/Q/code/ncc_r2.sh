set -e
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon
cd /remote-home/xufang/YGC/moon-exp-r2
for x in Q4:step1500 Q4s1:step1500 P8:step8000 P4s1:step5000 Q7s1:step3500; do n=${x%%:*}; s=${x##*:}
  nice -n 10 /opt/envs/wb/bin/python -m baselines.diagnose_reward --pred /remote-home/xufang/YGC/results/finetune/$n/sweep/S/preds/$s/val.jsonl --out /remote-home/xufang/YGC/results/finetune/_ncc_r2/$n --workers 12 > /remote-home/xufang/YGC/results/finetune/_ncc_r2/$n.log 2>&1
done
echo ALLDONE
