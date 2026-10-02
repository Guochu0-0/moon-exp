# 一次性：停掉 GPU7 上所有第二轮队列，清 Q8/Q9，重开一个队列。mkdir 锁保证重复执行无副作用。
mkdir /tmp/r2fix7_done 2>/dev/null || { echo "already done"; exit 0; }
R=/remote-home/xufang/YGC/results/finetune
for p in $(pgrep -f "[f]irst_round.py scripts/finetune/jobs/round2-a6000.txt"); do
  g=$(tr "\0" "\n" < /proc/$p/environ | grep "^GPU=" | cut -d= -f2)
  [ "$g" = 7 ] || continue
  for k in $(pgrep -P $p); do for kk in $(pgrep -P $k); do pkill -TERM -P $kk; kill -TERM $kk; done; pkill -TERM -P $k; kill -TERM $k; done
  kill -TERM $p; echo "stopped $p"
done
sleep 15
pgrep -fa "[f]inetune.train.*finetune/Q[89] " && { echo "still running, abort"; exit 1; }
rm -rf $R/Q8 $R/Q9 $R/_claims/Q8 $R/_claims/Q9
cd /remote-home/xufang/YGC/moon-exp-r2 && GPU=7 nohup setsid /opt/envs/loftr/bin/python scripts/finetune/first_round.py scripts/finetune/jobs/round2-a6000.txt >> $R/_queue_r2_a6k_g7.log 2>&1 < /dev/null &
echo relaunched
