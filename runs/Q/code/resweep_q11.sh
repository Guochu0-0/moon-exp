mkdir /remote-home/xufang/YGC/results/finetune/_resweep_q11.lock 2>/dev/null || { echo "already started"; exit 0; }
cd /remote-home/xufang/YGC/moon-exp-r2
GPU=3 nohup setsid /opt/envs/loftr/bin/python -c "
import sys; sys.argv=['x']; sys.path.insert(0,'scripts/finetune')
import first_round as f
for n in ('Q11','Q11p'):
    try:
        f.job(n, ['--rl-pair'], 3); f.log('OK '+n)
    except Exception as e:
        f.log(f'FAIL {n}: {e}')
" >> /remote-home/xufang/YGC/results/finetune/_queue_r2_150_resweep.log 2>&1 < /dev/null &
echo started
