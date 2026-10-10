#!/bin/bash
# runs/N 的无人值守调度（#130）：在 126 上常驻，每 5 分钟看一轮，日志写 runs/N/raw/auto.log。
#   1. 第 1 轮方法 <x>__r1 跑完（队列日志出现 OK）→ 在一张空卡上用它重新打标（code/relabel.sh）；
#   2. 标签与重合比例写出（extra/overlap_<x>__r1.json）→ 撤掉 <x>__r2 的占位（.claims/<x>__r2/held），放行第 2 轮；
#   3. 还有没被领走的方法、又有空卡 → 起一个 run.py 队列。
# 用卡（docs/agents/servers.md）：只用 126 的 GPU 1–3、只用没有任何进程的卡；自己同时最多占 2 张，
# 23:00–04:30 之间新起的任务可占到 3 张（3 小时内跑完，08:00 前让出）。打标失败不重试，记进日志。
# 第 2 轮全部出结果后退出。用法（在本票 worktree 根目录，代码已提交）：nohup bash runs/N/code/auto.sh &
set -uo pipefail
Y=/remote-home/xufang/YGC
R=runs/N
POOL="1 2 3"
NOISES="geo photo erase blockmask droppath loradrop vggdrop"
PY=/opt/envs/loftr/bin/python
mkdir $Y/n130_auto.lock || { echo "调度已在跑（锁 $Y/n130_auto.lock）"; exit 1; }
log() { echo "$(date '+%F %T') $*" >> $R/raw/auto.log; }

ours() {   # 我们在本机占着的 GPU
    for p in $(pgrep -f "scripts/finetune/run.py runs/N"); do tr '\0' '\n' < /proc/$p/environ | sed -n 's/^GPU=//p'; done
    for p in $(pgrep -f "runs/N/code/relabel.sh"); do tr '\0' '\n' < /proc/$p/environ | sed -n 's/^GPUS=//p' | tr ' ' '\n'; done
}

cap() {
    local t=$((10#$(date +%H%M)))
    if [ $t -ge 2300 ] || [ $t -lt 430 ]; then echo 3; else echo 2; fi
}

free_gpu() {   # 池里一张不是我们占的、也没有任何进程的卡；没有则输出空
    local used=" $(ours | tr '\n' ' ') "
    for g in $POOL; do
        [[ "$used" == *" $g "* ]] && continue
        [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] && { echo $g; return; }
    done
}

unclaimed() {
    for f in $R/configs/*.toml; do
        m=$(basename $f .toml)
        [[ $m == _* ]] && continue
        [ -d $R/.claims/$m ] || { echo $m; return; }
    done
}

mkdir -p $R/raw
log "调度开始，pid $$"
declare -A started   # 本脚本起的打标：方法 → pid
while true; do
    for x in $NOISES; do   # 2. 放行第 2 轮；打标进程已结束却没有结果的记一次失败
        m=${x}__r1
        if [ -s $R/extra/overlap_$m.json ] && [ -f $R/.claims/${x}__r2/held ]; then
            rm -rf $R/.claims/${x}__r2
            log "放行 ${x}__r2：$(cat $R/extra/overlap_$m.json | tr -d '\n ')"
        elif [ -n "${started[$m]:-}" ] && ! kill -0 ${started[$m]} 2> /dev/null && [ ! -s $R/extra/overlap_$m.json ]; then
            log "打标 $m 失败（进程已退出、无结果），见 raw/relabel_$m.out；不重试"
            unset "started[$m]"
        fi
    done
    n_ours=$(ours | sort -u | grep -c .)
    while [ $n_ours -lt $(cap) ]; do
        g=$(free_gpu)
        [ -n "$g" ] || break
        todo=""
        for x in $NOISES; do   # 1. 优先重新打标
            m=${x}__r1
            if grep -qs "OK $m\$" $R/ckpt/queue_*.log && mkdir $Y/n130_relabel_$m.lock 2> /dev/null; then
                todo=$m; break
            fi
        done
        if [ -n "$todo" ]; then
            GPUS=$g nohup bash $R/code/relabel.sh $todo > $R/raw/relabel_$todo.out 2>&1 < /dev/null &
            started[$todo]=$!
            log "打标 $todo：GPU $g，pid $!"
        elif [ -n "$(unclaimed)" ]; then   # 3. 起队列
            GPU=$g nohup $PY scripts/finetune/run.py $R > $R/ckpt/queue_$(hostname)_g$g.log 2>&1 < /dev/null &
            log "队列：GPU $g，pid $!，下一个未领的方法 $(unclaimed)"
        else
            break
        fi
        sleep 60   # 等新进程上卡，再数一次
        n_ours=$(ours | sort -u | grep -c .)
    done
    if [ $(ls -d $R/preds/*__r2 2> /dev/null | wc -l) -ge 7 ]; then
        log "第 2 轮全部出结果，调度退出"
        exit 0
    fi
    sleep 300
done
