#!/bin/bash
# runs/N 的无人值守调度（#130）：在 126 上常驻，每 5 分钟看一轮，日志写 runs/N/raw/auto.log。
# 每一项「老师:学生[:旧标签]」：
#   1. 老师方法出了结果（sweep/<老师>/peak.json 与 preds/<老师>/）→ 在一张空卡上用它重新打标（code/relabel.sh），
#      重合比例与旧标签比（默认 zero-shot 标签）；
#   2. 标签与重合比例写出（extra/overlap_<老师>.json）→ 撤掉学生的占位（.claims/<学生>/held），放行；
#   3. 还有没被领走的方法、又有空卡 → 起一个 run.py 队列。
# 用卡（docs/agents/servers.md）：只用 126 的 GPU 1–3、只用没有任何进程的卡；自己同时最多占 2 张，
# 23:00–04:30 之间新起的任务可占到 3 张（3–4 小时内跑完，08:00 前让出）。打标失败不重试，记进日志。
# 列出的学生全部出结果、且没有未领的方法后退出。
# 用法（在本票 worktree 根目录，代码已提交；学生的占位要先建好）：
#   nohup bash runs/N/code/auto.sh <标签> <老师:学生[:旧标签]>… &
# 不给参数 = 第 1 轮 → 第 2 轮的 7 种噪声（标签 r2）。
set -uo pipefail
Y=/remote-home/xufang/YGC
R=runs/N
POOL="1 2 3"
PY=/opt/envs/loftr/bin/python
tag=${1:-r2}
shift || true
if [ $# -gt 0 ]; then ITEMS="$*"; else
    ITEMS=$(for x in geo photo erase blockmask droppath loradrop vggdrop; do echo ${x}__r1:${x}__r2; done)
fi
mkdir $Y/n130_auto_$tag.lock || { echo "调度 $tag 已在跑或跑过（锁 $Y/n130_auto_$tag.lock）"; exit 1; }
log() { echo "$(date '+%F %T') [$tag] $*" >> $R/raw/auto.log; }

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
log "调度开始，pid $$：$ITEMS"
declare -A started   # 本脚本起的打标：老师 → pid
while true; do
    for it in $ITEMS; do   # 2. 放行学生；打标进程已结束却没有结果的记一次失败
        IFS=: read -r t s old <<< "$it"
        if [ -s $R/extra/overlap_$t.json ] && [ -f $R/.claims/$s/held ]; then
            rm -rf $R/.claims/$s
            log "放行 $s：$(cat $R/extra/overlap_$t.json | tr -d '\n ')"
        elif [ -n "${started[$t]:-}" ] && ! kill -0 ${started[$t]} 2> /dev/null && [ ! -s $R/extra/overlap_$t.json ]; then
            log "打标 $t 失败（进程已退出、无结果），见 raw/relabel_$t.out；不重试"
            unset "started[$t]"
        fi
    done
    n_ours=$(ours | sort -u | grep -c .)
    while [ $n_ours -lt $(cap) ]; do
        g=$(free_gpu)
        [ -n "$g" ] || break
        todo="" todo_old=""
        for it in $ITEMS; do   # 1. 优先重新打标
            IFS=: read -r t s old <<< "$it"
            if [ -f $R/sweep/$t/peak.json ] && [ -f $R/preds/$t/test.jsonl ] && mkdir $Y/n130_relabel_$t.lock 2> /dev/null; then
                todo=$t todo_old=$old; break
            fi
        done
        if [ -n "$todo" ]; then
            GPUS=$g nohup bash $R/code/relabel.sh $todo $todo_old > $R/raw/relabel_$todo.out 2>&1 < /dev/null &
            started[$todo]=$!
            log "打标 $todo：GPU $g，pid $!"
        elif [ -n "$(unclaimed)" ]; then   # 3. 起队列
            GPU=$g nohup $PY scripts/finetune/run.py $R >> $R/ckpt/queue_$(hostname)_g$g.log 2>&1 < /dev/null &
            log "队列：GPU $g，pid $!，下一个未领的方法 $(unclaimed)"
        else
            break
        fi
        sleep 60   # 等新进程上卡，再数一次
        n_ours=$(ours | sort -u | grep -c .)
    done
    left=0
    for it in $ITEMS; do
        IFS=: read -r t s old <<< "$it"
        [ -f $R/preds/$s/test.jsonl ] || left=$((left + 1))
    done
    if [ $left -eq 0 ] && [ -z "$(unclaimed)" ] && ! pgrep -f "scripts/finetune/run.py runs/N" > /dev/null; then
        log "列出的学生全部出结果，调度退出"
        exit 0
    fi
    sleep 300
done
