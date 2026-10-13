#!/bin/bash
# runs/O 的无人值守调度（#131），照 runs/N/code/auto.sh 改：常驻在一台机器上，每 5 分钟看一轮，日志写 runs/O/raw/auto.log。
# 每一项「老师:学生[:旧标签]」：
#   1. 老师方法出了结果（sweep/<老师>/peak.json 与 preds/<老师>/test.jsonl）→ 在一张空卡上用它重新打标（code/relabel.sh，
#      其中包含 code/prep.sh 的选对），重合比例与旧标签比（默认发布权重打的标签）；
#   2. 标签与重合比例写出（extra/overlap_<老师>.json）→ 撤掉学生的占位（.claims/<学生>/held），放行；
#   3. 还有没被领走的方法、又有空卡 → 起一个 run.py 队列。
# 用卡（docs/agents/servers.md）：只用 POOL 里没有任何进程的卡；自己同时最多占 CAP_DAY 张，23:00–04:30 之间新起的任务
# 可占到 CAP_NIGHT 张（3–4 小时内跑完，08:00 前让出）。默认值按 150 写（8 张卡，白天 3、夜间 6）。打标失败不重试，记进日志。
# 环境变量 SHARDS=n：每次打标在同一张卡上并行 n 个分片。列出的学生全部出结果、且没有未领的方法后退出。
# 用法（在本票 worktree 根目录，代码已提交；学生的占位要先建好，见 code/start.sh）：
#   nohup bash runs/O/code/auto.sh <标签> <老师:学生[:旧标签]>… &
set -uo pipefail
Y=/remote-home/xufang/YGC
R=runs/O
POOL=${POOL:-"0 1 2 3 4 5 6 7"}
PY=/opt/envs/loftr/bin/python
tag=${1:?标签}
shift
ITEMS="$*"
mkdir $Y/o131_auto_$tag.lock || { echo "调度 $tag 已在跑或跑过（锁 $Y/o131_auto_$tag.lock）"; exit 1; }
mkdir -p $R/raw $R/ckpt
log() { echo "$(date '+%F %T') [$tag] $*" >> $R/raw/auto.log; }

ours() {   # 我们在本机占着的 GPU
    for p in $(pgrep -f "scripts/finetune/run.py runs/O"); do tr '\0' '\n' < /proc/$p/environ | sed -n 's/^GPU=//p'; done
    for p in $(pgrep -f "runs/O/code/relabel.sh"); do tr '\0' '\n' < /proc/$p/environ | sed -n 's/^GPUS=//p' | tr ' ' '\n'; done
    for p in $(pgrep -f "runs/O0/code/infer.sh"); do tr '\0' '\n' < /proc/$p/environ | sed -n 's/^GPU=//p'; done
}

cap() {
    local t=$((10#$(date +%H%M)))
    if [ $t -ge 2300 ] || [ $t -lt 430 ]; then echo ${CAP_NIGHT:-6}; else echo ${CAP_DAY:-3}; fi
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

log "调度开始，pid $$，host $(hostname)：$ITEMS"
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
            if [ -f $R/sweep/$t/peak.json ] && [ -f $R/preds/$t/test.jsonl ] && mkdir $Y/o131_relabel_$t.lock 2> /dev/null; then
                todo=$t todo_old=$old; break
            fi
        done
        if [ -n "$todo" ]; then
            GPUS="$(for _ in $(seq ${SHARDS:-1}); do echo -n "$g "; done)" nohup bash $R/code/relabel.sh $todo $todo_old > $R/raw/relabel_$todo.out 2>&1 < /dev/null &
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
    if [ $left -eq 0 ] && [ -z "$(unclaimed)" ] && ! pgrep -f "scripts/finetune/run.py runs/O" > /dev/null; then
        log "列出的学生全部出结果，调度退出"
        exit 0
    fi
    sleep 300
done
