#!/bin/bash
# 多卡多机共用一个任务清单，逐个跑 scenes.sh（「把 S1 调好」#50）。
# usage: GPU=<id> queue.sh <jobs.txt>      每张空卡起一个；154 / 126 / 160 共用 gpfs，靠 mkdir 占位，不会重复跑。
# jobs.txt 每行 `<name> <finetune.train 参数...>`，# 开头为注释。每跑完一个就从头重读清单，所以中途追加的任务也会被领走。
# 各任务的输出：$MOON_RESULTS/finetune/<name>/（见 scenes.sh）与 <name>.driver.log；占位在 _claims/<name>/host。
cd "${REPO:-/remote-home/xufang/YGC/moon-exp-ft50}"
F=/remote-home/xufang/YGC/results/finetune
mkdir -p $F/_claims
jobs=$(realpath "$1")
while true; do
  line=""
  while read -r name args; do
    [[ -z $name || $name == \#* ]] && continue
    if mkdir $F/_claims/$name 2>/dev/null; then line="$name $args"; break; fi
  done < "$jobs"
  [ -z "$line" ] && { echo "$(date +%T) queue empty"; exit 0; }
  read -r name args <<< "$line"
  echo "$(hostname) gpu$GPU $(date +%T)" > $F/_claims/$name/host
  echo "$(date +%T) START $name $args"
  REPO=$PWD GPU=$GPU bash scripts/finetune/${DRIVER:-scenes.sh} $name $args > $F/$name.driver.log 2>&1 < /dev/null \
    && echo "$(date +%T) OK $name" || echo "$(date +%T) FAIL $name (see $F/$name.driver.log)"
done
