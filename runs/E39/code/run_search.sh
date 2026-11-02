#!/bin/bash
# E39（#240）：只用 CPU。开跑前核对 → 步长核对（每 16 对取一对）→ 全量搜索 → SAR 映射换 minmax 的敏感性。
# 在本票 worktree 根目录运行：nohup bash runs/E39/code/run_search.sh > runs/E39/raw/search.log 2>&1 < /dev/null &
set -euo pipefail
export MOON_DATA=${MOON_DATA:-/remote-home/xufang/YGC/dataset/Moon} PYTHONPATH=. OMP_NUM_THREADS=1
PY=/opt/envs/wb/bin/python
D=runs/E39/extra/data
W=${WORKERS:-24}
mkdir -p $D/conv runs/E39/raw
ok=fail
$PY -m workbench.launch begin runs/E39 main -- bash runs/E39/code/run_search.sh
trap '$PY -m workbench.launch end runs/E39 main $ok' EXIT

nice -n 10 $PY runs/E39/code/sanity.py

for m in zero_shot e37; do
  # 步长核对：粗（平移 4 px、3 档、2 起点）、用的（2 px、3 档、8 起点）、细（1 px、5 档、32 起点、最终步长减半）
  nice -n 10 $PY runs/E39/code/search.py --model $m --every 16 --workers $W --t-step 4 --lin-levels 3 --topk 2 --topk-t 2 \
    --rows trans,full,identity --out $D/conv/${m}_coarse.jsonl
  nice -n 10 $PY runs/E39/code/search.py --model $m --every 16 --workers $W \
    --rows trans,full,identity --out $D/conv/${m}_used.jsonl
  nice -n 10 $PY runs/E39/code/search.py --model $m --every 16 --workers $W --t-step 1 --lin-levels 5 --topk 32 --topk-t 16 --final 0.03125 \
    --rows trans,full,identity --out $D/conv/${m}_fine.jsonl
done

for m in zero_shot e37; do
  nice -n 10 $PY runs/E39/code/search.py --model $m --workers $W --out $D/rows_$m.jsonl
done
for m in zero_shot e37; do
  nice -n 10 $PY runs/E39/code/search.py --model $m --workers $W --sar minmax --out $D/rows_${m}_minmax.jsonl
done
ok=ok
echo ALL-DONE
