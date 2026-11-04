#!/bin/bash
# E40（#242）抽样噪声对照：推理完成后合成 resample、resample2 两行预测，重新评测与汇总（只用 CPU）。在本票 worktree 根目录运行：
#   bash runs/E40/code/post_resample.sh
# 三项约束的逐视角结果（rows_s*.jsonl）在第一轮收尾时已挪到 gpfs 主 checkout 的 runs/E40/raw/，这里与本 worktree 的 raw/ 合并读取；
# 已提交的五行预测不重写。
set -euo pipefail
Y=/remote-home/xufang/YGC
WB=/opt/envs/wb/bin/python
export MOON_DATA=$Y/dataset/Moon PYTHONPATH=.
ok=fail
$WB -m workbench.launch begin runs/E40 post_resample -- bash runs/E40/code/post_resample.sh
trap '$WB -m workbench.launch end runs/E40 post_resample $ok' EXIT
$WB runs/E40/code/fit.py --raw $Y/moon-exp/runs/E40/raw runs/E40/raw --methods resample,resample2
$WB -m workbench eval E40
$WB runs/E40/code/summary.py
ok=ok
