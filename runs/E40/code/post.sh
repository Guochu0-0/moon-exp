#!/bin/bash
# E40（#242）：全部推理分片完成后，合成五行预测、评测、汇总（只用 CPU）。在本票 worktree 根目录运行：bash runs/E40/code/post.sh
set -euo pipefail
Y=/remote-home/xufang/YGC
WB=/opt/envs/wb/bin/python
export MOON_DATA=$Y/dataset/Moon PYTHONPATH=.
ok=fail
$WB -m workbench.launch begin runs/E40 post -- bash runs/E40/code/post.sh
trap '$WB -m workbench.launch end runs/E40 post $ok' EXIT
$WB runs/E40/code/fit.py
$WB -m workbench eval E40
$WB runs/E40/code/summary.py
ok=ok
