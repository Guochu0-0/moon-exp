#!/bin/bash
# 只重出相图与汇总（读 run.sh 留下的 raw/ 分片，不碰 GPU），写一次启动记录（#195）。
# 在 worktree 根目录：bash runs/E21/code/replot.sh > runs/E21/.claims/replot.log 2>&1
set -uo pipefail
R=runs/E21
PY=/opt/envs/loftr/bin/python
$PY -m workbench.launch begin $R main -- bash $R/code/replot.sh || exit 2
st=ok
$PY $R/code/phase.py > $R/.claims/phase.log 2>&1 || { echo "phase FAIL"; st=fail; }
$PY $R/code/report.py > $R/.claims/report.log 2>&1 || { echo "report FAIL"; st=fail; }
$PY -m workbench.launch end $R main $st
