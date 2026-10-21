#!/bin/bash
# MegaDepth-1500 的深度图（#187）：E14 只解出了影像（runs/E14/code/fetch_megadepth.sh），这里从同一个包里再解出两个测试场景
# 0015、0022 的深度（dense*/depths/*.h5），放进同一个目录。tar 不能随机访问，会顺序读完整包（217 GB），所以低 IO 优先级跑。
#   bash runs/E17/code/fetch_depth.sh [输出目录]
set -euo pipefail
TAR=/remote-home/xufang/YGC/dataset/MegaDepth_v1.tar
OUT=${1:-/remote-home/xufang/YGC/dataset/megadepth_test_1500}
mkdir "$OUT/.depth_lock" || { echo "已在解"; exit 1; }
ionice -c3 nice -n 19 tar -xf "$TAR" -C "$OUT" --wildcards \
    'phoenix/S6/zl548/MegaDepth_v1/0015/dense*/depths/*' 'phoenix/S6/zl548/MegaDepth_v1/0022/dense*/depths/*'
touch "$OUT/.depth_done"
ls "$OUT"/phoenix/S6/zl548/MegaDepth_v1/*/dense*/depths | head -3
