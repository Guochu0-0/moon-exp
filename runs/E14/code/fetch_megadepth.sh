#!/bin/bash
# MegaDepth-1500（LoFTR 测试集）的影像：154 的下载通道不通（2026-10-20），从 gpfs 上的 MegaDepth_v1.tar 只解出两个测试场景
# 0015、0022 的影像（dense*/imgs，不要深度）。tar 不能随机访问，会顺序读完整包（217 GB），所以低 IO 优先级跑。
#   bash runs/E14/code/fetch_megadepth.sh [输出目录]
set -euo pipefail
TAR=/remote-home/xufang/YGC/dataset/MegaDepth_v1.tar
OUT=${1:-/remote-home/xufang/YGC/dataset/megadepth_test_1500}
mkdir -p "$OUT"
ionice -c3 nice -n 19 tar -xf "$TAR" -C "$OUT" --wildcards \
    'phoenix/S6/zl548/MegaDepth_v1/0015/dense*/imgs/*' 'phoenix/S6/zl548/MegaDepth_v1/0022/dense*/imgs/*'
touch "$OUT/.done"
ls "$OUT"/phoenix/S6/zl548/MegaDepth_v1/*/dense*/imgs | head -3
