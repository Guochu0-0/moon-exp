#!/bin/bash
# RoMa 原版室外权重（#187；romatch 的 roma_outdoor 默认下载地址，GitHub release Parskatt/storage@roma）。在能连 GitHub 的机器（126、154）上跑。
#   bash runs/E17/code/fetch_roma.sh
set -euo pipefail
OUT=/remote-home/xufang/YGC/weights/official/roma
mkdir -p "$OUT"
[ -f "$OUT/roma_outdoor.pth" ] && { echo "已有"; exit 0; }
curl -fL --retry 3 -o "$OUT/roma_outdoor.pth.part" https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth
mv "$OUT/roma_outdoor.pth.part" "$OUT/roma_outdoor.pth"
sha256sum "$OUT/roma_outdoor.pth"
