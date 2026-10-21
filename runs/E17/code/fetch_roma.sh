#!/bin/bash
# RoMa 原版室外权重（#187；romatch 的 roma_outdoor 默认下载地址，GitHub release Parskatt/storage@roma）。在能连 GitHub 的机器（126、154）上跑。
# 126 到 GitHub release 只有几十 KB/s，HTTP/2 连接会中途断开（2026-10-21），所以用 HTTP/1.1、断点续传、断了重试。
#   bash runs/E17/code/fetch_roma.sh
set -uo pipefail
OUT=/remote-home/xufang/YGC/weights/official/roma
mkdir -p "$OUT"
[ -f "$OUT/roma_outdoor.pth" ] && { echo "已有"; exit 0; }
for i in $(seq 1 30); do
  curl -fL --http1.1 -C - -o "$OUT/roma_outdoor.pth.part" \
      https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth && break
  echo "第 $i 次中断，10 秒后续传"; sleep 10
done
/opt/envs/loftr/bin/python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip()" "$OUT/roma_outdoor.pth.part" || { echo "文件不完整"; exit 1; }
mv "$OUT/roma_outdoor.pth.part" "$OUT/roma_outdoor.pth"
sha256sum "$OUT/roma_outdoor.pth"
