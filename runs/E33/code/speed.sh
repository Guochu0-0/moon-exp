#!/bin/bash
# E33 只用（#226）：在一张卡上依次测各提速配置一步的耗时（profile_step.py plain，60 步，前 10 步不计）。
# 测过的 fused、cudnn、foreach、all 四份配置随提速开关一起删了（用户决定不用），见提交 965e65b。
#   GPU=<卡> nohup bash runs/E33/code/speed.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
for t in ${TAGS:-e32ema sdpa}; do
  GPU=$GPU CFG=runs/E33/code/cfg/$t.toml TAG=$t MODES=plain bash runs/E33/code/profile.sh
done
echo SPEED_DONE >> runs/E33/ckpt/profile/speed.log
