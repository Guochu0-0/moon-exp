#!/bin/bash
# E33 只用（#226）：在一张卡上依次测各提速配置一步的耗时（profile_step.py plain，60 步，前 10 步不计），最后对 all 再做一次 detail。
#   GPU=<卡> nohup bash runs/E33/code/speed.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
for t in ${TAGS:-e32ema sdpa fused cudnn foreach all}; do
  GPU=$GPU CFG=runs/E33/code/cfg/$t.toml TAG=$t MODES=plain bash runs/E33/code/profile.sh
done
GPU=$GPU CFG=runs/E33/code/cfg/all.toml TAG=all MODES=detail bash runs/E33/code/profile.sh
echo SPEED_DONE >> runs/E33/ckpt/profile/speed.log
