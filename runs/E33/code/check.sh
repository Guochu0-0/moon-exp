#!/bin/bash
# E33 只用（#226）：提速前后的核对，在一张卡上依次跑。
#   梯度：grad_check.py 旧做法（e32ema）存一份，再跑一遍旧做法（量出同一份代码两次之间的差）和新做法（sdpa）各比一次；
#   短跑：KIND=short 时同种子训 200 步（finetune.train --stop-at 200），TAGS 给出配置名（runs/E33/code/cfg/<名>.toml），
#   OUT 下每个配置一个目录；同一配置要跑两遍时写 名:目录名。
#   GPU=<卡> [KIND=grad|short] [TAGS="e32ema sdpa"] nohup bash runs/E33/code/check.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
source runs/E33/code/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
if [ "${KIND:-grad}" = grad ]; then
  d=runs/E33/ckpt/gradcheck
  mkdir -p $d
  $PY runs/E33/code/grad_check.py runs/E33/code/cfg/e32ema.toml --out $d/e32ema --save >> $d/run.log 2>&1
  $PY runs/E33/code/grad_check.py runs/E33/code/cfg/e32ema.toml --out $d/e32ema_again --against $d/e32ema >> $d/run.log 2>&1
  for t in ${TAGS:-sdpa}; do
    $PY runs/E33/code/grad_check.py runs/E33/code/cfg/$t.toml --out $d/$t --against $d/e32ema >> $d/run.log 2>&1
  done
  echo GRAD_DONE >> $d/run.log
else
  d=runs/E33/ckpt/short
  mkdir -p $d
  for spec in $TAGS; do
    t=${spec%%:*}; name=${spec#*:}
    $PY -m finetune.train runs/E33/code/cfg/$t.toml --out $d/$name --stop-at 200 >> $d/$name.log 2>&1
    echo "SHORT_DONE $name" >> $d/run_g$GPU.log
  done
fi
