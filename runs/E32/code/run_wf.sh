#!/bin/bash
# E32 只用（#221）：在两张卡上并行跑 wf_grad.py（未训练模型 zs、E19 gauss8 第 1000 步 ck1000），各带启动记录。
#   GPU0=<卡> GPU1=<卡> nohup bash runs/E32/code/run_wf.sh > runs/E32/ckpt/wf.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
Y=/remote-home/xufang/YGC
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/opt/envs/loftr/bin/python
one() {   # 名字 卡 额外参数…
  local m=$1 gpu=$2; shift 2
  local cmd=($PY runs/E32/code/wf_grad.py runs/E32/configs/_e32.toml --tag "$m" --n 100 "$@")
  $PY -m workbench.launch begin runs/E32 "wf_$m" -- "${cmd[@]}"
  if CUDA_VISIBLE_DEVICES=$gpu nice -n 10 "${cmd[@]}" > "runs/E32/ckpt/wf_$m.log" 2>&1; then s=ok; else s=fail; fi
  $PY -m workbench.launch end runs/E32 "wf_$m" $s
}
mkdir -p runs/E32/ckpt
one zs "$GPU0" &
one ck1000 "$GPU1" --ckpt $Y/moon-exp/runs/E19/ckpt/gauss8/ckpt_1000.pt &
wait
