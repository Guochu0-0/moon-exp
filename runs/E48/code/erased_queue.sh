#!/bin/bash
# E48（#256）：训练结束后，在擦过的 Val 上逐个存档跑 erased_val.py（E41 门槛 0.8 的最后存档 + 本实验全部存档），
# 原始结果写 runs/E48/raw/erased/<名字>.jsonl（不进 git，由 summary.py 汇总）。每个存档靠 mkdir .evalclaims/erased_<名字>
# 分工，可在几张卡上同时起；跑完的写 done。中途断掉的：删掉对应的 .evalclaims/erased_<名字>（没有 done 的）再起，
# erased_val.py 会接着已写的行续跑。
#   GPU=<空卡> nohup bash runs/E48/code/erased_queue.sh > runs/E48/ckpt/erased_<host>_g<GPU>.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
R=runs/E48
E41=/remote-home/xufang/YGC/moon-exp/runs/E41/ckpt/main__screen0.8/ckpt_7907.pt
mkdir -p $R/.evalclaims $R/raw/erased
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID MOON_DATA=${MOON_DATA:-/remote-home/xufang/YGC/dataset/Moon} \
       TORCH_HOME=${TORCH_HOME:-/remote-home/xufang/YGC/weights/torch_home}
for c in $E41 $(ls $R/ckpt/main/ckpt_*.pt | sort -t_ -k2 -n); do
  case $c in $E41) tag=e41_screen0.8_7907 ;; *) tag=step$(basename $c .pt | cut -d_ -f2) ;; esac
  mkdir $R/.evalclaims/erased_$tag 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) START $tag gpu$GPU"
  if /opt/envs/loftr/bin/python $R/code/erased_val.py --ckpt $c --out $R/raw/erased/$tag.jsonl; then
    touch $R/.evalclaims/erased_$tag/done
    echo "$(date -u +%FT%TZ) OK $tag"
  else
    echo "$(date -u +%FT%TZ) FAIL $tag"
  fi
done
echo "$(date -u +%FT%TZ) all claimed"
