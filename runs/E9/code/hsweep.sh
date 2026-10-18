#!/usr/bin/env bash
# 逐 ckpt 的难版 Val 评测（#157）：方法 <m> 的每个 ckpt × 9 个扰动变体，只跑 Val，用来报告「按原始 + 难版 9 档平均选模」会选到哪一步。
# 结果是工作台记录 runs/E9/sweep/<m>/H（方法名 step<N>__<变体>，perturb = true，扰动文件同 runs/E9/perturb），preds 不进 git；
# 原始档沿用驱动的 sweep/<m>/S。正式结果仍是驱动按原始 Val AUC@5 选的 preds/<m>/。
#
#   GPU=<空卡> nohup bash runs/E9/code/hsweep.sh <m> > runs/E9/ckpt/hs_<m>_<host>_g<GPU>.log 2>&1 < /dev/null &
#
# 在本票 worktree 根目录运行；可在多张卡上对同一个 <m> 各起一个，靠 mkdir runs/E9/.claims/hs_<m>_step<N>__<变体> 占位。
# 推理配置按方法前缀：rns* 用 RoMa（V100），其余用 LoFTR（RTX 3090），同 code/queue.sh。
set -uo pipefail
R=runs/E9
m=${1:?方法名}
case $m in rns*) cfg=configs/baselines/anymatch_roma__minmax.json ;; *) cfg=configs/baselines/anymatch_loftr.json ;; esac
Y=/remote-home/xufang/YGC
PY=/opt/envs/loftr/bin/python; WB=/opt/envs/wb/bin/python
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights TORCH_HOME=$Y/weights/torch_home
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
SW=$R/sweep/$m
H=$SW/H
VARIANTS=(shift12 shift64 shift128 rot3 rot10 rot20 scale3 scale10 scale20)
mkdir -p $R/.claims
if mkdir $SW/.hinit 2>/dev/null; then   # 只建一次
  [ -d $H ] || $WB -m workbench --runs $SW new H --title "$R/$m 逐 ckpt 难版 Val"
  grep -q '^perturb' $H/exp.toml || echo 'perturb = true' >> $H/exp.toml
  cp -r $R/perturb $H/
fi
while [ ! -d $H/perturb ]; do sleep 5; done
for c in $(ls $R/ckpt/$m/ckpt_*.pt | sort -t_ -k2 -n); do
  st=$(basename $c .pt); st=${st#ckpt_}
  [ "$st" = 0 ] && continue
  for v in "${VARIANTS[@]}"; do
    name=step${st}__$v
    [ -e $R/.stop ] || [ -e $R/.stop_g$GPU ] && { echo "$(date +%T) stop file, exit"; exit 0; }
    mkdir $R/.claims/hs_${m}_$name 2>/dev/null || continue
    echo "$(date +%T) START $m $name"
    if [ ! -f $H/preds/$name/val.jsonl ]; then
      nice -n 10 $PY -m baselines.match $cfg --split val --weights $c --perturb $H/perturb/$v --name $name --out $SW/hmatch \
        >> $SW/hsweep.log 2>&1 &&
      $WB -m baselines.fit $SW/hmatch/$name --split val --run $H >> $SW/hsweep.log 2>&1 &&
      echo "$(date +%T) OK $m $name" || echo "$(date +%T) FAIL $m $name"
    fi
  done
done
echo "$(date +%T) queue empty"
