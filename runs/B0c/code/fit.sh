#!/bin/bash
. /remote-home/xufang/YGC/moon-exp-cf70/cf70/common.sh
LOCK=$MOON_RESULTS/coarse70/fit.lock
mkdir $LOCK || { echo "lock exists: $LOCK"; exit 1; }
cd $D
WB=/opt/envs/wb/bin/python
for r in "B0c B0 anymatch_loftr" "S1c S1 main" "Cc C C2" "Qc Q Q4 Q4s1" "Pc P P8"; do
  set -- $r; id=$1; par=$2; shift 2
  [ -d runs/$id ] || $WB -m workbench --runs runs new $id --parent $par --title "$par 粗级匹配（LoFTR coarse，#70）" || $WB -m workbench --runs runs new $id --title "$par 粗级匹配（LoFTR coarse，#70）"
  for m in "$@"; do for s in val test; do for t in 3 8; do
    timeout 3600 nice -n 10 $WB -m baselines.fit $MOON_RESULTS/coarse70/$m-coarse --split $s --run runs/$id --ransac $t --name ${m}_r$t
  done; done; done
done
echo FITDONE
