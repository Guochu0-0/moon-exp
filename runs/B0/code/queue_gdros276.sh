#!/usr/bin/env bash
# Call from the ticket worktree after checking availability and choosing a GPU.
set -euo pipefail
weight=${1:?expected whu or os}
gpu=${2:?expected GPU index}
python_bin=${3:?expected inference Python path}
upstream=${4:?expected upstream checkout}
ygc=/remote-home/xufang/YGC
case "$weight" in whu|os) ;; *) exit 2 ;; esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir "$ygc/gdros276-${weight}.lock"
trap 'rmdir "$ygc/gdros276-${weight}.lock"' EXIT
for split in val test; do
  timeout 3h "$python_bin" -u runs/B0/code/run_gdros276.py "$weight" "$split" \
    --repo "$upstream" --weights-root "$ygc/weights/gdros" --data "$ygc/dataset/Moon"
done
