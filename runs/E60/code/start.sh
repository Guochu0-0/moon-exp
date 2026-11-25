#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_HOME=/remote-home/xufang/YGC/weights/supplement268_torch_home
phase=${1:?phase required}
method=${2:?method required}
shift 2
gpu=${GPU:?physical GPU number required}
case "$gpu" in (*[!0-9]*) exit 2 ;; esac
if [[ -n $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo "GPU $gpu is occupied" >&2
    exit 1
fi

run=runs/E60
lock="$run/.claims/worker"
mkdir -p "$run/ckpt" "$run/.claims"
mkdir "$lock"
log="$run/ckpt/$(date -u +%Y%m%dT%H%M%S)_$(hostname)_g${gpu}_${phase}_${method}.log"
export CUDA_VISIBLE_DEVICES="$gpu"
nohup bash -c '
    lock=$1
    shift
    trap '\''rmdir -- "$lock"'\'' EXIT
    echo "worker=$$ gpu=$CUDA_VISIBLE_DEVICES"
    timeout --kill-after=90 14300 /opt/envs/loftr/bin/python -u -m runs.E60.code.run "$@"
' _ "$lock" "$phase" "$method" "$@" > "$log" 2>&1 < /dev/null &
echo "pid=$! log=$log"
