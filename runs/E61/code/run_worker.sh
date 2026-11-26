#!/usr/bin/env bash
set -euo pipefail

node=$1
gpu=$2
minutes=${3:-220}
ygc=/remote-home/xufang/YGC
run=runs/E61
reservations=$ygc/.gpu-reservations-$node
slot=$reservations/$gpu
holder="$$ E61/checkpoint-evaluation"

mkdir -p "$reservations"
exec 9>"$ygc/.gpu-launch-$node.lock"
flock -x 9
if [[ -s $slot ]]; then
    read -r pid _ < "$slot"
    if kill -0 "$pid" 2>/dev/null; then
        echo "GPU $gpu on $node is already reserved"
        exit 1
    fi
fi
active=$(timeout 10 nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n $active ]]; then
    echo "GPU $gpu on $node has running processes"
    exit 1
fi
if (( $(timeout 10 nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) > 64 )); then
    echo "GPU $gpu on $node has allocated memory"
    exit 1
fi
printf '%s\n' "$holder" > "$slot"
flock -u 9
release() {
    flock -x 9
    if [[ -f $slot && $(cat "$slot") == "$holder" ]]; then
        rm -- "$slot"
    fi
}
trap release EXIT

scratch=$(mktemp -d /dev/shm/moon-e61-XXXXXX)
release_all() {
    rmdir -- "$scratch" || true
    release
}
trap release_all EXIT
export TMPDIR=$scratch CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MOON_GIT_TIMEOUT=240
/opt/envs/wb/bin/python "$run/code/evaluate_grid.py" worker --minutes "$minutes"
