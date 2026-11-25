#!/bin/bash
set -euo pipefail

gpu=${1:?GPU index required}
attempt=${2:?Unique attempt label required}
shift 2
[[ $gpu =~ ^[0-9]+$ && $attempt =~ ^[a-zA-Z0-9_-]+$ && $# -gt 0 ]]
run=runs/E59
mkdir -p "$run/.claims" "$run/ckpt"
active=$(timeout 12 nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
if [[ -n $active ]]; then
    printf 'GPU %s already has compute processes.\n' "$gpu" >&2
    exit 1
fi
mkdir "$run/.claims/start-$attempt"

log="$run/ckpt/${attempt}_$(hostname)_g${gpu}.log"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    nohup /opt/envs/loftr/bin/python -m runs.E59.code.run "$@" >"$log" 2>&1 </dev/null &
printf 'pid=%s log=%s\n' "$!" "$log"
