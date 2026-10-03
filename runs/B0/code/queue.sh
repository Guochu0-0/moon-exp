#!/bin/bash
# run batch-1 deep methods sequentially on one GPU; continue past failures
GPU=${GPU:-0}
for m in "$@"; do
  echo "$(date +%T) START $m"
  /root/scripts/run-method.sh $m loftr $GPU > /root/scripts/run-$m.log 2>&1 && echo "$(date +%T) OK $m" || echo "$(date +%T) FAIL $m"
done
echo "$(date +%T) QUEUE DONE"
