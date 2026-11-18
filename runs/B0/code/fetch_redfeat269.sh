#!/bin/bash
# Resume the pinned official sparse checkout, no local-machine weight relay.
# Run with nohup timeout 1200 bash ...; monitor fetch_redfeat269.log and pack sizes.
set -euo pipefail
cd /remote-home/xufang/YGC/wt/baseline-269
YGC=/remote-home/xufang/YGC
mkdir "$YGC/.redfeat-fetch269-background.lock"
trap 'code=$?; date -u; printf "fetch_exit=%s\n" "$code"' EXIT
export GIT_SSH_COMMAND="ssh -i $YGC/.ssh/moon-exp_deploy -o IdentitiesOnly=yes -o UserKnownHostsFile=$YGC/.ssh/known_hosts -o ConnectTimeout=15"
repo=third_party/ReDFeat
test -d "$repo/.git"
git -C "$repo" sparse-checkout set --no-cone '/lib/' '/Pretrained/VIS_SAR.pth' '/match.py' '/README.md'
git -C "$repo" -c url.'git@github.com:'.insteadOf=https://github.com/ checkout 66df80027dcb649118259a91cadbff4b50adc880
git -C "$repo" rev-parse HEAD
sha256sum "$repo/Pretrained/VIS_SAR.pth"
printf 'source_ready\n'
