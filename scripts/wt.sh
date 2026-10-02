#!/bin/bash
# 服务器上每张票一个 git worktree（#76）。规矩见 docs/agents/experiments.md。在 gpfs 主机（154 / 126）上跑；A6000 上同理（它自己的主 checkout）。
#
#   scripts/wt.sh new <分支> [起点，默认 origin/main]   建 $WT/<分支>：已有远端分支就跟踪它，否则从起点新建；子模块从主 checkout 引用初始化
#   scripts/wt.sh close <分支> [--dry-run]              结票收尾：核对记录已进 main → 把 runs/ 下不进 git 的文件挪回主 checkout → 删 worktree
#
# 主 checkout（$MAIN，永远停在 main、只做 pull）是全部结果的权威存储，也是 workbench sync 的拉取源。
# close 的前提：分支已合并、主 checkout 已 pull 到含这些记录的 main。它只 mv（同一 gpfs，是改名），不删文件；
# 目标处已有同名文件且大小不同时停下，留给人处理。挪完还有剩下的 runs/ 产物（.claims 除外）也停下，不删 worktree。
set -euo pipefail
MAIN=${MAIN:-/remote-home/xufang/YGC/moon-exp}
WT=${WT:-/remote-home/xufang/YGC/wt}
g() { git --no-optional-locks "$@"; }
die() { echo "wt: $*" >&2; exit 1; }

new() {
  local br=$1 start=${2:-origin/main} d=$WT/$1
  [ -e "$d" ] && die "$d 已存在"
  [ "$(g -C "$MAIN" rev-parse --abbrev-ref HEAD)" = main ] || die "$MAIN 不在 main 上"
  timeout 300 git -C "$MAIN" fetch -q origin || die "fetch 失败（154 / 126 才能连 GitHub）"
  mkdir -p "$WT"
  if g -C "$MAIN" rev-parse -q --verify "refs/remotes/origin/$br" >/dev/null; then
    git -C "$MAIN" worktree add -q --track -b "$br" "$d" "origin/$br"
  else
    git -C "$MAIN" worktree add -q -b "$br" "$d" "$start"
  fi
  # 子模块：主 checkout 里已初始化的，用它当 --reference（只借对象，不走外网）；嵌套子模块再递归补上
  local p
  for p in $(g -C "$MAIN" submodule status | awk '$1 !~ /^-/ {print $2}'); do
    timeout 600 git -C "$d" submodule update -q --init --reference "$MAIN/$p" -- "$p" || die "子模块 $p 初始化失败"
  done
  timeout 600 git -C "$d" submodule update -q --init --recursive || die "嵌套子模块初始化失败"
  echo "$d"
}

close() {
  local br=$1 dry=${2:-} d=$WT/$1
  [ -d "$d" ] || die "$d 不存在"
  [ -z "$(g -C "$d" status --porcelain --untracked-files=all -- . ':(exclude,top)runs' ':(glob,top)runs/*/code/**')" ] \
    || die "$d 有未提交的代码"
  [ "$(g -C "$MAIN" rev-parse --abbrev-ref HEAD)" = main ] || die "$MAIN 不在 main 上"
  # 本分支动过的实验（相对它与 main 的分叉点），其记录必须与主 checkout 的 HEAD 一致，即已合并且已 pull
  local base ids id
  base=$(g -C "$d" merge-base HEAD "$(g -C "$MAIN" rev-parse HEAD)")
  ids=$(g -C "$d" diff --name-only "$base" HEAD -- runs | cut -d/ -f2 | sort -u)
  for id in $ids; do
    g -C "$MAIN" diff --quiet HEAD "$(g -C "$d" rev-parse HEAD)" -- "runs/$id" \
      || die "runs/$id 与 $MAIN 的 main 不一致：先合并 PR，再在主 checkout 上 git pull --ff-only"
  done
  # 不进 git 的产物：逐个 mv 到主 checkout 的同一相对路径
  local f n=0
  while IFS= read -r f; do
    case $f in runs/*/.claims/*) continue ;; esac
    if [ -e "$MAIN/$f" ]; then
      [ "$(stat -c %s "$MAIN/$f")" = "$(stat -c %s "$d/$f")" ] || die "$MAIN/$f 已存在且大小不同，停下"
      echo "same  $f"; [ -n "$dry" ] || rm -- "$d/$f"
    else
      echo "move  $f"
      [ -n "$dry" ] || { mkdir -p "$(dirname "$MAIN/$f")"; mv -- "$d/$f" "$MAIN/$f"; }
    fi
    n=$((n + 1))
  done < <(g -C "$d" ls-files --others --ignored --exclude-standard -- runs)
  echo "wt: $n 个文件"
  [ -n "$dry" ] && { echo "wt: dry-run，未改动"; return; }
  [ -z "$(g -C "$d" ls-files --others --ignored --exclude-standard -- runs | grep -v '^runs/[^/]*/\.claims/' || true)" ] \
    || die "挪完仍有产物留在 $d/runs，未删 worktree"
  git -C "$MAIN" worktree remove --force "$d"   # 只剩 .claims、__pycache__ 之类；--force 是因为子模块
  echo "wt: 已删 $d（分支 $br 保留）"
}

case ${1:-} in
  new) [ $# -ge 2 ] || die "用法见文件头"; new "$2" "${3:-}" ;;
  close) [ $# -ge 2 ] || die "用法见文件头"; close "$2" "${3:-}" ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
