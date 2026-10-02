"""一次性：把旧做法留下的目录挪进 YGC/_archive/2026-10/（#77「历史代码与结果整理」）。

    /opt/envs/wb/bin/python scripts/archive77.py            # dry-run：只打印计划
    /opt/envs/wb/bin/python scripts/archive77.py --apply    # 执行

在 migrate77.py 之后运行。全部是 rename（同一 gpfs），不删文件：
1. results/finetune/ 下剩余的方法目录（R 系、S1 变体等）：ckpt 只留峰值与最后一个，其余挪进 _archive/2026-10/ckpt/finetune/<目录>/。
2. tmp/envpack/ 挪到 YGC/envpack/（150 的环境从这里解出，仍在用）。
3. 挪进 _archive/2026-10/：results/finetune/ 除 labels_* 以外的全部、results/{coarse70,diag_reward}、
   旧 worktree 与 archive 副本 moon-exp-*、tmp/、projects/optical-sar-matching/。
   results/{baselines,baselines_ablation} 留在原位（scripts/baselines/ 的输入）。
之后在主 checkout 上 `git worktree prune`，清掉已挪走的 worktree 的登记（分支保留）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

Y = Path("/remote-home/xufang/YGC")
F = Y / "results/finetune"
ARC = Y / "_archive/2026-10"
COPIES = ["comir", "ft25", "ft26", "ft50", "w2", "w3", "w4", "w5", "ft49", "ft51", "r2", "roma", "cf70"]


def step_of(p: Path):
    return int(re.search(r"(\d+)$", p.stem).group(1))


def peak(d: Path):
    if (d / "peak.json").exists():
        return json.loads((d / "peak.json").read_text())["step"]
    m = d / "sweep/S/metrics.json"
    if not m.exists():
        return None
    ms = json.loads(m.read_text(encoding="utf-8"))["methods"]
    pts = [(int(k[4:]), v["val"]["summary"]["auc@5"]) for k, v in ms.items()
           if k.startswith("step") and "val" in v and int(k[4:]) > 0]
    return max(pts, key=lambda p: p[1])[0] if pts else None


class Plan:
    def __init__(self, apply):
        self.apply, self.n = apply, 0

    def mv(self, a: Path, b: Path, quiet=False):
        if b.exists():
            raise SystemExit(f"目标已存在：{b}")
        self.n += 1
        if not quiet:
            print(f"  mv {a} -> {b}")
        if self.apply:
            b.parent.mkdir(parents=True, exist_ok=True)
            os.rename(a, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    P = Plan(a.apply)

    print("# 1. 剩余方法目录的 ckpt 修剪")
    for d in sorted(p for p in F.iterdir() if p.is_dir()):
        ck = sorted(d.glob("ckpt_*.pt"), key=step_of)
        if not ck:
            continue
        st = peak(d)
        keep = {step_of(ck[-1])} | ({st} if st is not None else set())
        move = [p for p in ck if step_of(p) not in keep]
        print(f"[{d.name}] ckpt {len(ck)} 个，peak={st} 留 {sorted(keep)}，挪走 {len(move)} 个"
              + ("（无 Val 记录，只留最后一个）" if st is None else ""))
        for p in move:
            P.mv(p, ARC / "ckpt/finetune" / d.name / p.name, quiet=True)

    print("# 2. envpack")
    P.mv(Y / "tmp/envpack", Y / "envpack")

    print("# 3. 归档")
    for p in sorted(F.iterdir()):
        if not p.name.startswith("labels_"):
            P.mv(p, ARC / "results/finetune" / p.name, quiet=True)
    print(f"  results/finetune/ 除 labels_* 外 {sum(1 for p in F.iterdir() if not p.name.startswith('labels_'))} 项"
          f" -> {ARC / 'results/finetune'}")
    for n in ("coarse70", "diag_reward"):
        P.mv(Y / "results" / n, ARC / "results" / n)
    for c in COPIES:
        P.mv(Y / f"moon-exp-{c}", ARC / f"moon-exp-{c}")
    P.mv(Y / "tmp", ARC / "tmp")
    P.mv(Y / "projects/optical-sar-matching", ARC / "projects/optical-sar-matching")
    print(f"\n共 {P.n} 次 mv{'（已执行）' if a.apply else '（dry-run）'}")


if __name__ == "__main__":
    main()
