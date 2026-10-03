"""首轮实验汇总表（#51 #53）：每个 run 的 Val AUC@5 峰值与 step、后段均值 / std（step ≥ --from）、峰值 ckpt 的 Test 指标。
读 first_round.py 的产物 <finetune>/<name>/sweep/S/metrics.json 与 peak.json。

    python scripts/finetune/summary.py Q0 Q1 ... [--root $MOON_RESULTS/finetune] [--from 2000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


def row(root, name, start):
    R = Path(root) / name
    m = json.loads((R / "sweep/S/metrics.json").read_text(encoding="utf-8"))["methods"]
    pts = sorted((int(k[4:]), v["val"]["summary"]["auc@5"]) for k, v in m.items() if "val" in v)
    late = [v for s, v in pts if s >= start]
    pk = json.loads((R / "peak.json").read_text(encoding="utf-8")) if (R / "peak.json").exists() else None
    out = {"curve": pts, "late_mean": statistics.mean(late), "late_std": statistics.pstdev(late)}
    if pk:
        out.update(peak_step=pk["step"], val=pk["val"], test=pk["test"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--root", default=str(Path(os.environ.get("MOON_RESULTS", ".")) / "finetune"))
    ap.add_argument("--from", dest="start", type=int, default=2000)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    res = {}
    print("| run | Val 峰值 (step) | 后段均值 ± std | Test AUC@5 | Test AUC@3 / @10 | Test SR@3 / @10 |")
    print("|---|---|---|---|---|---|")
    for n in args.names:
        try:
            r = row(args.root, n, args.start)
        except FileNotFoundError:
            print(f"| {n} | （未完成） | | | | |")
            continue
        res[n] = r
        if "test" in r:
            v, t = r["val"], r["test"]
            print(f"| {n} | {v['auc@5']:.3f} ({r['peak_step']}) | {r['late_mean']:.3f} ± {r['late_std']:.3f} | "
                  f"{t['auc@5']:.3f} | {t['auc@3']:.3f} / {t['auc@10']:.3f} | {t['sr@3']:.3f} / {t['sr@10']:.3f} |")
        else:
            print(f"| {n} | — | {r['late_mean']:.3f} ± {r['late_std']:.3f} | | | |")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
