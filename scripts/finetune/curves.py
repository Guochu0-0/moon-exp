"""各 run 的 Val AUC@5 随 step 的曲线（「把 S1 调好」#50）。读 scenes.sh 产出的 <finetune>/<name>/sweep/S/metrics.json。

    python scripts/finetune/curves.py S1 S1s1 T1 ... [--root $MOON_RESULTS/finetune] [--from 2000] [--json out.json]

每个 run 一行：各 step 的 AUC@5；峰值及其 step（选模规则）；step ≥ --from 的均值、标准差、极差（看平台期的波动）。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


def curve(root, name, metric="auc@5"):
    m = json.loads((Path(root) / name / "sweep/S/metrics.json").read_text(encoding="utf-8"))["methods"]
    pts = sorted((int(k[4:]), v["val"]["summary"][metric]) for k, v in m.items() if k.startswith("step") and "val" in v)
    return pts


def summarize(pts, start):
    late = [v for s, v in pts if s >= start and s > 0]
    best = max((p for p in pts if p[0] > 0), key=lambda p: p[1])
    return {"peak": best[1], "peak_step": best[0], "late_mean": statistics.mean(late),
            "late_std": statistics.pstdev(late), "late_range": max(late) - min(late), "n_late": len(late)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--root", default=str(Path(os.environ.get("MOON_RESULTS", ".")) / "finetune"))
    ap.add_argument("--from", dest="start", type=int, default=2000, help="平台期统计的起始 step")
    ap.add_argument("--metric", default="auc@5")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    out = {}
    for n in args.names:
        try:
            pts = curve(args.root, n, args.metric)
        except FileNotFoundError:
            print(f"{n:10s} (no sweep yet)")
            continue
        sm = summarize(pts, args.start)
        out[n] = {"curve": pts, **sm}
        print(f"{n:10s} peak {sm['peak']:.4f}@{sm['peak_step']:<5d} late mean {sm['late_mean']:.4f} "
              f"std {sm['late_std']:.4f} range {sm['late_range']:.4f} | "
              + " ".join(f"{s // 1000}k:{v:.3f}" for s, v in pts))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
