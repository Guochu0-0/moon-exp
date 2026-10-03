"""只用无标注代理量选模（地图 Not yet specified「消融实验矩阵」里的低优先级候选；第二轮 #58 #59）。

代理量只用 Val 影像上的匹配与 RANSAC 结果，不看检查点：
- inl_med：逐对内点数的中位数；
- inl_ratio：逐对内点率（内点 / 匹配）的中位数；
- ok200：内点数 ≥ 200 的对的比例。
对每个 run：各 ckpt 的代理量与 Val AUC@5（不含 step 0）；run 内按代理量取 ckpt 相对 Val 峰值的损失（regret）；
run 间用各 run 代理量最优 ckpt 的值排序，与按 Val AUC@5 峰值排序比较（Spearman）。

    python scripts/finetune/proxy_select.py Q0 Q4 P8 ... [--root $MOON_RESULTS/finetune] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

PROXIES = ("inl_med", "inl_ratio", "ok200")


def proxies(pred_jsonl):
    rows = [json.loads(line) for line in Path(pred_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    inl = np.array([r.get("n_inliers") or 0 for r in rows], float)
    nm = np.array([r.get("n_matches") or 0 for r in rows], float)
    ratio = np.where(nm > 0, inl / np.maximum(nm, 1), 0.0)
    return {"inl_med": float(np.median(inl)), "inl_ratio": float(np.median(ratio)), "ok200": float(np.mean(inl >= 200))}


def rank(x):
    return np.argsort(np.argsort(x))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--root", default=str(Path(os.environ.get("MOON_RESULTS", ".")) / "finetune"))
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    runs = {}
    for n in args.names:
        S = Path(args.root) / n / "sweep/S"
        m = json.loads((S / "metrics.json").read_text(encoding="utf-8"))["methods"]
        pts = []
        for k, v in m.items():
            st = int(k[4:])
            if st == 0 or "val" not in v or not (S / "preds" / k / "val.jsonl").exists():
                continue
            pts.append({"step": st, "auc5": v["val"]["summary"]["auc@5"], **proxies(S / "preds" / k / "val.jsonl")})
        runs[n] = sorted(pts, key=lambda p: p["step"])
    out = {"runs": runs, "within": {}, "across": {}}
    print("| run | Val 峰值 | " + " | ".join(f"按 {p} 选：AUC@5 (step)" for p in PROXIES) + " |")
    print("|---|---|" + "---|" * len(PROXIES))
    for n, pts in runs.items():
        best = max(pts, key=lambda p: p["auc5"])
        cells = []
        for p in PROXIES:
            pick = max(pts, key=lambda q: q[p])
            out["within"].setdefault(p, []).append(best["auc5"] - pick["auc5"])
            cells.append(f"{pick['auc5']:.3f} ({pick['step']})")
        print(f"| {n} | {best['auc5']:.3f} ({best['step']}) | " + " | ".join(cells) + " |")
    peak = np.array([max(p["auc5"] for p in pts) for pts in runs.values()])
    for p in PROXIES:
        sel = np.array([max(pts, key=lambda q: q[p])[p] for pts in runs.values()])
        rho = float(np.corrcoef(rank(sel), rank(peak))[0, 1]) if len(peak) > 2 else float("nan")
        winner = list(runs)[int(np.argmax(sel))]
        out["across"][p] = {"spearman": round(rho, 3), "winner": winner,
                            "winner_val_peak": round(float(peak[int(np.argmax(sel))]), 4)}
        out["within"][p] = {"mean_regret": round(float(np.mean(out["within"][p])), 4),
                            "max_regret": round(float(np.max(out["within"][p])), 4)}
    print(json.dumps({"within": out["within"], "across": out["across"],
                      "best_by_val": list(runs)[int(np.argmax(peak))]}, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
