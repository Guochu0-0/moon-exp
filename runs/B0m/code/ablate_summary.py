"""Compare SAR input mappings on Val: B0 (p2p98) vs B0m (<m>__zscore_2p5, <m>__minmax)."""
import json
from pathlib import Path

B = json.load(open("runs/B0/metrics.json"))["methods"]
A = json.load(open("runs/B0m/metrics.json"))["methods"]
RAW = {"p2p98": Path("/remote-home/xufang/YGC/results/baselines"),
       "zscore_2p5": Path("/remote-home/xufang/YGC/results/baselines_ablation"),
       "minmax": Path("/remote-home/xufang/YGC/results/baselines_ablation")}
order = ["loftr", "xoftr", "spsg", "minima_loftr", "minima_xoftr", "minima_splg", "minima_roma", "minima_eloftr",
         "anymatch_loftr", "anymatch_edm", "anymatch_roma", "ma_eloftr", "ma_roma", "geoformer"]


def median_n(path):
    ns = sorted(json.loads(l).get("n", 0) for l in path.read_text().splitlines() if l.strip()) if path.exists() else []
    return ns[len(ns) // 2] if ns else "-"


print("| method | map | auc@10 | sr@10 | sr@3 | fail | mis | median n |")
print("|---|---|---|---|---|---|---|---|")
for m in order:
    for mp in ("p2p98", "zscore_2p5", "minmax"):
        key = m if mp == "p2p98" else f"{m}__{mp}"
        src = B if mp == "p2p98" else A
        if key not in src or "val" not in src[key]:
            continue
        x = src[key]["val"]["summary"]
        n = median_n(RAW[mp] / key / "val.jsonl")
        print(f"| {m} | {mp} | {x['auc@10']:.3f} | {x['sr@10']:.3f} | {x['sr@3']:.3f} | {x['fail_rate']:.3f} | "
              f"{x['mis_rate']:.3f} | {n} |")
