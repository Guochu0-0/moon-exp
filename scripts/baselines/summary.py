"""Summarize runs/B0/metrics.json + raw match logs into a markdown table."""
import json
from pathlib import Path

R = Path("/remote-home/xufang/YGC/results/baselines")
m = json.load(open("runs/B0/metrics.json"))["methods"]
order = ["identity", "loftr", "xoftr", "spsg", "rift2", "minima_loftr", "minima_xoftr", "minima_splg", "minima_roma",
         "minima_eloftr", "anymatch_loftr", "anymatch_edm", "anymatch_roma", "ma_eloftr", "ma_roma", "geoformer"]
keys = ["auc@3", "auc@5", "auc@10", "auc@20", "sr@3", "sr@5", "sr@10", "fail_rate", "mis_rate"]
for s in ("val", "test"):
    print(f"\n### {s}\n")
    print("| method | " + " | ".join(keys) + " | CI auc@10 | match err | median n | sec/pair |")
    print("|" + "---|" * (len(keys) + 5))
    for k in order:
        if k not in m or s not in m[k]:
            continue
        x = m[k][s]["summary"]
        errs, ns, secs = 0, [], []
        log = R / k / f"{s}.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                d = json.loads(line)
                errs += "error" in d
                if "n" in d:
                    ns.append(d["n"])
                if "sec" in d:
                    secs.append(d["sec"])
        ns.sort(); secs.sort()
        ci = x.get("auc@10_ci") or [float("nan")] * 2
        row = [f"{x[q]:.3f}" for q in keys]
        print(f"| {k} | " + " | ".join(row) + f" | {ci[0]:.3f}–{ci[1]:.3f} | {errs} | "
              f"{ns[len(ns)//2] if ns else '-'} | {secs[len(secs)//2] if secs else '-'} |")
