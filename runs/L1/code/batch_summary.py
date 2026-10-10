"""第四批（L1–L7）各方法的汇总数字，供各实验的 Notes 引用（只读，不写文件）。

    /opt/envs/wb/bin/python runs/L1/code/batch_summary.py

每个方法：Val 后段均值（第 2000–8000 步 7 个存档的 Val AUC@5 平均）、Val 最高及其步数、正式结果的 Test AUC@5、
第 7000–7999 步的平均训练损失与尺度 1 上相对目标的中位 EPE（原网格 px）、显存峰值、每步耗时中位数，
以及方法特有的监控量（对的权重、可匹配像素比例、α）。参照为 T13、T14 的同一配方。
"""
import glob
import json
import re
import statistics as S
from pathlib import Path

RUNS = [f"L{i}" for i in range(1, 8)] + ["T13", "T14"]


def val_curve(e, m):
    d = json.loads(Path(f"runs/{e}/sweep/{m}/S/metrics.json").read_text(encoding="utf-8"))["methods"]
    return {int(re.sub(r"\D", "", k)): v["val"]["summary"]["auc@5"] for k, v in d.items() if "val" in v}


def log_stats(e, m):
    f = Path(f"runs/{e}/ckpt/{m}/log.jsonl")
    if not f.exists():
        return {}
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    late = [r for r in rows if 7000 <= r["step"] < 8000]
    out = {"train_loss": S.mean(r["loss"] for r in late),
           "epe1_px": S.median(r["epe1_px"] for r in late if r.get("epe1_px") is not None),
           "mem_gb": rows[0].get("mem_gb"), "sec": S.median(r["sec"] for r in rows[1:])}
    for k in ("pair_w", "match_frac"):
        xs = [r[k] for r in rows if r.get(k) is not None]
        if xs:
            out[k] = (S.mean(xs), min(xs), max(xs))
    if any("ema_alpha" in r for r in rows):
        out["ema_alpha_8000"] = rows[-1]["ema_alpha"]
    return out


for e in RUNS:
    mt = json.loads(Path(f"runs/{e}/metrics.json").read_text(encoding="utf-8"))["methods"]
    for m in sorted(mt):
        if not Path(f"runs/{e}/sweep/{m}/S/metrics.json").exists():
            continue
        v = val_curve(e, m)
        late = [a for s, a in v.items() if s >= 2000]
        pk = max((s for s in v if s > 0), key=v.get)
        test = mt[m].get("test", {}).get("summary", {}).get("auc@5")
        print(json.dumps({"exp": e, "method": m, "val_late": round(S.mean(late), 4), "n_late": len(late),
                          "val_peak": round(v[pk], 4), "peak_step": pk,
                          "test": None if test is None else round(test, 4),
                          "curve": {s: round(a, 4) for s, a in sorted(v.items())},
                          **{k: (round(x, 4) if isinstance(x, float) else
                                 [round(y, 4) for y in x] if isinstance(x, tuple) else x)
                             for k, x in log_stats(e, m).items()}}, ensure_ascii=False))
