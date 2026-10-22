"""定锚点（#197）：读 _anchor 试跑的 log.jsonl（细化部分学习率 = 粗级 decoder 的 3e-5），对第 11–100 步取各部分参数每步
相对改变量（upd_*）的中位数，并算粗级各口径 ÷ 细化部分（fine：尺度 16、8、4 细化层与 proj + VGG）的比值。
AdamW 下每步改变量大致与学习率成正比，所以这个比值就是让两者每步改变量相等的细化学习率倍数。

    python runs/E22/code/anchor.py runs/E22/extra/data/anchor_log.jsonl
"""
import json
import sys

import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
rows = [r for r in rows if 10 < r.get("step", 0) <= 100]
keys = sorted({k for r in rows for k in r if k.startswith("upd_")})
med = {k: float(np.median([r[k] for r in rows if k in r])) for k in keys}
for k in keys:
    v = [r[k] for r in rows if k in r]
    print(f"{k:12s} median {med[k]:.3e}  p10 {np.percentile(v, 10):.3e}  p90 {np.percentile(v, 90):.3e}")
for k in ("upd_coarse", "upd_lora"):
    if k in med and med.get("upd_fine"):
        print(f"{k} / upd_fine = {med[k] / med['upd_fine']:.4g}")
for k in ("d_ref", "d_16", "d_8", "d_4", "d90_16", "d90_8", "d90_4", "rew", "rew_16", "rew_8", "rew_4", "zero_16", "zero_8", "zero_4", "sec"):
    v = [r[k] for r in rows if k in r]
    if v:
        print(f"{k:8s} mean {np.mean(v):.4g}  median {np.median(v):.4g}")
