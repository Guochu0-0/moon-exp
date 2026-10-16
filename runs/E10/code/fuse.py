"""后期融合（#150）：把两个输入各自推理出的点对按 pair 拼在一起，写成 baselines.match 同格式的原始目录，再交给 baselines.fit。

    python runs/E10/code/fuse.py runs/E10/raw/anymatch_loftr__s1 runs/E10/raw/anymatch_loftr__cpr --out runs/E10/raw/anymatch_loftr__fuse_s1_cpr --split val

任一来源报错的 pair 记为报错；sec 为两者之和。
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="val")
a = ap.parse_args()

srcs = [Path(s) for s in a.src]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
logs = [{json.loads(l)["pair"]: json.loads(l) for l in (s / f"{a.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        for s in srcs]
zs = [np.load(s / f"{a.split}.npz") for s in srcs]
store, recs = {}, []
for pair in logs[0]:
    rs = [lg.get(pair) for lg in logs]
    rec = {"pair": pair, "sec": round(sum((r or {}).get("sec") or 0 for r in rs), 3)}
    bad = [r.get("error", "missing") if r else "missing" for r in rs if r is None or "error" in r]
    if bad:
        rec["error"] = "; ".join(bad)
    else:
        k = pair.replace("/", "__")
        M = np.concatenate([z[k] if k in z.files else np.zeros((0, 5), np.float32) for z in zs])
        store[k] = M.astype(np.float32)
        rec["n"] = len(M)
    recs.append(rec)
np.savez(out / f"{a.split}.npz", **store)
(out / f"{a.split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
(out / f"{a.split}.meta.json").write_text(json.dumps({"method": out.name, "split": a.split, "fused_from": [str(s) for s in srcs]},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
print(out.name, len(recs), "pairs,", sum("error" in r for r in recs), "errors")
