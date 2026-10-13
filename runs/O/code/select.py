"""按规则从一份伪标签里选对，写出同格式的标签文件：没选中的对 keep 改为 false，训练时配 [pseudo] top = 1.0（#131）。

    # 与另一份标签相比，角点位置平均相差不超过 3 px 的对才用
    python runs/O/code/select.py <新标签.jsonl> --ref <旧标签.jsonl> --max-diff 3 --out runs/O/raw/sel_x.jsonl
    # 按 runs/O0/code/pair_metrics.py 算出的某个度量取最可信的前一半（--sign -1：越小越可信）
    python runs/O/code/select.py <标签.jsonl> --metrics <度量.jsonl> --by resid_med --sign -1 --top 0.5 --out …

两条规则可以同时给，先按 --ref 去掉，再在剩下的对里按度量取前 --top。角点位置差的定义同 runs/N/code/overlap.py。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CORNERS = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], float)


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--max-diff", type=float, default=3.0)
    ap.add_argument("--metrics")
    ap.add_argument("--by")
    ap.add_argument("--sign", type=int, default=1, choices=(1, -1))
    ap.add_argument("--top", type=float, default=1.0)
    a = ap.parse_args(argv)
    rows = read(a.labels)
    ok = {d["pair"] for d in rows if d["keep"]}
    n0 = len(ok)
    info = {"labels": a.labels, "keep_in": n0}
    if a.ref:
        ref = {d["pair"]: d for d in read(a.ref)}
        diff = {}
        for d in rows:
            r = ref.get(d["pair"])
            if d["pair"] in ok and r is not None and r["A"] is not None:
                diff[d["pair"]] = float(np.linalg.norm(CORNERS @ (np.asarray(d["A"]) - np.asarray(r["A"])).T, axis=1).mean())
        ok = {q for q in ok if diff.get(q, math.inf) <= a.max_diff}
        v = np.array(list(diff.values()))
        info.update(ref=a.ref, max_diff=a.max_diff, after_ref=len(ok), diff_median=round(float(np.median(v)), 3),
                    diff_quantiles={str(p): round(float(np.percentile(v, p)), 3) for p in (25, 50, 75, 90, 95)})
    if a.by:
        met = {d["pair"]: d[a.by] for d in read(a.metrics)}
        order = sorted((q for q in (d["pair"] for d in rows) if q in ok), key=lambda q: -a.sign * met[q])
        ok = set(order[: int(round(len(order) * a.top))])
        info.update(metrics=a.metrics, by=a.by, sign=a.sign, top=a.top)
    info["keep_out"] = len(ok)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(dict(d, keep=d["pair"] in ok)) + "\n" for d in rows), encoding="utf-8")
    tmp.replace(out)
    out.with_suffix(".info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
