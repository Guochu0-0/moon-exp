"""新旧两份伪标签的前 50%（按内点数，同 finetune.label.load_labels）有多少对重合，供样本筛选参考（#130）。

    python runs/N/code/overlap.py <旧标签.jsonl> <新标签.jsonl> --json runs/N/extra/overlap_<m>.json

输出：两份各自通过门槛（keep）的对数、前 50% 的对数、前 50% 的交集占新前 50% 的比例，以及交集上两份伪仿射
在 512 网格四个角点处的平均差（px），看重合的对上标签本身变了多少。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from finetune.label import load_labels  # noqa: E402

CORNERS = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--top", type=float, default=0.5)
    ap.add_argument("--teacher", default="", help="新标签的老师（只写进输出）")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    keep = lambda p: sum(json.loads(s)["keep"] for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip())
    old, new = load_labels(a.old, a.top), load_labels(a.new, a.top)
    so = {q for q, A in old.items() if A is not None}
    sn = {q for q, A in new.items() if A is not None}
    both = sorted(so & sn)
    d = [float(np.linalg.norm(CORNERS @ (np.asarray(new[q]) - np.asarray(old[q])).T, axis=1).mean()) for q in both]
    res = {"old": a.old, "new": a.new, "teacher": a.teacher, "top": a.top,
           "keep_old": keep(a.old), "keep_new": keep(a.new), "top_old": len(so), "top_new": len(sn),
           "overlap": len(both), "overlap_frac": round(len(both) / max(len(sn), 1), 4),
           "corner_diff_px_median": round(float(np.median(d)), 3) if d else None,
           "corner_diff_px_mean": round(float(np.mean(d)), 3) if d else None}
    print(json.dumps(res, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
