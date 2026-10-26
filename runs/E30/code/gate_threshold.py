"""E30（#216）：参照合理性检查里「画面内格子比例」的阈值。按票取 E28 三个健康种子（fm_s0–s2）训练日志里单步画面内格子比例
（log.jsonl 的 in_frame，batch 1 即每对一个值）合在一起的 1% 分位；另列每个种子的分位数，以及两个失效种子（fm_s3、fm_s4）
第一次跌破阈值的步，供 Notes 引用。只读 E28 的日志，CPU 上跑。

    /opt/envs/wb/bin/python runs/E30/code/gate_threshold.py /remote-home/xufang/YGC/moon-exp/runs/E28 > runs/E30/extra/gate_threshold.json
"""
import json
import sys
from pathlib import Path

import numpy as np


def series(ck):
    return np.array([json.loads(l).get("in_frame", np.nan) for l in open(ck / "log.jsonl", encoding="utf-8")], float)


def main():
    E28 = Path(sys.argv[1])
    s = {m: series(E28 / "ckpt" / m) for m in (f"fm_s{i}" for i in range(5))}
    healthy = np.concatenate([s[f"fm_s{i}"] for i in range(3)])
    healthy = healthy[np.isfinite(healthy)]
    thr = float(np.quantile(healthy, 0.01))
    out = {"threshold_q01": round(thr, 4), "n_steps": int(len(healthy)),
           "per_seed": {m: {"q001": round(float(np.nanquantile(v, 0.001)), 4), "q01": round(float(np.nanquantile(v, 0.01)), 4),
                            "q50": round(float(np.nanquantile(v, 0.5)), 4), "min": round(float(np.nanmin(v)), 4),
                            "frac_below": round(float(np.mean(v < thr)), 4),
                            "first_below": int(np.argmax(v < thr)) + 1 if (v < thr).any() else None}
                        for m, v in s.items()}}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
