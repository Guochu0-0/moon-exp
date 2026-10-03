"""训练日志分段汇总（#49）：每 --bin 步一行，取各监控量的中位数。

    python scripts/finetune/trainlog.py <finetune>/<name>/log.jsonl [--bin 100] [--keys n_match,n_inl,...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

KEYS = "cexp,n_match,n_inl,n_ident,dist_I,diag,neg_n_match,neg_n_inl,neg_n_ident,neg_diag,gnorm"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--bin", type=int, default=100)
    ap.add_argument("--keys", default=KEYS)
    args = ap.parse_args(argv)
    rows = [json.loads(x) for x in open(args.log, encoding="utf-8") if x.strip()]
    keys = [k for k in args.keys.split(",") if any(k in r for r in rows)]
    print("steps".ljust(11) + " ".join(k[:10].rjust(10) for k in keys))
    for lo in range(0, rows[-1]["step"], args.bin):
        xs = [r for r in rows if lo < r["step"] <= lo + args.bin]
        cells = []
        for k in keys:
            v = [r[k] for r in xs if r.get(k) is not None]
            cells.append(f"{np.median(v):10.4g}" if v else " " * 9 + "—")
        print(f"{lo + 1}-{lo + args.bin}".ljust(11) + " ".join(cells))


if __name__ == "__main__":
    main()
