"""RIFT2（MATLAB）的 Python 两端：prep 把输入映射好写成 .mat，collect 把 MATLAB 输出收成与 baselines.match 相同的产物。

    python -m baselines.rift2 prep    configs/baselines/rift2.json --split val --io /opt/tmp/rift2
    /root/matlab-batch.sh "addpath('baselines/matlab'); addpath('third_party/RIFT2'); rift2_split('/opt/tmp/rift2/val/in','/opt/tmp/rift2/val/out',0,4)"
    python -m baselines.rift2 collect configs/baselines/rift2.json --split val --io /opt/tmp/rift2 --out $MOON_RESULTS/baselines

之后与其他方法一样用 baselines.fit。中间 .mat 放容器本地盘（--io），不放 gpfs。需要 scipy。
坐标：RIFT2 输出 MATLAB 1-based、像素中心为整数、已 round（kptsOrientation.m:20-21），减 1 即我们的约定。
使用 FSC 之前的原始匹配（去重后），统一的仿射 RANSAC 在 fit 里做；官方 FSC 是 similarity 模型、无种子、M ≤ 2 时死循环。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import inputs
from .data import Data
from .match import REPO, env_info, git_head, key


def prep(cfg, split, data, io, limit=0):
    from scipy.io import savemat

    d = Path(io) / split / "in"
    d.mkdir(parents=True, exist_ok=True)
    mo, ms = inputs.get("optical", cfg["input"]["optical"]), inputs.get("sar", cfg["input"]["sar"])
    pairs = data.pairs(split)[:limit or None]
    for p in pairs:
        f = d / f"{key(p)}.mat"
        if not f.exists():
            savemat(f, {"opt": mo(data.optical(split, p)), "sar": ms(data.sar(split, p))}, do_compression=False)
    print(f"prep {split}: {len(pairs)} pairs -> {d}")


def collect(cfg, split, data, io, out, limit=0):
    from scipy.io import loadmat

    src = Path(io) / split / "out"
    dst = Path(out) / cfg["method"]
    dst.mkdir(parents=True, exist_ok=True)
    store, n_err, n_missing = {}, 0, 0
    pairs = data.pairs(split)[:limit or None]
    with open(dst / f"{split}.jsonl", "w", encoding="utf-8") as log:
        for p in pairs:
            f = src / f"{key(p)}.mat"
            if not f.exists():
                n_missing += 1
                continue
            r = loadmat(f)
            err = str(r["err"][0]) if r["err"].size else ""
            rec = {"pair": p, "sec": round(float(r["sec"].ravel()[0]), 3)}
            if err:
                rec["error"] = err
                n_err += 1
            else:
                p1 = np.asarray(r["p1"], np.float64).reshape(-1, 2) - 1.0
                p2 = np.asarray(r["p2"], np.float64).reshape(-1, 2) - 1.0
                store[key(p)] = np.c_[p1, p2, np.ones(len(p1))].astype(np.float32)
                rec["n"] = len(p1)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
    np.savez_compressed(dst / f"{split}.npz", **store)
    meta = {"method": cfg["method"], "split": split, "config": cfg, "commit": git_head(REPO),
            "repo_commit": git_head(REPO / cfg["repo"]), "weights": None, "weights_sha256": None,
            "notes": "MATLAB rift2_split.m: float input, no im2uint8, raw matches before FSC, 1-based -> 0-based",
            "env": env_info(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "n_pairs": len(pairs),
            "n_missing": n_missing, "n_errors": n_err}
    (dst / f"{split}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"collect {split}: {len(store)} ok, {n_err} errors, {n_missing} missing -> {dst}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "collect"))
    ap.add_argument("config")
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--io", required=True, help="中间 .mat 目录（容器本地盘）")
    ap.add_argument("--out", help="collect 的输出根目录，同 baselines.match --out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    data = Data(a.data)
    if a.step == "prep":
        prep(cfg, a.split, data, a.io, a.limit)
    else:
        if not a.out:
            sys.exit("collect 需要 --out")
        collect(cfg, a.split, data, a.io, a.out, a.limit)


if __name__ == "__main__":
    main()
