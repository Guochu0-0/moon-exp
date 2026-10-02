"""SCENES 式离线伪标签：用起点模型在 Train 上推理一遍，每对估一个伪仿射（「复现 SCENES 式伪标签微调 baseline」#26）。

    python -m finetune.label configs/baselines/anymatch_loftr.json --out $MOON_RESULTS/finetune/labels_b0.jsonl

筛选照 SCENES（§4.2）：匹配数 ≥ 100 且内点 ≥ 20，否则 keep=false，训练时不监督。
每行 {"pair", "A"（原网格 2×3，角点约定，同 workbench）, "n_match", "n_inliers", "keep"}。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .data import PairSet
from .model import Base
from .pseudo import fit_pseudo

REPO = Path(__file__).resolve().parents[1]
HW = 512
MIN_MATCHES, MIN_INLIERS = 100, 20


def load_labels(path, top=1.0):
    """label.py 的输出 → {pair: A（list）或 None}。keep=false 的记 None。
    top < 1：只留 keep 的对里内点数最多的前 top 比例（课程式取子集，#53 调研 C4），其余记 None。"""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = sorted((d for d in rows if d["keep"]), key=lambda d: -d["n_inliers"])
    ok = {d["pair"] for d in kept[: int(round(len(kept) * top))]}
    return {d["pair"]: d["A"] if d["pair"] in ok else None for d in rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--init", help="起点 ckpt，默认配置里的底座权重")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    weights = args.init or str(Path(args.weights_root) / cfg["weights"])
    base = Base(REPO / cfg["repo"], weights, device=args.device, **cfg.get("params", {}))
    s = HW / base.long_side
    ds = PairSet(args.data, "train", base.resize, limit=args.limit, **cfg["input"])
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, t0 = 0, time.time()
    with open(out, "w", encoding="utf-8") as f, torch.no_grad():
        for it, batch in enumerate(dl):
            data = base.forward(batch["image0"].to(args.device), batch["image1"].to(args.device))
            k0 = data["mkpts0_f"].cpu().numpy().astype("float64")
            k1 = data["mkpts1_f"].cpu().numpy().astype("float64")
            A, n = fit_pseudo(k0, k1, s, args.ransac)
            keep = A is not None and len(k0) >= MIN_MATCHES and n >= MIN_INLIERS
            kept += keep
            f.write(json.dumps({"pair": batch["pair"][0], "A": None if A is None else A.round(6).tolist(),
                                "n_match": len(k0), "n_inliers": n, "keep": bool(keep)}) + "\n")
            if (it + 1) % 500 == 0:
                print(f"{it + 1}/{len(ds)} kept={kept} {(time.time() - t0) / 60:.1f}min", flush=True)
    print(f"done: {len(ds)} pairs, kept {kept}")


if __name__ == "__main__":
    main()
