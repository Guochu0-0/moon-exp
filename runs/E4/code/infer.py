"""对一个模型在 Val 全部 652 对上推理一次，记下仿射、内点、重要性图（见 common.py）。

    GPU=1 /opt/envs/loftr/bin/python runs/E4/code/infer.py loftr_zs [--limit N]

产物 runs/E4/raw/<模型>/val.npz（不进 git）：每对 A__<pair>（2×3，失败为 nan）、inl__<pair>（内点 N×4：光学 x,y、SAR x,y）、
imp__<pair>（LoFTR (8,2,80,80) / RoMa (2,64,64)，float16）。另写 val.jsonl：匹配数、内点数、耗时、与已有预测的仿射差。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common import MODELS, RUN, Runner, affine_change, key, ref_preds, val_pairs

from workbench.launch import begin, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    tag = f"infer_{a.model}"
    begin(RUN, tag, ["infer.py", a.model] + (["--limit", str(a.limit)] if a.limit else []),
          entry="runs/E4/code/infer.py", model=MODELS[a.model])
    status = "fail"
    try:
        out = RUN / "raw" / a.model
        out.mkdir(parents=True, exist_ok=True)
        pairs = val_pairs()[: a.limit or None]
        ref = ref_preds(a.model)
        r = Runner(a.model)
        store, t0 = {}, time.time()
        with open(out / "val.jsonl", "w", encoding="utf-8") as log:
            for i, p in enumerate(pairs):
                t = time.time()
                opt, sar = r.load_pair(p)
                A, inl, imp, n = r(opt, sar)
                k = key(p)
                store[f"A__{k}"] = np.full((2, 3), np.nan) if A is None else A
                store[f"inl__{k}"] = inl
                store[f"imp__{k}"] = imp
                rec = {"pair": p, "n_matches": n, "n_inliers": len(inl), "sec": round(time.time() - t, 3),
                       "diff_vs_ref_px": affine_change(A, ref.get(p)), "fail": A is None,
                       "ref_fail": ref.get(p) is None}
                log.write(json.dumps(rec) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"{a.model} {i + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
        np.savez(out / "val.npz", **store)
        status = "ok"
    finally:
        end(RUN, tag, status)


if __name__ == "__main__":
    main()
