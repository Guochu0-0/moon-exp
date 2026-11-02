"""E39（#240）：把交换输入后的点对按 SAR → 光学估仿射（RANSAC 同主表），求逆后写 extra/data/swap_<模型>.jsonl。

    python runs/E39/code/swap_fit.py zero_shot|e37
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from moonlib.ransac import fit_affine
from workbench.dataset import Dataset

m = sys.argv[1]
raw = Path(f"runs/E39/raw/swap_{m}")
ds = Dataset(os.environ["MOON_DATA"])
n_ok = 0
with np.load(raw / "val.npz") as z, open(f"runs/E39/extra/data/swap_{m}.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for pair in ds.labelled("val"):
        k = pair.replace("/", "__")
        rec = {"pair": pair, "A_inv": None}
        if k in z.files:
            M = z[k]                        # (x_opt, y_opt, x_sar, y_sar, conf)
            A, inl, fail = fit_affine(M[:, [2, 3, 0, 1, 4]], 3.0)      # SAR → 光学，阈值在光学图上量
            if A is not None:
                rec["A_inv"] = cv2.invertAffineTransform(A).tolist()   # 光学 → SAR
                rec["n_inliers"] = int(inl.sum())
                n_ok += 1
            else:
                rec["fail"] = fail
        else:
            rec["fail"] = "no_matches"
        f.write(json.dumps(rec) + "\n")
print(m, "ok", n_ok)
