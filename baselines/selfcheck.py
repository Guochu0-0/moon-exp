"""Adapter self-check: match an optical patch against a shifted copy of itself; the fitted affine must recover the shift.
usage: python selfcheck.py <config.json> [n_pairs]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from baselines import adapters  # noqa: E402
from moonlib import inputs  # noqa: E402
from baselines.data import Data  # noqa: E402

cfg = json.loads(Path(sys.argv[1]).read_text())
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
import os  # noqa: E402

repo = Path.cwd() / cfg["repo"] if cfg.get("repo") else None
wc = cfg.get("weights")
w = (repo / wc[5:] if wc.startswith("repo:") else Path(os.environ["MOON_WEIGHTS"]) / wc) if wc else None
m = adapters.load(cfg["adapter"])(repo=repo, weights=w, device="cuda", **cfg.get("params", {}))
d = Data(os.environ["MOON_DATA"])
dx, dy = map(float, os.environ.get("SHIFT", "7,-5").split(","))
import cv2  # noqa: E402

for pair in d.pairs("val")[:n]:
    a = inputs.optical_div255(d.optical("val", pair))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    b = cv2.warpAffine(a, M, a.shape[::-1], flags=cv2.INTER_LINEAR)
    k0, k1, c = m.match(a, b)
    k0, k1 = np.asarray(k0, float), np.asarray(k1, float)
    A, inl = cv2.estimateAffine2D(k0, k1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print(pair, "n", len(k0), "inl", int(inl.sum()) if inl is not None else None,
          "t", None if A is None else np.round(A[:, 2], 3), "lin", None if A is None else np.round(A[:, :2].ravel(), 4),
          "median resid", None if A is None else round(float(np.median(np.linalg.norm(k1 - (k0 + [dx, dy]), axis=1))), 3))
