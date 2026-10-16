"""复合自检（#138）：把 E1 的标注自拟合仿射 A 写成 E7 的 gtfit（原样）与 gtfit__<变体>（T∘A）。
理论上每对误差不变，只有画面外被剔除的检查点会让个别 pair 的误差变化；用来确认 T 的存储、复合与评测口径一致。

    /opt/envs/wb/bin/python runs/E7/code/gtfit.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from moonlib import perturb as P  # noqa: E402
from workbench.records import PredWriter  # noqa: E402

R = REPO / "runs" / "E7"


def compose(T, A):
    H = lambda M: np.r_[np.asarray(M, np.float64).reshape(2, 3), [[0, 0, 1]]]
    return (H(T) @ H(A))[:2]


for split in ("val", "test"):
    rows = [json.loads(l) for l in (REPO / "runs/E1/preds/gtfit" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    for v in [None, *P.VARIANTS]:
        Ts = P.load(R, v, split) if v else {}
        with PredWriter(R, f"gtfit__{v}" if v else "gtfit", split) as w:
            for d in rows:
                A = d["A"] if d["A"] is None or v is None else compose(Ts[d["pair"]], d["A"]).tolist()
                w.write(d["pair"], A, fail=d.get("fail"))
    print(split, len(rows), "pairs")
