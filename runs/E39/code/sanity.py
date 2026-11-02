"""E39（#240）开跑前核对：search.py 的相似度与 diagnose_reward.surface 逐点相同；两个模型答案的 Val AUC@5 与记录一致。

    MOON_DATA=... python runs/E39/code/sanity.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import search as S  # noqa: E402
from baselines import diagnose_reward as DR  # noqa: E402
from workbench import protocol  # noqa: E402
from workbench.dataset import Dataset  # noqa: E402

root = os.environ["MOON_DATA"]
ds = Dataset(root)
pairs = ds.labelled("val")
print("labelled", len(pairs))

for m, path in S.MODELS.items():
    preds = {json.loads(l)["pair"]: json.loads(l).get("A") for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()}
    err = np.array([protocol.pair_error(preds.get(p), *ds.checkpoints("val", p)) for p in pairs])
    print(m, {k: round(v, 4) for k, v in protocol.summarize(err).items() if k in ("auc@3", "auc@5", "sr@3", "sr@5", "median", "fail_rate")})

preds = {json.loads(l)["pair"]: json.loads(l).get("A") for l in Path(S.MODELS["e37"]).read_text(encoding="utf-8").splitlines() if l.strip()}
worst = 0.0
for p in pairs[::130]:
    sc = S.Scene(ds, p, "p2p98")
    A0 = np.asarray(preds[p], float)
    surf = DR.surface(DR.warp_feat(sc.Fo, A0, sc.shape), sc.Fs[None], "gradncc")
    for dy in (-6, -3, 0, 2, 6):
        for dx in (-6, 0, 5):
            # 同一个位移：slice 比较（网格阶段）、重新 warp（细化阶段）都要等于 surface
            a = S.ncc(sc.warped(A0), sc.ref, dx=dx, dy=dy)
            b = S.ncc(sc.warped(S.candidate(A0, sc.c, {"tx": -dx, "ty": -dy})), sc.ref)
            t = surf[dy + DR.R, dx + DR.R]
            worst = max(worst, abs(a - t), abs(b - t))
print("max |ncc - surface| =", worst)
A = np.array([[1.01, 0.02, 3.0], [-0.01, 0.98, -2.0]])
c = np.array([256.0, 256.0])
Ai = S.with_linear(np.eye(2), A, c)
print("center kept:", np.allclose(Ai[:, :2] @ c + Ai[:, 2], A[:, :2] @ c + A[:, 2]),
      "| identity params keep A:", np.allclose(S.candidate(A, c, {}), A))
