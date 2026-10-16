"""E10 示例图（#150）：几对 Val 的光学与各 SAR 输入并排，写 extra/examples.png。另算各 SAR 输入与光学的梯度幅值相关，写 extra/data/grad_corr.csv。

    PYTHONPATH=. /opt/envs/loftr/bin/python runs/E10/code/examples.py      # 在 worktree 根目录跑（wb 环境没有 matplotlib）

梯度幅值相关：先按标注自拟合的仿射把 SAR 输入重采样到光学网格，两图各做 σ = 2 px 高斯平滑（压掉斑点）后取 Sobel 梯度幅值，
在重叠区域内算 Pearson 相关；
每对一个值，报 Val 全部有标注 pair 的中位数。它只描述两图的边缘是否落在同一处，不代表匹配器能否用上。
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from baselines.data import Data
from moonlib import inputs
from workbench.dataset import Dataset

R = Path("runs/E10")
root = os.environ.get("MOON_DATA", "/remote-home/xufang/YGC/dataset/Moon")
data, ds = Data(root), Dataset(root)
MAPS = ["p2p98", "oc", "sc", "mchi_vol", "mchi_odd", "mchi_even", "cpr", "m", "cpr_ml9", "m_ml9"]
LABEL = {"p2p98": "S1", "oc": "OC", "sc": "SC", "mchi_vol": "m-chi volume", "mchi_odd": "m-chi odd", "mchi_even": "m-chi even",
         "cpr": "CPR", "m": "m", "cpr_ml9": "CPR 9x9 looks", "m_ml9": "m 9x9 looks"}   # 服务器没有中文字体，图中用英文
pairs = ds.labelled("val")


def gradmag(x):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2.0)
    return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3))


rows = []
for p in pairs:
    po, ps = ds.checkpoints("val", p)
    if len(po) < 3:
        continue
    A, _ = cv2.estimateAffine2D(po.astype(np.float32), ps.astype(np.float32), method=cv2.LMEDS)
    if A is None:
        continue
    opt = inputs.optical_div255(data.optical("val", p))
    sar = data.sar("val", p)
    h, w = opt.shape
    go = gradmag(opt)
    mask = cv2.warpAffine(np.ones_like(opt), A, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST) > 0
    row = {"pair": p}
    for k in MAPS:
        s = cv2.warpAffine(inputs.get("sar", k)(sar), A, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        row[k] = float(np.corrcoef(go[mask], gradmag(s)[mask])[0, 1])
    rows.append(row)
out = R / "extra" / "data"
out.mkdir(parents=True, exist_ok=True)
with open(out / "grad_corr.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["pair"] + MAPS)
    wr.writeheader()
    wr.writerows(rows)
print("梯度幅值相关（中位数，n=%d）：" % len(rows), {k: round(float(np.median([r[k] for r in rows])), 3) for k in MAPS})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ex = pairs[::len(pairs) // 3][:3]
fig, axes = plt.subplots(len(ex), len(MAPS) + 1, figsize=(2.1 * (len(MAPS) + 1), 2.2 * len(ex)))
for i, p in enumerate(ex):
    sar = data.sar("val", p)
    axes[i, 0].imshow(inputs.optical_div255(data.optical("val", p)), cmap="gray", vmin=0, vmax=1)
    axes[i, 0].set_ylabel(p, fontsize=7)
    for j, k in enumerate(MAPS, 1):
        axes[i, j].imshow(inputs.get("sar", k)(sar), cmap="gray", vmin=0, vmax=1)
    for j, t in enumerate(["optical"] + [LABEL[k] for k in MAPS]):
        axes[0, j].set_title(t, fontsize=9)
for a in axes.ravel():
    a.set_xticks([]), a.set_yticks([])
fig.tight_layout()
fig.savefig(R / "extra" / "examples.png", dpi=110)
print("wrote", R / "extra" / "examples.png")
