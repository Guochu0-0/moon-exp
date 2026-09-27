"""PROTOTYPE（节点页，#20）：从本地数据集渲染「可视化结果」用的图，输出 img/ 和 pairs.js（都不进 git）。

    py -3.11 workbench/static/prototype-node-page/render_pairs.py G:/Lunar_Optical_SAR_Registration_Dataset

只渲染原型会用到的一小批 Val pair：
改善最多、退化最多、本方法误差最大（本方法 = B0m/anymatch_roma__minmax，参考 = B0/anymatch_roma），
参考方法误差最大（给 B0 页），以及几个两者都配准的样例。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from workbench.dataset import Dataset, to_display  # noqa: E402
from workbench.records import load_runs  # noqa: E402

ME, REF = ("B0m", "anymatch_roma__minmax"), ("B0", "anymatch_roma")
SPLIT, N, CELL, ARROW = "val", 12, 64, 1.0
OUT = HERE / "img"
OUT.mkdir(exist_ok=True)

ds = Dataset(sys.argv[1])
runs = load_runs(REPO / "runs")
labelled = ds.labelled(SPLIT)
allp = ds.pairs(SPLIT)
preds = {f"{r}/{m}": runs[r].preds(m, SPLIT) for r, m in (ME, REF)}
metrics = {f"{r}/{m}": json.loads((REPO / "runs" / r / "metrics.json").read_text(encoding="utf-8"))["methods"][m][SPLIT]["errors"]
           for r, m in (ME, REF)}
kme, kref = f"{ME[0]}/{ME[1]}", f"{REF[0]}/{REF[1]}"
cap = lambda e: 50.0 if e is None else min(e, 50.0)
em, er = [cap(e) for e in metrics[kme]], [cap(e) for e in metrics[kref]]
idx = range(len(labelled))
gain = sorted(idx, key=lambda i: er[i] - em[i], reverse=True)[:N]
loss = sorted(idx, key=lambda i: em[i] - er[i], reverse=True)[:N]
worst_me = sorted(idx, key=lambda i: em[i], reverse=True)[:N]
worst_ref = sorted(idx, key=lambda i: er[i], reverse=True)[:N]
both_ok = [i for i in idx if em[i] <= 2 and er[i] <= 2][:6]
chosen = sorted(set(gain + loss + worst_me + worst_ref + both_ok))
print(len(chosen), "pairs")


def gray(a):
    return Image.fromarray(to_display(a)).convert("L")


def warp(sar_img, A, size):
    # 输出光学坐标 (x, y) 处取 SAR 在 A·[x, y, 1] 的值；PIL AFFINE 正是这个映射
    a = np.asarray(A, dtype=np.float64)
    return sar_img.transform(size, Image.AFFINE, tuple(a.reshape(-1)), resample=Image.BILINEAR)


def checker(opt, wsar):
    o, w = np.asarray(opt), np.asarray(wsar)
    yy, xx = np.mgrid[:o.shape[0], :o.shape[1]]
    m = ((yy // CELL + xx // CELL) % 2).astype(bool)
    return Image.fromarray(np.where(m, w, o)).convert("RGB")


def arrows(img, A, opt_pts, sar_pts):
    # 检查点：光学点位（圆）→ 按估计仿射，真实 SAR 点落回光学坐标的位置；ARROW = 1 即原尺寸
    a = np.vstack([np.asarray(A, dtype=np.float64), [0, 0, 1]])
    inv = np.linalg.inv(a)
    d = ImageDraw.Draw(img)
    for (ox, oy), (sx, sy) in zip(opt_pts, sar_pts):
        tx, ty, _ = inv @ [sx, sy, 1]
        ex, ey = ox + (tx - ox) * ARROW, oy + (ty - oy) * ARROW
        d.line([(ox, oy), (ex, ey)], fill=(230, 60, 40), width=2)
        L = math.hypot(ex - ox, ey - oy)
        if L > 4:
            ux, uy = (ex - ox) / L, (ey - oy) / L
            d.polygon([(ex, ey), (ex - 7 * ux + 4 * uy, ey - 7 * uy - 4 * ux), (ex - 7 * ux - 4 * uy, ey - 7 * uy + 4 * ux)], fill=(230, 60, 40))
        d.ellipse([ox - 3.5, oy - 3.5, ox + 3.5, oy + 3.5], outline=(255, 214, 0), width=2)
    return img


def save(img, name):
    img.save(OUT / name, quality=82)


info = {}
for i in chosen:
    pair = labelled[i]
    stem = pair.replace("/", "__")
    opt, sar = gray(ds.optical(SPLIT, pair)), gray(ds.sar(SPLIT, pair))
    save(opt, f"{stem}__opt.jpg")
    save(sar, f"{stem}__sar.jpg")
    op, sp = ds.checkpoints(SPLIT, pair)
    got = {}
    for k in (kme, kref):
        pr = preds[k].get(pair)
        tag = k.replace("/", "__")
        if pr is None or pr.A is None:
            got[k] = {"fail": (pr.fail if pr else "no_pred") or "fail"}
            continue
        w = warp(sar, pr.A, opt.size)
        save(w, f"{stem}__{tag}__warp.jpg")
        save(arrows(checker(opt, w), pr.A, op, sp), f"{stem}__{tag}__chk.jpg")
        got[k] = {"ok": True}
    info[pair] = {"no": allp.index(pair), "methods": got}

data = {"split": SPLIT, "labelled": labelled, "no": [allp.index(p) for p in labelled], "rendered": info,
        "alias": {"E1/main": kme}, "arrow": ARROW, "cell": CELL}
(HERE / "pairs.js").write_text("window.PAIRS = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
print("wrote pairs.js; images:", len(list(OUT.glob("*.jpg"))))
