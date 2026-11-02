"""E39（#240）：对每个有标注的 Val 图对，构造票里的候选仿射，全部只用 CPU。不看标注（「标注拟合的线性部分」一行除外，它不搜索）。

    MOON_DATA=... python runs/E39/code/search.py --model zero_shot --out runs/E39/extra/data/rows_zero_shot.jsonl

相似度：两幅图梯度幅值的 NCC，特征、warp、比较区都用 baselines/diagnose_reward.py 的 gradmag、warp_feat、MARGIN：
光学梯度幅值按仿射 warp 到 SAR 网格，在 SAR 去掉 24 px 边框后与 warp 结果的有效重叠区上算；有效像素不足一半的候选不算。

参数化：候选 = D·L0 的线性部分 + 平移，使 patch 中心 c 映到 A0(c) + (tx, ty)。
D = 旋转(rot) · diag(sx, sy) · 剪切(shear)，共 4 个数，合起来 6 个数。
范围：平移各 ±8 px，旋转 ±2°，sx、sy 各 ±5%，剪切 ±2°。

搜索：先网格（平移步长 --t-step，线性部分每个数 --lin-levels 档），取最好的 --topk 个互相分开的起点，
再逐个做坐标搜索（每个数往两边试，没有改进就把步长减半，直到平移步长 --final 及线性部分的等比例步长）。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from baselines.diagnose_reward import MARGIN, fit_gt, gradmag, warp_feat  # noqa: E402
from moonlib import inputs  # noqa: E402
from workbench.dataset import Dataset  # noqa: E402

cv2.setNumThreads(0)

MODELS = {"zero_shot": "runs/B0m/preds/anymatch_roma__minmax/val.jsonl", "e37": "runs/E37/preds/all/val.jsonl"}
LIM = {"tx": 8.0, "ty": 8.0, "rot": 2.0, "sx": 0.05, "sy": 0.05, "shear": 2.0}   # 平移 px、角度 °、缩放为偏离 1 的比例
LIN = ("rot", "sx", "sy", "shear")
TRANS = ("tx", "ty")
# 中心到角约 362 px 处产生的位移约等于 1 px 时各线性数的大小：用来让线性部分的步长与平移步长同比例
UNIT = {"tx": 1.0, "ty": 1.0, "rot": np.degrees(1 / 362), "sx": 1 / 256, "sy": 1 / 256, "shear": np.degrees(1 / 256)}


def delta(p):
    th, k = np.radians(p.get("rot", 0.0)), np.radians(p.get("shear", 0.0))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = np.diag([1.0 + p.get("sx", 0.0), 1.0 + p.get("sy", 0.0)])
    H = np.array([[1.0, np.tan(k)], [0.0, 1.0]])
    return R @ S @ H


def with_linear(L, A0, c):
    """线性部分换成 L，保持 patch 中心 c 的映射位置等于 A0(c)。"""
    return np.c_[L, A0[:, :2] @ c + A0[:, 2] - L @ c]


def candidate(A0, c, p):
    A = with_linear(delta(p) @ A0[:, :2], A0, c)
    A[:, 2] += [p.get("tx", 0.0), p.get("ty", 0.0)]
    return A


def ncc(warped, ref, dx=0, dy=0):
    """SAR(p) 比 warped(p + d)，等价于把仿射的平移减去 d。有效重叠不足一半时返回 -inf（同 diagnose_reward.surface）。"""
    H, W = warped.shape
    m = MARGIN
    mov = warped[m + dy:H - m + dy, m + dx:W - m + dx]
    ok = np.isfinite(mov)
    if ok.mean() < 0.5:
        return -np.inf
    a, b = mov[ok], ref[ok]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


class Scene:
    def __init__(self, ds, pair, sar_map):
        opt = inputs.get("optical", "div255")(ds.optical("val", pair))
        sar = inputs.get("sar", sar_map)(ds.sar("val", pair))
        self.Fo, self.Fs = gradmag(opt), gradmag(sar)[0]
        self.shape = self.Fs.shape
        m = MARGIN
        self.ref = self.Fs[m:self.shape[0] - m, m:self.shape[1] - m]
        h, w = opt.shape
        self.c = np.array([w / 2.0, h / 2.0])
        self.evals = 0

    def warped(self, A):
        return warp_feat(self.Fo, A, self.shape)[0]

    def score(self, A):
        self.evals += 1
        return ncc(self.warped(A), self.ref)


def clamp(p):
    return {k: float(np.clip(v, -LIM[k], LIM[k])) for k, v in p.items()}


def refine(sc, A0, p, s, active, t_step, final):
    """坐标搜索：从平移步长 t_step 减半到 final；线性部分用同比例的步长（UNIT）。"""
    step = t_step
    while step >= final - 1e-9:
        for _ in range(30):
            moved = False
            for k in active:
                for sign in (1, -1):
                    q = clamp({**p, k: p.get(k, 0.0) + sign * step * UNIT[k]})
                    if q[k] == p.get(k, 0.0):
                        continue
                    v = sc.score(candidate(A0, sc.c, q))
                    if v > s + 1e-7:
                        p, s, moved = q, v, True
            if not moved:
                break
        step /= 2
    return p, s


def search(sc, A0, full, t_step, lin_levels, topk, final):
    """full：6 个数一起调；否则只调平移。返回最优的参数与相似度。"""
    ts = np.arange(-8, 8 + 1e-9, t_step)
    int_grid = abs(t_step - round(t_step)) < 1e-9
    lin_grid = list(itertools.product(*[np.linspace(-LIM[k], LIM[k], lin_levels) for k in LIN])) if full else [(0.0,) * 4]
    pts = []
    for lin in lin_grid:
        pl = dict(zip(LIN, map(float, lin))) if full else {}
        A = candidate(A0, sc.c, pl)
        W = sc.warped(A)
        for tx, ty in itertools.product(ts, ts):
            if not int_grid:    # 非整数步长只能重新 warp
                v = ncc(sc.warped(candidate(A0, sc.c, {**pl, "tx": tx, "ty": ty})), sc.ref)
            else:               # 仿射平移 +t 等于比较 warped(p − t)
                v = ncc(W, sc.ref, dx=-int(tx), dy=-int(ty))
            sc.evals += 1
            pts.append((v, lin, float(tx), float(ty)))
    pts.sort(key=lambda x: -x[0])
    starts = []
    for v, lin, tx, ty in pts:      # 起点之间要隔开：线性部分不同，或平移至少差两个网格步长
        if not np.isfinite(v):
            break
        if all(lin != b[1] or max(abs(tx - b[2]), abs(ty - b[3])) >= 2 * t_step for b in starts):
            starts.append((v, lin, tx, ty))
            if len(starts) == topk:
                break
    active = (TRANS + LIN) if full else TRANS
    best = None
    for v, lin, tx, ty in starts:
        p = {"tx": tx, "ty": ty, **(dict(zip(LIN, map(float, lin))) if full else {})}
        p, s = refine(sc, A0, p, v, active, t_step / 2 if t_step > final else t_step, final)
        if best is None or s > best[1]:
            best = (p, s)
    return best if best else ({}, -np.inf)


def one(job):
    root, pair, A0, sar_map, cfg, rows = job
    rec = {"pair": pair}
    if A0 is None:
        return {**rec, "fail": True}
    ds = Dataset(root)
    sc = Scene(ds, pair, sar_map)
    A0 = np.asarray(A0, float)
    rec["model"] = {"A": A0.tolist(), "ncc": sc.score(A0)}
    if "trans" in rows:
        p, s = search(sc, A0, False, cfg["t_step"], 1, cfg["topk_t"], cfg["final"])
        rec["trans"] = {"A": candidate(A0, sc.c, p).tolist(), "ncc": s, "p": p}
    if "full" in rows:
        p, s = search(sc, A0, True, cfg["t_step"], cfg["lin_levels"], cfg["topk"], cfg["final"])
        rec["full"] = {"A": candidate(A0, sc.c, p).tolist(), "ncc": s, "p": p}
    if "identity" in rows:
        A1 = with_linear(np.eye(2), A0, sc.c)
        p, s = search(sc, A1, False, cfg["t_step"], 1, cfg["topk_t"], cfg["final"])
        rec["identity"] = {"A": candidate(A1, sc.c, p).tolist(), "ncc": s, "p": p}
    if "gt_linear" in rows:     # 只作提升空间的参照：线性部分来自标注，不搜索
        o, s_ = ds.checkpoints("val", pair)
        A = with_linear(fit_gt(o, s_)[:, :2], A0, sc.c)
        rec["gt_linear"] = {"A": A.tolist(), "ncc": sc.score(A)}
    rec["evals"] = sc.evals
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--sar", default="p2p98", help="算相似度时 SAR 的映射；模型的输入不受影响")
    ap.add_argument("--rows", default="trans,full,identity,gt_linear")
    ap.add_argument("--t-step", type=float, default=2.0)
    ap.add_argument("--lin-levels", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--topk-t", type=int, default=4)
    ap.add_argument("--final", type=float, default=1 / 16)
    ap.add_argument("--every", type=int, default=1, help="每隔几对取一对，步长核对时用")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    preds = {}
    for line in Path(MODELS[args.model]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            preds[d["pair"]] = d.get("A")
    pairs = Dataset(args.data).labelled("val")[::args.every]
    cfg = dict(t_step=args.t_step, lin_levels=args.lin_levels, topk=args.topk, topk_t=args.topk_t, final=args.final)
    rows = args.rows.split(",")
    jobs = [(args.data, p, preds.get(p), args.sar, cfg, rows) for p in pairs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool, open(out, "w", encoding="utf-8", newline="\n") as f:
        for i, rec in enumerate(pool.imap(one, jobs, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{args.model} {i + 1}/{len(jobs)}", flush=True)
    print(f"done {args.model} {len(jobs)} {out}", flush=True)


if __name__ == "__main__":
    main()
