"""真值诊断（「【RoMa·RL】诊断：用真值看粗级抽签后的结果是否普遍比不抽签差」#182）：只读 zero-shot AnyMatch-RoMa，不训练。

    python runs/E16/code/probe.py --n 200 [--group 16]

数据：训练集没有标注，改用 Val 有标注的 pair（用户 2026-10-21 定），均匀取 n 对；扰动同 E12（难版几何 + 擦除，不加光度）。
抽签同 E12（runs/E12/code/probe.py）：每格按 softmax(gm_cls / T) 独立抽锚点，其余照推理细化到尺度 1、按推理同款抽点 + RANSAC 拟合仿射。
不抽签那份取 mode（与 romatch 原前向一致，E12 已核对）。

真值：SAR 一侧施加了已知仿射 T，标注点 (p, q) 在扰动后的 SAR 上为 T(q)。
- 标注点误差（主指标，同评测协议与难版评测）：把预测仿射 A 换回原 SAR 坐标系（T⁻¹∘A），光学标注点映射后到 q 的平均距离，原始像素。
- 4 角误差：标注点最小二乘拟合仿射 A_lab（光学 → 原 SAR），图块 4 角经 T⁻¹∘A 与经 A_lab 的平均距离。
- 逐格真值落点：粗格中心经 T∘A_lab 落在扰动后 SAR 上的位置；只统计落在画面内的格。
RANSAC 失败的仿射误差记 ∞。

温度中心值（票里开跑前定的规则）：每对算「16 次抽签标注点误差的中位数 − 不抽签的标注点误差」（两者都是 ∞ 记 0），
五档里选这个差值在 n 对上的中位数 ≤ 1 px 的最高一档；都不满足取 0.1 并标记。

产物：runs/E16/extra/data/pairs.csv（逐对逐温度）、draws.csv（逐次抽签）、summary.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from finetune import config as C                    # noqa: E402
from finetune.data import PairSet                   # noqa: E402
from finetune.geom import CORNERS, HW, compose, inv_affine   # noqa: E402
from finetune.models import roma                    # noqa: E402
from finetune.parts import aug, rloo                # noqa: E402
from finetune.parts.pseudo import affine_norm, grid  # noqa: E402

CFG = {"model": {"name": "roma", "train_vgg": True, "lora_r": 8},
       "optim": {"lr": 3e-5},
       "aug": {"shift": 128.0, "shift_disk": True, "rot": 20.0, "scale": 0.2, "photo": False, "erase": True},
       "rloo": {}}
INF = float("inf")


def labels(ds, pair):
    """(光学 N×2, 原 SAR N×2)，与 workbench/dataset.py checkpoints 同口径（Label 里 y 存为负值）。"""
    a = np.loadtxt(ds.data.path(ds.src[pair], pair, "Label"), dtype=np.float64, ndmin=2)
    a[:, [1, 3]] *= -1
    return a[:, :2], a[:, 2:4]


def lstsq_affine(p, q):
    X = np.c_[p, np.ones(len(p))]
    return np.linalg.lstsq(X, q, rcond=None)[0].T          # 2×3


def amap(A, x):
    return x @ A[:, :2].T + A[:, 2]


def spearman(a, b):
    """秩相关（∞ 当作最大值；任一方全相同时为 nan）。"""
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def diff(med, e0):
    if np.isinf(med) and np.isinf(e0):
        return 0.0
    return med - e0


def pct(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, np.float64)
    x = x[~np.isnan(x)]
    if not len(x):
        return None
    out = {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs}
    fin = x[np.isfinite(x)]
    return out | {"mean_finite": round(float(fin.mean()), 4) if len(fin) else None, "n": int(len(x)),
                  "n_inf": int(np.isinf(x).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2, 0.1])
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    cfg = C.expand(CFG)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    model = roma.Model(cfg, a.weights_root)
    p = cfg["rloo"]
    ds = PairSet(a.data, a.split, model.resize, **model.input, seed=0, **aug.data_kw(cfg["aug"]))
    lab = [q for q in ds.pairs if ds.data.path(ds.src[q], q, "Label").exists()]
    ds.pairs = lab[:: max(1, len(lab) // a.n)][: a.n]
    print(f"{a.split}: {len(lab)} 对有标注，取 {len(ds.pairs)} 对", flush=True)
    dev = "cuda"
    rows, draws, secs = [], [], []
    cellpool = {tp: {"d_mode": [], "d_samp": []} for tp in a.temps}

    for i in range(len(ds)):
        it = ds[i]
        T = it["T"].numpy().astype(np.float64)
        Ti = inv_affine(T)
        po, qo = labels(ds, it["pair"])
        A_lab = lstsq_affine(po, qo)
        lab_res = float(np.linalg.norm(amap(A_lab, po) - qo, axis=1).mean())
        A_true = compose(T, A_lab)                              # 光学 → 扰动后 SAR
        c_true = amap(A_lab, CORNERS)

        def errs(A):
            if A is None:
                return INF, INF
            B = compose(Ti, A)
            return (float(np.linalg.norm(amap(B, po) - qo, axis=1).mean()),
                    float(np.linalg.norm(amap(B, CORNERS) - c_true, axis=1).mean()))

        i0, i1 = it["image0"][None].to(dev), it["image1"][None].to(dev)
        t0 = time.time()
        with torch.no_grad():
            f1, f2 = rloo.features(model, i0, i1)
            cls, cert = rloo.coarse(model, f1, f2)
            cls = cls.float()
            C_, h, w = cls.shape[1:]
            P = cls.softmax(1)
            mode = P[0].argmax(0)
            G_, _ = rloo.anchors(C_, cls.device)
            valid = rloo.sar_valid(i1[0])
            m_flow, m_ce = rloo.rollout(model, f1, f2, P, cert.float(), mode[None], p["chunk"])
            A0, n0 = rloo.fit(m_flow[0], m_ce[0], p["num"], p["ransac"])
            e0, c0 = errs(A0)
            r1_0 = float(rloo.r1_cells(m_flow[0], A0, p["ransac"], (h, w)).mean())
            nc0 = rloo.ncc(i0[0], i1[0], valid, A0)
            conf = (F.adaptive_avg_pool2d(m_ce.sigmoid(), (h, w))[0, 0] > 0.5).reshape(-1).cpu().numpy()
            # 粗格中心的真值落点（归一化坐标），只看落在画面内的格
            An = torch.from_numpy(affine_norm(A_true)).to(dev).float()
            xt = torch.einsum("ij,hwj->hwi", An[:, :2], grid(h, w, dev)) + An[:, 2]       # (h,w,2)
            inside = (xt.abs() < 1).all(-1).reshape(-1).cpu().numpy()
            d_mode = ((G_[mode] - xt).norm(dim=-1) * (HW / 2)).reshape(-1).cpu().numpy()  # 原网格 px
            for tp in a.temps:
                qt = (cls[0] / tp).softmax(0)
                k = torch.multinomial(qt.reshape(C_, -1).T, a.group, replacement=True).T.reshape(-1, h, w)
                flow, ce = rloo.rollout(model, f1, f2, P, cert.float(), k, p["chunk"])
                fits = rloo.fit_many(flow, ce, p["num"], p["ransac"])
                ee = np.array([errs(A) for A, _ in fits])                                   # (G,2)
                e, c = ee[:, 0], ee[:, 1]
                r1 = np.array([float(rloo.r1_cells(flow[g], A, p["ransac"], (h, w)).mean())
                               for g, (A, _) in enumerate(fits)])
                nc = np.array([rloo.ncc(i0[0], i1[0], valid, A) for A, _ in fits])
                d_samp = ((G_[k] - xt[None]).norm(dim=-1) * (HW / 2)).reshape(a.group, -1).cpu().numpy()
                nonmode = (k != mode).reshape(a.group, -1).cpu().numpy()
                cp = cellpool[tp]
                cp["d_mode"].append(d_mode[inside].astype(np.float32))
                cp["d_samp"].append(d_samp[:, inside].reshape(-1).astype(np.float32))
                for g in range(a.group):
                    draws.append({"pair": it["pair"], "temp": tp, "draw": g, "err": e[g], "corner_err": c[g],
                                  "ncc": nc[g], "r1": r1[g], "n_inl": fits[g][1]})
                top_n, top_r = int(np.argmax(nc)), int(np.argmax(r1))
                rank = lambda j: int((e < e[j]).sum()) + 1                                 # 1 = 组内真值误差最小
                ins = inside
                rows.append({
                    "pair": it["pair"], "temp": tp, "label_fit_res": lab_res, "n_labels": len(po),
                    "shift": float(np.linalg.norm(T[:, 2] - (np.eye(2) - T[:, :2]) @ [HW / 2, HW / 2])),
                    "err0": e0, "corner_err0": c0, "ncc0": nc0, "r1_0": r1_0, "n_inl0": n0,
                    "err_med": float(np.median(e)), "err_min": float(e.min()), "err_max": float(e.max()),
                    "diff_med": diff(float(np.median(e)), e0),
                    "corner_diff_med": diff(float(np.median(c)), c0),
                    "n_worse": int((e > e0).sum()), "n_better": int((e < e0).sum()),
                    "best_better": bool(e.min() < e0), "n_fail": int(np.isinf(e).sum()),
                    "rho_ncc": spearman(nc, e), "rho_r1": spearman(r1, e),
                    "err_top_ncc": float(e[top_n]), "rank_top_ncc": rank(top_n),
                    "err_top_r1": float(e[top_r]), "rank_top_r1": rank(top_r),
                    "inside_frac": float(ins.mean()),
                    "d_mode_med": float(np.median(d_mode[ins])) if ins.any() else np.nan,
                    "d_samp_med": float(np.median(d_samp[:, ins])) if ins.any() else np.nan,
                    "d_mode_med_conf": float(np.median(d_mode[ins & conf])) if (ins & conf).any() else np.nan,
                    "d_samp_med_conf": float(np.median(d_samp[:, ins & conf])) if (ins & conf).any() else np.nan,
                    "frac_samp_closer": float((d_samp[:, ins] < d_mode[ins][None]).mean()) if ins.any() else np.nan,
                    "frac_samp_farther": float((d_samp[:, ins] > d_mode[ins][None]).mean()) if ins.any() else np.nan,
                    "nonmode": float(nonmode[:, ins].mean()) if ins.any() else np.nan,
                })
        secs.append(time.time() - t0)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(ds)} {secs[-1]:.2f}s", flush=True)

    out = RUN / "extra" / "data"
    out.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: f"{v:.6g}" if isinstance(v, float) else v
    for name, rs in (("pairs.csv", rows), ("draws.csv", draws)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, list(rs[0]))
            wr.writeheader()
            for r in rs:
                wr.writerow({k: fmt(v) for k, v in r.items()})

    by_t, rule = {}, {}
    r0 = [r for r in rows if r["temp"] == a.temps[0]]
    for tp in a.temps:
        rs = [r for r in rows if r["temp"] == tp]
        col = lambda k: np.array([r[k] for r in rs], np.float64)
        e0, med = col("err0"), col("err_med")
        dm, ds_ = np.concatenate(cellpool[tp]["d_mode"]), np.concatenate(cellpool[tp]["d_samp"])
        rule[str(tp)] = float(np.median(col("diff_med")))
        by_t[str(tp)] = {
            "diff_med": pct(col("diff_med")), "corner_diff_med": pct(col("corner_diff_med")),
            "err_med": pct(med), "err_min": pct(col("err_min")),
            "n_worse": pct(col("n_worse")), "frac_all16_worse": round(float((col("n_worse") == a.group).mean()), 4),
            "frac_best_better": round(float(col("best_better").mean()), 4),
            "frac_med_worse": round(float((col("diff_med") > 0).mean()), 4),
            "n_fail_mean": round(float(col("n_fail").mean()), 4),
            "rho_ncc": pct(col("rho_ncc")), "rho_r1": pct(col("rho_r1")),
            "rank_top_ncc": pct(col("rank_top_ncc")), "rank_top_r1": pct(col("rank_top_r1")),
            "frac_top_ncc_below_med": round(float((col("err_top_ncc") < med).mean()), 4),
            "frac_top_r1_below_med": round(float((col("err_top_r1") < med).mean()), 4),
            "frac_top_ncc_beats_noSample": round(float((col("err_top_ncc") < e0).mean()), 4),
            "frac_top_r1_beats_noSample": round(float((col("err_top_r1") < e0).mean()), 4),
            "auc5_noSample": round(float(np.clip(1 - e0 / 5, 0, None).mean()), 4),
            "auc5_draws": round(float(np.mean([np.clip(1 - d["err"] / 5, 0, None) for d in draws if d["temp"] == tp])), 4),
            "cell_d_mode": pct(dm), "cell_d_samp": pct(ds_),
            "cell_frac_samp_closer": pct(col("frac_samp_closer")), "cell_frac_samp_farther": pct(col("frac_samp_farther")),
            "nonmode": pct(col("nonmode")),
        }
    ok = [tp for tp in a.temps if rule[str(tp)] <= 1.0]
    chosen = max(ok) if ok else 0.1
    summ = {"split": a.split, "n_labelled": len(lab), "n_pairs": len(ds), "group": a.group,
            "noSample": {"err0": pct([r["err0"] for r in r0]), "corner_err0": pct([r["corner_err0"] for r in r0]),
                         "label_fit_res": pct([r["label_fit_res"] for r in r0]), "shift": pct([r["shift"] for r in r0])},
            "rule_median_diff_px": rule, "chosen_temp": chosen, "rule_satisfied": bool(ok),
            "by_temp": by_t, "sec_per_pair": pct(secs)}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
