"""pair_metrics.py 的输出 → 各可信度度量的检验表（#131；做法见调研 #133 第四节：有真值时看度量与真实误差的关系，
没有真值时看度量之间的秩相关和各自选出的对的重合）。

    # Val：每个度量与真实误差的关系
    python runs/O0/code/table.py val runs/O0/extra/data/val_t1.jsonl --data $MOON_DATA \
        --alt gen=runs/O0/extra/data/val_t0.jsonl --out runs/O0/extra/data/val_t1_table.csv
    # Train：度量两两的秩相关，以及各自选出的前若干比例与内点数选出的重合
    python runs/O0/code/table.py train runs/O0/extra/data/train_t1.jsonl \
        --alt gen=<labels_roma0.jsonl> --out runs/O0/extra/data/train_t1

--alt 名字=另一份标签（任何带 pair、A 的 jsonl）：增加一个度量 diff_<名字>，即两份伪仿射把 512 网格四个角点
映到的位置的平均距离（px，同 runs/N/code/overlap.py）；另一份里没有这一对或 A 为空时记 inf。

Val 表的列：spearman 是「度量判为越不可信」与「真实误差越大」的秩相关；med_top20 / med_top50 是按度量取最可信的
20% / 50% 后真实误差的中位数（px）；le3_top50 是这 50% 里真实误差不超过 3 px 的比例；ause 是稀疏化曲线
（按度量从最不可信起逐步去掉 0%–95% 的对，剩余的对截断到 20 px 的平均误差）与按真实误差去掉的理想曲线之差的平均，
越小越好，random 一行是不按任何度量去掉时的值（Ilg 等 1802.07095 §5）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

CORNERS = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], float)
# 度量 → 符号：+1 表示越大越可信，-1 表示越小越可信
SIGN = {"n_inliers": 1, "inlier_ratio": 1, "conf_mean": 1, "conf_med": 1, "cover": 1, "resid_med": -1, "boot_spread": -1}
CLIP = 20.0


def read(path):
    return {d["pair"]: d for d in (json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip())}


def corner_diff(A, B):
    if A is None or B is None:
        return math.inf
    return float(np.linalg.norm(CORNERS @ (np.asarray(A) - np.asarray(B)).T, axis=1).mean())


def rank(x):
    """平均秩（并列取平均），inf 排在最后。"""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def load(path, alts):
    """返回 (pairs, {度量: 不可信程度数组（越大越不可信）})。"""
    rows = read(path)
    pairs = sorted(rows)
    bad = {m: np.array([-s * rows[q][m] for q in pairs], float) for m, s in SIGN.items() if m in rows[pairs[0]]}
    for spec in alts:
        name, p = spec.split("=", 1)
        other = read(p)
        bad[f"diff_{name}"] = np.array([corner_diff(rows[q]["A"], other.get(q, {}).get("A")) for q in pairs])
    return pairs, rows, bad


def top(bad, frac):
    """最可信的 frac 比例的下标（并列时按原顺序，同 finetune.label.load_labels 的稳定排序）。"""
    return np.argsort(bad, kind="stable")[: int(round(len(bad) * frac))]


def val(a):
    from workbench.dataset import Dataset
    from workbench.protocol import pair_error

    ds = Dataset(a.data)
    pairs, rows, bad = load(a.metrics, a.alt)
    keep = [i for i, q in enumerate(pairs) if ds.checkpoints("val", q) is not None]
    err = np.array([pair_error(rows[pairs[i]]["A"], *ds.checkpoints("val", pairs[i])) for i in keep])
    e = np.minimum(err, CLIP)
    fr = np.arange(0, 0.951, 0.05)
    curve = lambda order: np.array([e[order[int(round(len(e) * f)):]].mean() for f in fr])   # order：最先去掉的在前
    oracle = curve(np.argsort(-err, kind="stable"))
    out = [{"metric": "random", "spearman": 0.0, "med_all": float(np.median(err)), "med_top20": float(np.median(err)),
            "med_top50": float(np.median(err)), "le3_top50": float((err <= 3).mean()),
            "ause": float((e.mean() - oracle).mean())},
           {"metric": "oracle", "spearman": 1.0, "med_all": float(np.median(err)),
            "med_top20": float(np.median(err[top(err, 0.2)])), "med_top50": float(np.median(err[top(err, 0.5)])),
            "le3_top50": float((err[top(err, 0.5)] <= 3).mean()), "ause": 0.0}]
    for m, b in bad.items():
        b = b[keep]
        out.append({"metric": m, "spearman": spearman(b, err), "med_all": float(np.median(err)),
                    "med_top20": float(np.median(err[top(b, 0.2)])), "med_top50": float(np.median(err[top(b, 0.5)])),
                    "le3_top50": float((err[top(b, 0.5)] <= 3).mean()),
                    "ause": float((curve(np.argsort(-b, kind="stable")) - oracle).mean())})
    write(a.out, out)
    print(f"n = {len(err)}")
    show(out)


def train(a):
    pairs, rows, bad = load(a.metrics, a.alt)
    names = list(bad)
    corr = [dict(metric=m, **{k: round(spearman(bad[m], bad[k]), 4) for k in names}) for m in names]
    write(a.out + "_corr.csv", corr)
    show(corr)
    ref = {f: set(top(bad["n_inliers"], f)) for f in a.fracs}
    ov = [dict(metric=m, **{f"overlap_top{int(f * 100)}": round(len(ref[f] & set(top(bad[m], f))) / len(ref[f]), 4)
                            for f in a.fracs}) for m in names]
    write(a.out + "_overlap.csv", ov)
    print(f"n = {len(pairs)}")
    show(ov)


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows([{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows])


def show(rows):
    keys = list(rows[0])
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("val", "train"):
        p = sub.add_parser(name)
        p.add_argument("metrics")
        p.add_argument("--alt", action="append", default=[])
        p.add_argument("--out", required=True)
    sub.choices["val"].add_argument("--data", required=True)
    sub.choices["train"].add_argument("--fracs", type=float, nargs="+", default=[0.2, 0.5])
    a = ap.parse_args(argv)
    {"val": val, "train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
