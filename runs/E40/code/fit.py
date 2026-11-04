"""E40（#242）：把 infer.py 的逐视角仿射按各行合成，写 preds/<行>/val.jsonl 与 extra/data/sets.jsonl。在工作台环境里跑。

    MOON_DATA=... python runs/E40/code/fit.py [--raw <目录> ...] [--methods resample,resample2]

--raw：读哪些目录下的 rows_s*.jsonl、resample_s*.jsonl、orig_matches_s*.npz（默认 runs/E40/raw；同一对图的视角合并）。
--methods：只写这些行的 preds（默认全部）；sets.jsonl 总是写全部行。补跑对照时只写新行，已提交的各行预测不动。

各行（方法名）及参与平均的视角：
    main       模型的答案：orig
    swap       正反向互逆：orig + swap
    equiv      对已知扰动等变：orig + eq0–7
    occl       随机遮挡：orig + oc0–7
    all        三项合用：orig + swap + eq0–7 + oc0–7（orig 只算一次）
    resample   抽样噪声对照（9 个估计）：orig + rs1–8，原输入只换随机种子
    resample2  抽样噪声对照（2 个估计）：orig + rs1

取平均：图块四个角 (0,0)、(512,0)、(512,512)、(0,512) 在各个估计下映射到 SAR 的位置取平均，再由四角最小二乘反解一个仿射
（仿射对矩阵是线性的，结果等于各估计矩阵逐元素平均）。没输出仿射的估计不计入；一对图全部失败记为失败。
离散程度：各估计四角位置到平均四角位置的距离，对估计和四个角取平均（px）。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from workbench.dataset import Dataset
from workbench.records import PredWriter

R = Path("runs/E40")
HW = 512
CORNERS = np.array([[0, 0], [HW, 0], [HW, HW], [0, HW]], np.float64)
SETS = {
    "main": ["orig"],
    "swap": ["orig", "swap"],
    "equiv": ["orig"] + [f"eq{j}" for j in range(8)],
    "occl": ["orig"] + [f"oc{j}" for j in range(8)],
    "all": ["orig", "swap"] + [f"eq{j}" for j in range(8)] + [f"oc{j}" for j in range(8)],
    "resample": ["orig"] + [f"rs{j}" for j in range(1, 9)],
    "resample2": ["orig", "rs1"],
}


def project(A):
    return CORNERS @ np.asarray(A)[:, :2].T + np.asarray(A)[:, 2]


def average(As):
    """返回 (平均仿射, 离散程度 px)。"""
    P = np.stack([project(A) for A in As])            # k × 4 × 2
    mean = P.mean(0)
    X = np.c_[CORNERS, np.ones(4)]
    A = np.linalg.lstsq(X, mean, rcond=None)[0].T      # 2 × 3
    disp = float(np.linalg.norm(P - mean, axis=-1).mean())
    return A, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=[str(R / "raw")])
    ap.add_argument("--methods", default=",".join(SETS))
    a = ap.parse_args()
    write = a.methods.split(",")

    rows, matches = {}, {}
    for d in map(Path, a.raw):
        for f in sorted(d.glob("rows_s*.jsonl")) + sorted(d.glob("resample_s*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.setdefault(r["pair"], {}).update(r["views"])
        if "main" in write:
            for f in sorted(d.glob("orig_matches_s*.npz")):
                with np.load(f) as z:
                    matches.update({k: z[k] for k in z.files})
    ds = Dataset(os.environ["MOON_DATA"])
    pairs = ds.labelled("val")
    need = {n for names in SETS.values() for n in names}
    missing = [p for p in pairs if not need <= set(rows.get(p, {}))]
    assert not missing, f"{len(missing)} 对缺视角，例如 {missing[:3]}"

    out = {}
    writers = {m: PredWriter(R, m, "val") for m in SETS if m in write}
    for pair in pairs:
        v = rows[pair]
        rec = {"pair": pair}
        for m, names in SETS.items():
            As = [v[n]["A"] for n in names if v[n]["A"] is not None]
            extra = {"n_est": len(As), "n_views": len(names)}
            w = writers.get(m)
            if m == "main":
                if w:
                    M = matches.get(pair.replace("/", "__"))
                    kw = dict(n_inliers=v["orig"]["n_inliers"], n_matches=v["orig"].get("n_matches"))
                    if M is not None:
                        kw["matches"] = M
                    w.write(pair, v["orig"]["A"], fail=v["orig"].get("fail"), **kw)
                rec[m] = extra
                continue
            if not As:
                if w:
                    w.write(pair, None, fail="all_failed", **extra)
                rec[m] = {**extra, "disp": None}
                continue
            A, disp = average(As)
            extra["disp"] = round(disp, 4)
            if w:
                w.write(pair, A, **extra)
            rec[m] = extra
        rec["views"] = {n: d["A"] for n, d in v.items()}       # 逐视角仿射（已换回原坐标），供 summary 算单个视角的误差
        out[pair] = rec
    for w in writers.values():
        w.close()
    (R / "extra/data").mkdir(parents=True, exist_ok=True)
    with open(R / "extra/data/sets.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for pair in pairs:
            f.write(json.dumps(out[pair]) + "\n")
    print({m: sum(out[p][m]["n_est"] == 0 for p in pairs) for m in SETS}, "pairs with no estimate; wrote preds for", list(writers))


if __name__ == "__main__":
    main()
