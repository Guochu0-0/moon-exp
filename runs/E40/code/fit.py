"""E40（#242）：把 infer.py 的逐视角仿射按票里的五行合成，写 preds/<行>/val.jsonl 与 extra/data/sets.jsonl。在工作台环境里跑。

    MOON_DATA=... python runs/E40/code/fit.py

五行（方法名）及参与平均的视角：
    main   模型的答案：orig
    swap   正反向互逆：orig + swap
    equiv  对已知扰动等变：orig + eq0–7
    occl   随机遮挡：orig + oc0–7
    all    三项合用：全部 18 个（orig 只算一次）

取平均：图块四个角 (0,0)、(512,0)、(512,512)、(0,512) 在各个估计下映射到 SAR 的位置取平均，再由四角最小二乘反解一个仿射
（仿射对矩阵是线性的，结果等于各估计矩阵逐元素平均）。没输出仿射的估计不计入；一对图全部失败记为失败。
离散程度：各估计四角位置到平均四角位置的距离，对估计和四个角取平均（px）。
"""
from __future__ import annotations

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
    rows = {}
    for f in sorted((R / "raw").glob("rows_s*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows[d["pair"]] = d["views"]
    matches = {}
    for f in sorted((R / "raw").glob("orig_matches_s*.npz")):
        with np.load(f) as z:
            matches.update({k: z[k] for k in z.files})
    ds = Dataset(os.environ["MOON_DATA"])
    pairs = ds.labelled("val")
    missing = [p for p in pairs if p not in rows]
    assert not missing, f"{len(missing)} 对没有推理结果，例如 {missing[:3]}"

    out = {}
    writers = {m: PredWriter(R, m, "val") for m in SETS}
    for pair in pairs:
        v = rows[pair]
        rec = {"pair": pair}
        for m, names in SETS.items():
            As = [v[n]["A"] for n in names if v[n]["A"] is not None]
            extra = {"n_est": len(As), "n_views": len(names)}
            if m == "main":
                M = matches.get(pair.replace("/", "__"))
                kw = dict(n_inliers=v["orig"]["n_inliers"], n_matches=v["orig"].get("n_matches"))
                if M is not None:
                    kw["matches"] = M
                writers[m].write(pair, v["orig"]["A"], fail=v["orig"].get("fail"), **kw)
                rec[m] = extra
                continue
            if not As:
                writers[m].write(pair, None, fail="all_failed", **extra)
                rec[m] = {**extra, "disp": None}
                continue
            A, disp = average(As)
            extra["disp"] = round(disp, 4)
            writers[m].write(pair, A, **extra)
            rec[m] = extra
        rec["views"] = {n: d["A"] for n, d in v.items()}       # 逐视角仿射（已换回原坐标），供 summary 算单个视角的误差
        out[pair] = rec
    for w in writers.values():
        w.close()
    (R / "extra/data").mkdir(parents=True, exist_ok=True)
    with open(R / "extra/data/sets.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for pair in pairs:
            f.write(json.dumps(out[pair]) + "\n")
    print({m: sum(out[p][m]["n_est"] == 0 for p in pairs) for m in SETS}, "pairs with no estimate")


if __name__ == "__main__":
    main()
