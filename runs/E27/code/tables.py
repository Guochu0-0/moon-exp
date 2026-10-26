"""Notes 里的表：失效前后观察集逐 10 步的数值、可疑步、可复现检查。先跑 summary.py。

    /opt/envs/wb/bin/python runs/E27/code/tables.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

from load import R, log, train_trace

D = R / "extra" / "data"
SUM = json.loads((D / "summary.json").read_text(encoding="utf-8"))
MAIN = [m for m in SUM if "_r" not in m.split("_s")[-1]]


def obs_csv(m):
    with open(D / f"{m}_obs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def around():
    """失效前后：观察集原图自估仿射离标注、粗匹配离自估仿射、粗匹配离标注 > 16 px 与离自估仿射 > 16 px 的格子比例。"""
    out = {}
    for m in MAIN:
        if not SUM[m]["failed"]:
            continue
        o = obs_csv(m)
        bad = ~(o["err_orig"] <= 20)
        i = len(bad) - 1
        while i > 0 and bad[i - 1]:
            i -= 1                                      # 从这一步起一直超过 20 px
        first = int(o["step"][np.flatnonzero(bad)[0]])
        rows = []
        for j in range(max(0, i - 3), min(len(o["step"]), i + 3)):
            rows.append({k: float(o[k][j]) for k in ("step", "err_orig", "d_self", "d_gt", "wrong_gt", "off_self", "cert1")})
        out[m] = {"first_excursion": first, "break": int(o["step"][i]), "rows": rows}
    return out


def before():
    """持续失效前 30 步（观察集）各量，与同一步健康运行的中位数并列。"""
    heal = [obs_csv(m) for m in MAIN if not SUM[m]["failed"]]
    keys = ("err_orig", "d_gt", "wrong_gt", "d_self", "off_self")
    out = {}
    for m, v in around().items():
        o = obs_csv(m)
        s = v["break"] - 30
        j = int(np.flatnonzero(o["step"] == s)[0])
        out[m] = {"break": v["break"], "step": s,
                  "run": {k: float(o[k][j]) for k in keys},
                  "healthy": {k: float(np.median([h[k][j] for h in heal])) for k in keys}}
    return out


def edge(x, y):
    return min(x, y, 512 - x, 512 - y) <= 12


def q0():
    """坍缩落点：位置、是否在边上、训练批次上 q0 处与全图的补零比例、训练批次上最高分位置落在 q0 的格子第一次过一成与过半的步；
    以及持续失效前 100 步与之后 100 步里有信号的格子比例、最强一成格子占的信号。"""
    out = {}
    for m in MAIN:
        if not SUM[m]["failed"]:
            continue
        b = around()[m]["break"]
        x, y = SUM[m]["q0_xy"]
        z = np.load(D / f"{m}_train.npz")
        tr = train_trace(m, ["step", "gnorm"])
        g = np.nan_to_num(tr["gnorm"].astype(float).reshape(len(tr["step"]), -1))
        sig = {}
        for name, (a, c) in (("before", (b - 100, b)), ("after", (b, b + 100))):
            s = (tr["step"] > a) & (tr["step"] <= c)
            gs = g[s]
            sig[name] = {"signal": float((gs > 0).mean()),
                         "top10": float(np.mean(np.sort(gs, 1)[:, -160:].sum(1) / (gs.sum(1) + 1e-12)))}
        out[m] = {"xy": [x, y], "edge": edge(x, y), "pad_q0": float(np.mean(z["pad_q0"])), "pad_all": float(np.mean(z["pad"])),
                  "tenth": SUM[m]["q0_first_train"]["tenth"], "half": SUM[m]["q0_first_train"]["half"],
                  "collapsed": SUM[m]["collapsed"], "break": b, "p99": SUM[m]["events"]["p99"], "signal": sig}
    return out


def repro():
    """续训与原运行：逐步比较 loss、离自估仿射的距离，记第一个不同的步与之后的最大差。"""
    out = {}
    for m, src in (("clip_s0_r300", "clip_s0"), ("clip_s0_r300v", "clip_s0"), ("f1_s1_r400v", "f1_s1")):
        try:
            B = log(m)
        except FileNotFoundError:
            continue
        A = {r["step"]: r for r in log(src)}
        diff = [b["step"] for b in B if b["loss"] != A[b["step"]]["loss"] or b.get("d_ref") != A[b["step"]].get("d_ref")]
        out[m] = {"source": src, "steps": [B[0]["step"], B[-1]["step"]],
                  "same_pairs": all(b["pairs"] == A[b["step"]]["pairs"] for b in B),
                  "first_diff": diff[0] if diff else None,
                  "d_ref_end": [A[B[-1]["step"]].get("d_ref"), B[-1].get("d_ref")],
                  "max_abs_d_ref": max(abs(b["d_ref"] - A[b["step"]]["d_ref"]) for b in B)}
    return out


def main():
    res = {"around": around(), "before": before(), "q0": q0(), "repro": repro()}
    (D / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for m, v in res["around"].items():
        print(m, "首次越界", v["first_excursion"], "持续失效起点", v["break"])
        for r in v["rows"]:
            print("  " + "  ".join(f"{k}={v:.3g}" for k, v in r.items()))
    for k in ("before", "q0", "repro"):
        print(json.dumps(res[k], ensure_ascii=False))


if __name__ == "__main__":
    main()
