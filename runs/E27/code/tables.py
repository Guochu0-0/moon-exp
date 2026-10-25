"""Notes 里的表：失效前后观察集逐 10 步的数值、可疑步、可复现检查。先跑 summary.py。

    /opt/envs/wb/bin/python runs/E27/code/tables.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

from load import R, log

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
    res = {"around": around(), "repro": repro()}
    (D / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for m, v in res["around"].items():
        print(m, "首次越界", v["first_excursion"], "持续失效起点", v["break"])
        for r in v["rows"]:
            print("  " + "  ".join(f"{k}={v:.3g}" for k, v in r.items()))
    print(json.dumps(res["repro"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
