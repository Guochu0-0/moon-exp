"""E27 每个方法的逐步汇总：训练日志的三个事件步（同 E26：50 步滑动中位数）、观察集各量随步数的变化、坍缩落点 q0。
写 extra/data/summary.json 与 extra/data/<方法>_obs.csv、<方法>_train.npz（画图脚本读）。

    /opt/envs/wb/bin/python runs/E27/code/summary.py [方法 …]
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from load import R, anchor_xy, first, log, obs, obs_steps, rolling_median, series, train_trace

D = R / "extra" / "data"
NEAR = 1.5        # 与 q0 相距不超过 1.5 个锚点（12 px）算落在 q0


def sv(A):
    if not np.isfinite(A).all():
        return np.nan, np.nan
    s = np.linalg.svd(A[:, :2], compute_uv=False)
    return float(s[0]), float(s[1])


def near(mode, q0):
    x, y = anchor_xy(mode)
    x0, y0 = anchor_xy(q0)
    return np.hypot(x - x0, y - y0) <= NEAR * 8


def events(rows):
    out = {}
    for key, name, cond in (("d_ref", "dist100", lambda v: v > 100), ("rew_sd", "sd1e-3", lambda v: v < 1e-3),
                            ("pmode", "p99", lambda v: v > 0.99)):
        st, v = series(rows, key)
        out[name] = first(st, rolling_median(v), cond)
    return out


def method(m):
    rows = log(m)
    ev = events(rows)
    steps = obs_steps(m)
    last = obs(m, steps[-1])
    vals, cnt = np.unique(last["mode"], return_counts=True)
    q0 = int(vals[cnt.argmax()])
    collapsed = bool(np.mean(near(last["mode"], q0)) > 0.5)
    recs = []
    for st in steps:
        o = obs(m, st)
        v = o["version"]
        r = {"step": st}
        for name, sel in (("orig", v == 0), ("pert", v == 1)):
            r[f"err_{name}"] = float(np.median(o["err"][sel]))
            r[f"auc5_{name}"] = float(np.mean(np.clip(1 - o["err"][sel] / 5, 0, None)))
        r["d_self"] = float(np.nanmedian(o["d_self"].astype(float)))
        r["d_gt"] = float(np.nanmedian(o["d_gt"].astype(float)))
        r["wrong_gt"] = float(np.nanmean(o["d_gt"].astype(float) > 16))
        r["off_self"] = float(np.nanmean(o["d_self"].astype(float) > 16))
        r["at_q0"] = float(np.mean(near(o["mode"], q0)))
        r["n_modes"] = float(np.median([len(np.unique(x)) for x in o["mode"]]))
        r["pmode1"] = float(np.mean(o["pmode1"].astype(float)))
        r["cert1"] = float(np.mean(o["cert1"].astype(float)))
        r["n_inl"] = float(np.median(o["n_inl"]))
        s = np.array([sv(A) for A in o["A"]])
        r["sv_min"] = float(np.nanmedian(s[:, 1]))
        recs.append(r)
    D.mkdir(parents=True, exist_ok=True)
    with open(D / f"{m}_obs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    tr = train_trace(m, ["step", "mode", "pad", "d", "rew", "rew_sd", "gnorm", "same", "n_inl", "A", "T", "mode_post"])
    at = near(tr["mode"], q0)
    qx, qy = int(q0 % 64), int(q0 // 64)
    pad_q0 = tr["pad"][:, qy, qx] / 255.0
    np.savez_compressed(D / f"{m}_train.npz", step=tr["step"], at_q0=at.mean(1), pad_q0=pad_q0,
                        pad=tr["pad"].reshape(len(tr["step"]), -1).mean(1) / 255.0,
                        d=np.nanmedian(tr["d"].reshape(len(tr["step"]), -1).astype(float), 1),
                        n_inl=tr["n_inl"], moved=(tr["mode"] != tr["mode_post"]).mean(1))
    first_q0 = {k: next((int(s) for s, a in zip(tr["step"], at.sum(1)) if a >= n), None) for k, n in
                (("any", 1), ("tenth", 160), ("half", 800))}
    return {"events": ev, "collapsed": collapsed, "q0": q0, "q0_xy": [float(x) for x in anchor_xy(q0)],
            "q0_first_train": first_q0, "n_obs": len(steps), "last": recs[-1]}


def main(argv):
    ms = argv or sorted(p.name for p in (R / "ckpt").iterdir() if (p / "trace").is_dir())
    f = D / "summary.json"
    out = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    for m in ms:
        out[m] = method(m)
        print(m, json.dumps(out[m], ensure_ascii=False))
    D.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
