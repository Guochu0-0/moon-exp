"""探查一段步数里训练批次与观察集的逐步细节（文字输出，供写分析脚本前看数据）。

    /opt/envs/wb/bin/python runs/E27/code/explore.py <方法> <起步> <止步>
"""
from __future__ import annotations

import sys

import numpy as np

from load import cell_centers, log, obs, obs_steps, train_trace


def mapped(A, x, y):
    return A[0, 0] * x + A[0, 1] * y + A[0, 2], A[1, 0] * x + A[1, 1] * y + A[1, 2]


def main(m, a, b):
    rows = {r["step"]: r for r in log(m)}
    tr = train_trace(m)
    cx, cy = cell_centers()
    print("step pair            shift  rot   pad   d_med  d90   n_inl  moved  dpos_med  gnorm  lr       upd_coarse upd_lora rew   same")
    for i, st in enumerate(tr["step"]):
        if not a <= st <= b:
            continue
        r = rows[int(st)]
        T = tr["T"][i]
        rot = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        d = tr["d"][i].astype(float)
        mv = (tr["mode"][i] != tr["mode_post"][i]).mean()
        dp = np.hypot(*(tr["pos_post"][i].astype(float) - tr["pos"][i].astype(float))).reshape(-1)
        print(f"{st:4d} {r['pairs'][0]:16s} {np.hypot(T[0,2]-256*(1-T[0,0])+256*T[0,1], 0):5.0f} {rot:5.1f} "
              f"{tr['pad'][i].mean()/255:.2f}  {np.nanmedian(d):6.1f} {np.nanpercentile(d,90):6.1f} {tr['n_inl'][i]:5d}  "
              f"{mv:.2f}  {np.median(dp):6.2f}  {r.get('gnorm', np.nan):6.3f} {r.get('lr', np.nan):.2e} "
              f"{r.get('upd_coarse', np.nan):.2e} {r.get('upd_lora', np.nan):.2e} {r.get('rew', np.nan):.3f} "
              f"{np.mean(tr['same'][i].astype(float)):.2f}")
    print("\n观察集：每个样本相对标注的粗匹配偏移（中位向量 dx, dy 与离散程度）、自估仿射误差")
    for st in obs_steps(m):
        if not a <= st <= b:
            continue
        o = obs(m, st)
        out = []
        for j in range(len(o["err"])):
            gx, gy = mapped(o["A_gt"][j], cx, cy)
            dx = o["pos"][j, 0].astype(float) - gx
            dy = o["pos"][j, 1].astype(float) - gy
            out.append((np.median(dx), np.median(dy), np.median(np.hypot(dx - np.median(dx), dy - np.median(dy))),
                        o["err"][j]))
        out = np.array(out)
        print(f"{st:4d} shift|med| {np.median(np.hypot(out[:,0], out[:,1])):6.1f}  spread {np.median(out[:,2]):6.1f}  "
              f"err {np.median(out[:,3]):6.1f}  " + " ".join(f"({x:.0f},{y:.0f})" for x, y, _, _ in out[::4]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
