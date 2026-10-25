"""探查「怎样聚到一点」：训练批次上每 50 步一段的有信号格子、移动方向与 q0（文字输出）。

    /opt/envs/wb/bin/python runs/E27/code/explore2.py <方法> [q0]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from load import R, anchor_xy, train_trace


def main(m, q0=None):
    if q0 is None:
        q0 = json.loads((R / "extra" / "data" / "summary.json").read_text(encoding="utf-8"))[m]["q0"]
    x0, y0 = anchor_xy(q0)
    tr = train_trace(m, ["step", "mode", "mode_post", "pos", "pos_post", "gnorm", "gup", "rew", "d", "pad", "same", "pmode"])
    st = tr["step"]
    mx, my = anchor_xy(tr["mode"])
    dist = np.hypot(mx - x0, my - y0)                       # 各格最高分位置离 q0（px）
    pos = tr["pos"].astype(float).reshape(len(st), 2, -1)
    post = tr["pos_post"].astype(float).reshape(len(st), 2, -1)
    mv = post - pos
    to = np.stack([x0 - pos[:, 0], y0 - pos[:, 1]], 1)
    to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-6
    toward = (mv * to).sum(1)                                # 移动在指向 q0 方向上的分量（px）
    g = tr["gnorm"].astype(float).reshape(len(st), -1)
    rew = tr["rew"].astype(float).reshape(len(st), -1)
    near = dist <= 12
    pad_q0 = tr["pad"][:, int(y0 // 8), int(x0 // 8)] / 255
    print(f"q0 = {q0} ({x0:.0f}, {y0:.0f})")
    print("段      at_q0  d_med  rew_med floor  g_med   g_top10share  g_near_q0/g_far  moved  toward_med toward_moved  pad_q0 pad_all pmode")
    for a in range(0, 1200, 50):
        s = (st > a) & (st <= a + 50)
        gs = g[s]
        top = np.sort(gs, 1)[:, -160:].sum(1) / (gs.sum(1) + 1e-12)
        ratio = [np.nanmean(gi[n]) / (np.nanmean(gi[~n]) + 1e-12) if n.any() and (~n).any() else np.nan
                 for gi, n in zip(gs, near[s])]
        moved = np.hypot(*mv[s].transpose(1, 0, 2)) > 1
        tw = toward[s]
        print(f"{a:4d}-{a+50:4d} {near[s].mean():.3f}  {np.nanmedian(tr['d'][s].astype(float)):6.1f}  "
              f"{np.median(rew[s]):6.3f} {np.mean(rew[s] < -0.099):.2f}  {np.median(gs):.2e} {np.median(top):.2f}  "
              f"{np.nanmedian(ratio):7.2f}  {moved.mean():.2f}  {np.median(tw):6.2f}  "
              f"{np.median(tw[moved]) if moved.any() else np.nan:6.2f}  {pad_q0[s].mean():.2f}  "
              f"{tr['pad'][s].mean()/255:.2f}  {tr['pmode'][s].astype(float).mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
