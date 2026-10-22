"""几何模型的相图（#195 第 4 项，不用数据）：以 s 为长度单位，T 在原点、R 在 (b/s, 0)；分布中心 M 在以 R 为圆心、
半径 |M − R|/s 的圆上，角度 φ（R → T 方向记 0）均匀取 360 个。对每组 (b/s, |M − R|/s, δ/s) 统计这一圈上
「对的比例」（ΔE[e] < 0 的角度占比）与净比例（−ΣΔE / Σ|ΔE|），以及这一圈上 |ΔE| 的均值（粗级锚点面积取 s²）。

    python runs/E21/code/phase.py

产物：runs/E21/extra/phase.png、extra/data/phase.npz。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))
import geom  # noqa: E402

FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 同 E14
if Path(FONT).exists():
    matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Microsoft YaHei"]   # 中文按字回退到 SimHei
plt.rcParams["axes.unicode_minus"] = False

BS = np.logspace(-1, 1.5, 61)          # b/s
MR = np.logspace(-1, 1.5, 61)          # |M − R|/s
DS = (0.25, 0.5, 1.0, 2.0, 4.0)        # δ/s
PHI = np.linspace(0, 2 * np.pi, 360, endpoint=False)


def sweep(kind):
    """→ right、net、mag，各 (len(DS), len(MR), len(BS))。"""
    b = torch.tensor(BS, dtype=torch.float64)[None, :, None]
    m = torch.tensor(MR, dtype=torch.float64)[:, None, None]
    f = torch.tensor(PHI, dtype=torch.float64)[None, None, :]
    R = torch.stack((b.expand(len(MR), -1, len(PHI)), torch.zeros(len(MR), len(BS), len(PHI), dtype=torch.float64)), -1)
    # φ = 0 指向 T（即 −x 方向）
    M = R + torch.stack((-m * torch.cos(f), m * torch.sin(f)), -1).expand_as(R)
    T = torch.zeros_like(R)
    out = []
    for d in DS:
        if kind == "coarse":
            dE = geom.coarse_a(M, T, R, torch.ones(M.shape[:-1], dtype=torch.float64), d, area=1.0)
        else:
            dE = geom.fine_a(M, T, R, 1.0, d)
        right = (dE < 0).double().mean(-1)
        net = -dE.sum(-1) / dE.abs().sum(-1)
        out.append((right.numpy(), net.numpy(), dE.abs().mean(-1).numpy()))
    return [np.stack(x) for x in zip(*out)]


def main():
    res = {k: sweep(k) for k in ("coarse", "fine")}
    out = RUN / "extra"
    (out / "data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data" / "phase.npz", bs=BS, mr=MR, ds=np.array(DS),
                        **{f"{k}_{n}": v for k, r in res.items() for n, v in zip(("right", "net", "mag"), r)})
    rows = [("coarse", 0, "粗级（逐锚点 logit）\n对的比例"), ("coarse", 1, "粗级\n净比例"),
            ("fine", 0, "细化（更新均值）\n对的比例"), ("fine", 1, "细化\n净比例")]
    fig, axes = plt.subplots(len(rows), len(DS), figsize=(3.0 * len(DS), 2.8 * len(rows)), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, (k, j, lab) in enumerate(rows):
        for c, d in enumerate(DS):
            ax = axes[i, c]
            z = res[k][j][c]
            im = ax.pcolormesh(BS, MR, z, cmap="RdBu", vmin=0 if j == 0 else -1, vmax=1, shading="nearest")
            ax.contour(BS, MR, z, levels=[0.5 if j == 0 else 0.0], colors="k", linewidths=0.8)
            ax.plot(BS, BS, ls=":", c="k", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"δ/s = {d:g}")
            if c == 0:
                ax.set_ylabel(lab + "\n|M - R| / s")
            if i == len(rows) - 1:
                ax.set_xlabel("b / s")
        fig.colorbar(im, ax=axes[i, :], shrink=0.9)
    fig.savefig(out / "phase.png", dpi=130)
    print("saved", out / "phase.png")


if __name__ == "__main__":
    main()
