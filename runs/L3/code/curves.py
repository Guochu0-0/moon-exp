"""L3 的训练曲线：尺度课程与同一代码重跑的基准配方（L1 的 main__ref）对比，供 Notes 的图引用。

    /opt/envs/loftr/bin/python runs/L3/code/curves.py --src /remote-home/xufang/YGC/moon-exp

--src 是存着训练日志（runs/<id>/ckpt/<m>/log.jsonl，不进 git）的 checkout，默认是本仓库。
Val 曲线读 runs/<id>/sweep/<m>/S/metrics.json（进 git）。
产物：runs/L3/extra/curves.png；画图用的数据 runs/L3/extra/data/curves.json（训练量按 200 步滑动平均，每 50 步取一点）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "extra"
FONT = "/remote-home/xufang/YGC/fonts/simhei.ttf"      # 服务器上没有中文字体，放了一份在 gpfs
RUNS = {"ref": ("L1", "main__ref", "基准配方（重跑）", "tab:gray"),
        "curr": ("L3", "main", "尺度课程", "tab:blue")}
WIN, EVERY, RAMP = 200, 50, (2000, 3000)


def val_curve(e, m):
    d = json.loads((ROOT / f"runs/{e}/sweep/{m}/S/metrics.json").read_text(encoding="utf-8"))["methods"]
    return {int(re.sub(r"\D", "", k)): v["val"]["summary"]["auc@5"] for k, v in d.items() if "val" in v}


def train_curves(src, e, m):
    rows = [json.loads(x) for x in (src / f"runs/{e}/ckpt/{m}/log.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    step = np.array([r["step"] for r in rows])
    out = {}
    for k in ("epe1_px", "cls16"):
        y = np.array([np.nan if r.get(k) is None else r[k] for r in rows], float)
        ok = np.isfinite(y)
        c = np.convolve(np.where(ok, y, 0), np.ones(WIN), "valid") / np.convolve(ok, np.ones(WIN), "valid")
        s = step[WIN - 1:]
        keep = s % EVERY == 0
        out[k] = {"step": s[keep].tolist(), "value": np.round(c[keep], 4).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT)
    src = ap.parse_args().src
    data = {"window": WIN, "every": EVERY, "ramp": RAMP, "runs": {}}
    for key, (e, m, name, _) in RUNS.items():
        data["runs"][key] = {"exp": e, "method": m, "name": name,
                             "val_auc5": {str(s): round(a, 4) for s, a in sorted(val_curve(e, m).items())},
                             **train_curves(src, e, m)}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data/curves.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    import matplotlib
    import matplotlib.font_manager
    matplotlib.use("Agg")
    if Path(FONT).exists():
        matplotlib.font_manager.fontManager.addfont(FONT)
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    panels = [("val_auc5", "(a) Val AUC@5（每 1000 步一个存档）"),
              ("epe1_px", "(b) 1 倍分辨率上与伪标签的中位误差（px）"),
              ("cls16", "(c) 粗级锚点分类损失")]
    for ax, (k, title) in zip(axs, panels):
        ax.axvspan(*RAMP, color="tab:orange", alpha=0.12, lw=0, label="细层系数从 0 升到 1")
        for key, (_, _, name, c) in RUNS.items():
            d = data["runs"][key][k]
            if k == "val_auc5":
                ax.plot([int(s) for s in d], list(d.values()), "-o", color=c, ms=4, label=name)
            else:
                ax.plot(d["step"], d["value"], color=c, lw=1.3, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("训练步数")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=150)
    print("ok", OUT / "curves.png")


if __name__ == "__main__":
    main()
