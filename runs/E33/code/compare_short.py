"""E33 只用（#226）：同种子 200 步短跑的日志曲线对比（提速前 base_a、提速后 all_a、提速前再跑一遍 base_b）。
每个量按 50 步一段取平均，写表 extra/short_runs.json；画 20 步滑动平均的曲线 extra/short_runs.png。

    /opt/envs/wb/bin/python runs/E33/code/compare_short.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[1]
RUNS = {"base_a": "提速前", "base_b": "提速前（再跑一遍）", "all_a": "提速后"}
KEYS = {"loss": "损失", "rew": "粗匹配平均得分", "d_ref": "离参照的距离 (px)", "in_frame": "画面内格子比例",
        "rew_16": "细化尺度 16 平均得分", "rew_4": "细化尺度 4 平均得分", "ref_gap": "参照与当前模型的仿射之差 (px)",
        "gnorm": "梯度范数"}


def load(name):
    return [json.loads(x) for x in open(R / "ckpt" / "short" / name / "log.jsonl", encoding="utf-8")]


def main():
    logs = {k: load(k) for k in RUNS}
    table = {}
    for k in KEYS:
        table[k] = {}
        for name, log in logs.items():
            v = np.array([r.get(k, np.nan) for r in log], float)
            table[k][name] = [round(float(np.nanmean(v[i:i + 50])), 4) for i in range(0, len(v), 50)]
    secs = {name: round(float(np.median([r["sec"] for r in log[10:]])), 3) for name, log in logs.items()}
    out = R / "extra"
    out.mkdir(exist_ok=True)
    (out / "short_runs.json").write_text(json.dumps({"segments_of_50_steps": table, "sec_median": secs},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (k, title) in zip(axes.flat, KEYS.items()):
        for name, log in logs.items():
            v = np.array([r.get(k, np.nan) for r in log], float)
            m = np.convolve(np.nan_to_num(v), np.ones(20) / 20, "valid")
            ax.plot(np.arange(20, len(v) + 1), m, label=RUNS[name], lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("步")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / "short_runs.png", dpi=110)
    print(json.dumps({"sec": secs, **{k: table[k] for k in ("d_ref", "in_frame", "rew")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
