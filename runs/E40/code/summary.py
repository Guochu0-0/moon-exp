"""E40（#242）：汇总。读 metrics.json（workbench eval 写的逐对误差）与 extra/data/sets.jsonl，写 extra/data/summary.json，
打印 Notes 用的表。在工作台环境里跑。

    MOON_DATA=... python runs/E40/code/summary.py

- 每行：AUC@5、AUC@3、SR@3、SR@5、误差中位数（workbench.protocol 的口径）。
- 相对 main 的 AUC@5 差：配对自助法，同一批 652 对有放回重抽 2000 次（种子 0），取 2.5%、97.5% 分位；
  变好 / 变差 = 这对的误差比 main 小 / 大（相差不到 0.01 px 记为不变）。
- 离散程度（fit.py 定义）：分布的分位数，以及它与本行误差、main 误差的 Spearman 相关。
- 核对：main 与 E37 `all` 的正式预测逐对比较；每个单独视角的 AUC@5（看换算回原坐标是否正确）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from workbench import protocol
from workbench.dataset import Dataset
from workbench.evaluate import pair_error

R = Path("runs/E40")
ROWS = ["main", "swap", "equiv", "occl", "all"]
B = 2000
TIE = 0.01


def main():
    ds = Dataset(os.environ["MOON_DATA"])
    pairs = ds.labelled("val")
    met = json.loads((R / "metrics.json").read_text(encoding="utf-8"))["methods"]
    err = {m: np.array([np.inf if e is None else e for e in met[m]["val"]["errors"]]) for m in ROWS}
    sets = {json.loads(l)["pair"]: json.loads(l) for l in (R / "extra/data/sets.jsonl").read_text(encoding="utf-8").splitlines()}
    out = {"rows": {}, "check": {}}

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(pairs), size=(B, len(pairs)))
    g0 = np.clip(1 - err["main"] / 5.0, 0, None)
    for m in ROWS:
        s = protocol.summarize(err[m])
        row = {k: s[k] for k in ("auc@5", "auc@3", "sr@3", "sr@5", "median", "fail_rate")}
        g = np.clip(1 - err[m] / 5.0, 0, None)
        if m != "main":
            d = g - g0
            boot = d[idx].mean(1)
            de = err[m] - err["main"]
            row.update(d_auc5=float(d.mean()), d_auc5_ci=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                       better=int((de < -TIE).sum()), worse=int((de > TIE).sum()), same=int((np.abs(de) <= TIE).sum()),
                       n_failed=int(np.isinf(err[m]).sum()))
            disp = np.array([np.nan if sets[p][m].get("disp") is None else sets[p][m]["disp"] for p in pairs])
            ok = np.isfinite(disp) & np.isfinite(err[m])
            row["disp_q"] = {q: float(np.quantile(disp[ok], q / 100)) for q in (10, 25, 50, 75, 90)}
            row["disp_mean"] = float(disp[ok].mean())
            row["rho_disp_err"] = float(spearmanr(disp[ok], err[m][ok]).statistic)
            ok0 = np.isfinite(disp) & np.isfinite(err["main"])
            row["rho_disp_err_main"] = float(spearmanr(disp[ok0], err["main"][ok0]).statistic)
            row["n_est_q"] = {q: float(np.quantile([sets[p][m]["n_est"] for p in pairs], q / 100)) for q in (0, 50, 100)}
        out["rows"][m] = row

    # 核对 1：main 与 E37 all 的正式预测
    e37 = {}
    for line in Path("runs/E37/preds/all/val.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        e37[d["pair"]] = d["A"]
    diff = [np.abs(np.array(sets[p]["views"]["orig"]) - np.array(e37[p])).max()
            for p in pairs if sets[p]["views"]["orig"] is not None and e37.get(p) is not None]
    out["check"]["main_vs_e37_max_abs"] = float(max(diff)) if diff else None
    out["check"]["main_vs_e37_n"] = len(diff)
    # 核对 2：每个单独视角
    views = list(sets[pairs[0]]["views"])
    single = {}
    for v in views:
        e = np.array([pair_error(ds, "val", p, sets[p]["views"][v]) for p in pairs], np.float64)
        single[v] = {"auc@5": protocol.auc(e, 5.0), "median": float(np.median(e)), "n_failed": int(np.isinf(e).sum())}
    out["check"]["single_view"] = single

    (R / "extra/data/summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("| 行 | AUC@5 | AUC@3 | SR@3 | SR@5 | 中位数 | 失败 | ΔAUC@5 [95%] | 好/差/平 |")
    for m in ROWS:
        r = out["rows"][m]
        d = f"{r['d_auc5']:+.4f} [{r['d_auc5_ci'][0]:+.4f}, {r['d_auc5_ci'][1]:+.4f}]" if m != "main" else "—"
        bw = f"{r['better']}/{r['worse']}/{r['same']}" if m != "main" else "—"
        print(f"| {m} | {r['auc@5']:.4f} | {r['auc@3']:.4f} | {r['sr@3']:.3f} | {r['sr@5']:.3f} | {r['median']:.3f} | "
              f"{r['fail_rate']:.3f} | {d} | {bw} |")
    print("\n| 行 | 离散 p10 | p25 | p50 | p75 | p90 | ρ(离散, 本行误差) | ρ(离散, main 误差) | 估计数 min/med/max |")
    for m in ROWS[1:]:
        r = out["rows"][m]
        q = r["disp_q"]
        print(f"| {m} | {q[10]:.2f} | {q[25]:.2f} | {q[50]:.2f} | {q[75]:.2f} | {q[90]:.2f} | {r['rho_disp_err']:.3f} | "
              f"{r['rho_disp_err_main']:.3f} | {r['n_est_q'][0]:.0f}/{r['n_est_q'][50]:.0f}/{r['n_est_q'][100]:.0f} |")
    print("\ncheck", json.dumps({k: v for k, v in out["check"].items() if k != "single_view"}))
    for v, s in single.items():
        print(f"  {v}: auc5 {s['auc@5']:.4f} med {s['median']:.3f} fail {s['n_failed']}")


if __name__ == "__main__":
    main()
