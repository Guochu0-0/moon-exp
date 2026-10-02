"""逐 ckpt 的塌缩信号汇总（「【类 RIPE】粗级闭式期望从 zero-shot 单独训练：是否塌缩」#49）。工作台环境。

    python scripts/finetune/collapse.py <finetune>/<name> [...] [--data $MOON_DATA] [--json out.json]

读 <name>/sweep/S（Val 预测与 metrics.json）与 <name>/neg/summary.json（finetune.negpairs），每个 step 一行：
- auc5：Val AUC@5；med：Val 误差中位数（px）；
- 匹配数与内点比例（Val 各对中位数）；
- dI：估出的仿射相对 [I|0] 的角点平均位移（Val 各对中位数，px）；
- idI：逐对误差与恒等变换误差相差 < 1 px 的对占比（有标注的 Val 对；越接近 1 越像恒等）；
- neg_inl / neg_ident：负样本对上的 RANSAC 内点数、恒等匹配数（中位数）；diag：正样本对的粗级 conf[i,i] 均值。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.dataset import Dataset  # noqa: E402
from workbench.evaluate import evaluate_identity  # noqa: E402


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def dist_identity(A):
    c = np.array([[0, 0], [512, 0], [0, 512], [512, 512]], float)
    A = np.asarray(A, float)
    return float(np.linalg.norm(c @ A[:, :2].T + A[:, 2] - c, axis=1).mean())


def table(run, e_id):
    run = Path(run)
    sw = run / "S" if (run / "S").is_dir() else run / "sweep/S"   # 新布局 runs/<id>/sweep/<m>/S；旧布局 <name>/sweep/S
    metrics = json.loads((sw / "metrics.json").read_text(encoding="utf-8"))["methods"]
    neg = json.loads((run / "neg/summary.json").read_text()) if (run / "neg/summary.json").exists() else {}
    rows = []
    for m in sorted(metrics, key=lambda k: int(k[4:])):
        v = metrics[m].get("val")
        if not v:
            continue
        preds = load_jsonl(sw / "preds" / m / "val.jsonl")
        nm = np.array([p.get("n_matches", 0) for p in preds], float)
        ni = np.array([p.get("n_inliers", 0) or 0 for p in preds], float)
        dI = [dist_identity(p["A"]) for p in preds if p.get("A") is not None]
        e = np.array([np.inf if x is None else x for x in v["errors"]], float)
        both = np.isfinite(e) & np.isfinite(e_id)
        ng = neg.get(m, {})
        g = lambda kind, k: (ng.get(kind) or {}).get(k, {}) and ng[kind][k]["median"]
        rows.append({"step": int(m[4:]), "auc5": v["summary"]["auc@5"], "med": v["summary"].get("median"),
                     "n_match": float(np.median(nm)), "inl_ratio": float(np.median(ni / np.maximum(nm, 1))),
                     "dI": float(np.median(dI)) if dI else None,
                     "idI": float(np.mean(np.abs(e[both] - e_id[both]) < 1.0)),
                     "neg_inl": g("neg", "n_inl"), "neg_ident": g("neg", "n_ident"), "diag": g("pos", "diag")})
    return rows


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    e_id = np.array([np.inf if x is None else x for x in evaluate_identity(Dataset(args.data))["val"]["errors"]], float)
    out = {}
    for r in args.runs:
        rows = table(r, e_id)
        out[Path(r).name] = rows
        print(f"== {Path(r).name}")
        print("step   auc5   med    n_match inl%   dI     idI    neg_inl neg_ident diag")
        for x in rows:
            print(f"{x['step']:<6d} {fmt(x['auc5'])}  {fmt(x['med'], 1):6s} {fmt(x['n_match'], 0):7s} "
                  f"{fmt(x['inl_ratio'], 2):6s} {fmt(x['dI'], 1):6s} {fmt(x['idI'], 2):6s} "
                  f"{fmt(x['neg_inl'], 0):7s} {fmt(x['neg_ident'], 0):9s} {fmt(x['diag'], 4)}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
