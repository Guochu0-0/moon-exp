"""实验记录：读（给工作台）与写（给跑实验的代码）。

一个实验一个目录 runs/<id>/，格式见 workbench/README.md：

    runs/<id>/
      exp.toml                      元数据：父实验、commit、假设、改动、状态、结论……
      preds/<method>/<split>.jsonl  每个 pair 一行：估计仿射（光学 → SAR，2×3）或失败原因
      preds/<method>/<split>_matches.npz   可选：每个 pair 的点对 N×4 (x_opt, y_opt, x_sar, y_sar)
      extra/                        灵活区产物，工作台原样展示
      metrics.json                  派生物：`python -m workbench eval` 写出，勿手改
"""
from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPLITS = ("val", "test")
STATUSES = ("baseline", "running", "kept", "dropped")
MAIN_METHOD = "main"


@dataclass
class Pred:
    pair: str
    A: list | None          # 2×3，光学像素 → SAR 像素；None = 失败
    fail: str | None = None  # 失败原因：no_output / few_inliers / error: ...
    extra: dict = field(default_factory=dict)  # n_matches、n_inliers、sec 等附加字段


@dataclass
class Run:
    id: str
    dir: Path
    meta: dict
    methods: list[str]

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent") or None

    def preds(self, method: str, split: str) -> dict[str, Pred]:
        path = self.dir / "preds" / method / f"{split}.jsonl"
        out = {}
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            pair, A, fail = d.pop("pair"), d.pop("A", None), d.pop("fail", None)
            out[pair] = Pred(pair, A, fail, d)
        return out

    def matches(self, method: str, split: str, pair: str) -> np.ndarray | None:
        path = self.dir / "preds" / method / f"{split}_matches.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            key = _npz_key(pair)
            return z[key] if key in z.files else None

    def extras(self) -> list[Path]:
        d = self.dir / "extra"
        return sorted(p for p in d.rglob("*") if p.is_file()) if d.exists() else []


def _npz_key(pair: str) -> str:
    return pair.replace("/", "__")


def load_runs(root: Path) -> dict[str, Run]:
    runs = {}
    if not Path(root).is_dir():
        return runs
    for d in sorted(p for p in Path(root).iterdir() if (p / "exp.toml").exists()):
        meta = tomllib.loads((d / "exp.toml").read_text(encoding="utf-8"))
        rid = meta.get("id", d.name)
        if rid != d.name:
            raise ValueError(f"{d}: exp.toml 里 id={rid!r} 与目录名不一致")
        pd = d / "preds"
        methods = sorted(p.name for p in pd.iterdir() if p.is_dir()) if pd.exists() else []
        runs[rid] = Run(rid, d, meta, methods)
    return runs


def check_runs(runs: dict[str, Run]) -> list[str]:
    """记录层面的一致性检查，返回问题列表。"""
    problems = []
    for r in runs.values():
        m = r.meta
        for k in ("title", "status"):
            if k not in m:
                problems.append(f"{r.id}: exp.toml 缺少 {k}")
        if m.get("status") not in STATUSES:
            problems.append(f"{r.id}: status 应为 {STATUSES} 之一，现为 {m.get('status')!r}")
        if r.parent and r.parent not in runs:
            problems.append(f"{r.id}: 父实验 {r.parent} 不存在")
        init = m.get("init")
        if init:
            iid, _, im = init.partition("/")
            if iid not in runs or (im and im not in runs[iid].methods):
                problems.append(f"{r.id}: init={init!r} 找不到")
        if not r.methods and m.get("status") != "running":
            problems.append(f"{r.id}: 没有 preds/")
    # 环检测
    for r in runs.values():
        seen, cur = set(), r.id
        while cur:
            if cur in seen:
                problems.append(f"{r.id}: 父链有环")
                break
            seen.add(cur)
            cur = runs[cur].parent if cur in runs else None
    return problems


class PredWriter:
    """给跑实验的代码用：逐 pair 写一个方法在一个 split 上的结果。

        with PredWriter("runs/B0", "roma", "test") as w:
            for pair in pairs:
                w.write(pair, A)                      # A: 2×3 或 None
                w.write(pair, None, fail="few_inliers", n_matches=2)
                w.write(pair, A, matches=M)           # 可选 N×4 点对
    """

    def __init__(self, run_dir, method: str, split: str):
        assert split in SPLITS, split
        self.dir = Path(run_dir) / "preds" / method
        self.dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self._f = open(self.dir / f"{split}.jsonl", "w", encoding="utf-8")
        self._matches: dict[str, np.ndarray] = {}

    def write(self, pair: str, A, fail: str | None = None, matches=None, **extra):
        if A is not None:
            A = np.asarray(A, dtype=np.float64).reshape(2, 3)
            if not np.isfinite(A).all():
                A, fail = None, fail or "non_finite"
            else:
                A = [[round(float(v), 6) for v in row] for row in A]
        if A is None and fail is None:
            fail = "no_output"
        rec = {"pair": pair, "A": A}
        if fail:
            rec["fail"] = fail
        rec.update({k: _jsonable(v) for k, v in extra.items()})
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if matches is not None:
            self._matches[_npz_key(pair)] = np.asarray(matches, dtype=np.float32).reshape(-1, 4)

    def close(self):
        self._f.close()
        if self._matches:
            np.savez_compressed(self.dir / f"{self.split}_matches.npz", **self._matches)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
