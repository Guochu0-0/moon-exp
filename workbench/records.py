"""实验记录：读（给工作台）与写（给跑实验的代码）。

一个实验一个目录 runs/<id>/，格式（记录格式 v2）见 workbench/README.md：

    runs/<id>/
      exp.toml                              元数据：id、title、parent、init、baseline、date、[methods.<m>]
      notes.md                              可选：实验唯一的自由文本
      preds/<method>/<split>.jsonl          每个 pair 一行：估计仿射（光学 → SAR，2×3）或失败原因
      preds/<method>/<split>.meta.json      写入时的 commit 与 dirty
      preds/<method>/<split>_matches.npz    点对 N×5 (x_opt, y_opt, x_sar, y_sar, conf)，不进 git
      extra/                                附件，Notes 引用时才显示
      metrics.json                          派生物：`python -m workbench eval` 写出，勿手改
"""
from __future__ import annotations

import json
import math
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPLITS = ("val", "test")
MAIN_METHOD = "main"
FIELDS = ("id", "title", "parent", "init", "baseline", "date", "methods")
METHOD_FIELDS = ("name", "caveat")
LEGACY_FIELDS = ("status", "commit", "hypothesis", "change", "verdict", "next")
MATCHES_CAP = 2000   # 每个 pair 的点对最多存多少个点（按 conf 取前若干）
REPO = Path(__file__).resolve().parent.parent


@dataclass
class Pred:
    pair: str
    A: list | None          # 2×3，光学像素 → SAR 像素；None = 失败
    fail: str | None = None  # 失败原因：no_output / few_inliers / error: ...
    extra: dict = field(default_factory=dict)  # n_matches、n_inliers、sec 等附加字段


@dataclass
class Run:
    id: str                 # 目录名；exp.toml 里的 id 应与它一致，由 check_runs 检查
    dir: Path
    meta: dict
    methods: list[str]
    warnings: list[str] = field(default_factory=list)
    runs: dict[str, Run] = field(default_factory=dict, repr=False, compare=False)   # 全部实验，供父链回退

    @property
    def title(self) -> str:
        return self.meta.get("title", "")

    @property
    def parent(self) -> str | None:
        return self.meta.get("parent") or None

    @property
    def init(self) -> str | None:
        return self.meta.get("init") or None

    @property
    def baseline(self) -> bool:
        return self.meta.get("baseline") is True

    @property
    def date(self) -> str | None:
        return self.meta.get("date") or None

    @property
    def lit(self) -> bool:
        """点亮：任一方法有任一 split 的 preds。"""
        return any(self._has(m, s) for m in self.methods for s in SPLITS)

    @property
    def notes(self) -> str:
        path = self.dir / "notes.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _has(self, method: str, split: str) -> bool:
        return (self.dir / "preds" / method / f"{split}.jsonl").exists()

    def method_info(self, method: str) -> dict:
        """显示名与 caveat。变体 `x__v` 没写 [methods] 时沿父链回退到基础方法 `x`，显示名后加「· v」。"""
        own = self.meta.get("methods", {}).get(method)
        if own is not None:
            return {"name": own.get("name", method), "caveat": own.get("caveat")}
        base, sep, variant = method.partition("__")
        if sep:
            for r in self._chain():
                cfg = r.meta.get("methods", {}).get(base)
                if cfg is not None:
                    return {"name": f"{cfg.get('name', base)} · {variant}", "caveat": cfg.get("caveat")}
        return {"name": method, "caveat": None}

    def _chain(self):
        """自己、父实验、祖父实验……遇到环或缺失就停。"""
        seen, cur = set(), self
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            yield cur
            cur = self.runs.get(cur.parent) if cur.parent else None

    def commit(self) -> dict:
        """从各方法的 preds meta 汇总 commit。

        state：single（全部相同，commit 为该值）/ multiple（不同，commit 为 None）/ unknown（都没有 meta）。
        methods：{方法: {split: {"commit", "dirty"} 或 None}}，供展开查看。
        """
        methods = {}
        for m in self.methods:
            for s in SPLITS:
                if self._has(m, s):
                    p = self.dir / "preds" / m / f"{s}.meta.json"
                    methods.setdefault(m, {})[s] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        metas = [v for d in methods.values() for v in d.values()]
        commits = {v.get("commit") if v else None for v in metas}
        if commits <= {None}:
            return {"state": "unknown", "commit": None, "dirty": None, "methods": methods}
        if len(commits) > 1:
            return {"state": "multiple", "commit": None, "dirty": None, "methods": methods}
        return {"state": "single", "commit": commits.pop(), "dirty": any(v.get("dirty") for v in metas),
                "methods": methods}

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
        """点对 N×5 float32；这个 pair 没有点对（出错、没写）时为 None。v1 的 N×4 读出来 conf 为 NaN。"""
        path = self.dir / "preds" / method / f"{split}_matches.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            key = _npz_key(pair)
            if key not in z.files:
                return None
            m = z[key].astype(np.float32)
        if m.shape[1] == 4:
            m = np.c_[m, np.full(len(m), np.nan, np.float32)]
        return m

    def extras(self) -> list[Path]:
        d = self.dir / "extra"
        return sorted(p for p in d.rglob("*") if p.is_file()) if d.exists() else []


def _npz_key(pair: str) -> str:
    return pair.replace("/", "__")


def _field_warnings(meta: dict) -> list[str]:
    out = []
    legacy = [k for k in meta if k in LEGACY_FIELDS]
    unknown = [k for k in meta if k not in FIELDS and k not in LEGACY_FIELDS]
    if legacy:
        out.append(f"exp.toml 有 v1 旧字段 {', '.join(legacy)}，已忽略")
    if unknown:
        out.append(f"exp.toml 有未知字段 {', '.join(unknown)}，已忽略")
    for m, cfg in meta.get("methods", {}).items():
        extra = [k for k in cfg if k not in METHOD_FIELDS] if isinstance(cfg, dict) else []
        if extra:
            out.append(f"[methods.{m}] 有未知字段 {', '.join(extra)}，已忽略")
    return out


def load_runs(root: Path) -> dict[str, Run]:
    """按目录名索引全部实验。字段有问题只记警告，不报错；一致性交给 check_runs。"""
    runs: dict[str, Run] = {}
    if not Path(root).is_dir():
        return runs
    for d in sorted(p for p in Path(root).iterdir() if (p / "exp.toml").exists()):
        meta = tomllib.loads((d / "exp.toml").read_text(encoding="utf-8"))
        pd = d / "preds"
        methods = sorted(p.name for p in pd.iterdir() if p.is_dir()) if pd.exists() else []
        runs[d.name] = Run(d.name, d, meta, methods, _field_warnings(meta), runs)
    return runs


def check_runs(runs: dict[str, Run]) -> tuple[list[str], list[str]]:
    """按 v2 规则检查记录，返回（问题, 警告）。旧字段、未知字段只算警告。"""
    problems, warnings = [], []
    for r in runs.values():
        m = r.meta
        warnings += [f"{r.id}: {w}" for w in r.warnings]
        for k in ("id", "title"):
            if k not in m:
                problems.append(f"{r.id}: exp.toml 缺少 {k}")
        if "id" in m and m["id"] != r.id:
            problems.append(f"{r.id}: exp.toml 里 id={m['id']!r} 与目录名不一致")
        if "baseline" in m and not isinstance(m["baseline"], bool):
            problems.append(f"{r.id}: baseline 应为 true / false，现为 {m['baseline']!r}")
        if r.parent and r.parent not in runs:
            problems.append(f"{r.id}: 父实验 {r.parent} 不存在")
        if r.init:
            iid, _, im = r.init.partition("/")
            if iid not in runs or not im or (runs[iid].lit and im not in runs[iid].methods):
                problems.append(f"{r.id}: init={r.init!r} 找不到（应写成 <实验>/<方法>）")
    for r in runs.values():
        seen, cur = set(), r.id
        while cur:
            if cur in seen:
                problems.append(f"{r.id}: 父链有环")
                break
            seen.add(cur)
            cur = runs[cur].parent if cur in runs else None
    return problems, warnings


def git_state(repo: Path) -> dict:
    """commit（完整 sha）与 dirty。dirty 只看已跟踪的代码：未跟踪文件和 runs/ 下的产物不算。"""
    def git(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip() if r.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "dirty": None}
    status = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(top,exclude)runs")
    return {"commit": commit, "dirty": bool(status)}


class PredWriter:
    """给跑实验的代码用：逐 pair 写一个方法在一个 split 上的结果。

        with PredWriter("runs/B0", "roma", "test") as w:
            for pair in pairs:
                w.write(pair, A, matches=M)            # A: 2×3 或 None；M: RANSAC 前的全部点对
                w.write(pair, None, fail="few_inliers", n_matches=2)

    点对 matches 取 matcher 的坐标约定（整数 = 像素中心），可以是 N×5 (x_opt, y_opt, x_sar, y_sar, conf)，
    也可以是 N×4 另给 conf=（不给则 conf 记 NaN）。写入端负责：
    - 坐标 +0.5，与 A 和标注（ArcGIS 角点原点）同一约定；
    - 超过 MATCHES_CAP 个点时按 conf 取前 MATCHES_CAP；
    - 0 个点写 0×5；fail 以 "error" 开头的 pair 不写点对。
    close 时写 <split>.meta.json：commit 与 dirty（取自构造时的 repo，默认本仓库）。
    """

    def __init__(self, run_dir, method: str, split: str, repo=REPO):
        assert split in SPLITS, split
        self.dir = Path(run_dir) / "preds" / method
        self.dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self._git = git_state(Path(repo))
        self._f = open(self.dir / f"{split}.jsonl", "w", encoding="utf-8")
        self._matches: dict[str, np.ndarray] = {}

    def write(self, pair: str, A, fail: str | None = None, matches=None, conf=None, **extra):
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
        if matches is not None and not (fail or "").startswith("error"):
            self._matches[_npz_key(pair)] = _pack_matches(matches, conf)

    def close(self):
        self._f.close()
        if self._matches:
            np.savez_compressed(self.dir / f"{self.split}_matches.npz", **self._matches)
        (self.dir / f"{self.split}.meta.json").write_text(json.dumps(self._git, indent=2) + "\n", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _pack_matches(matches, conf=None) -> np.ndarray:
    m = np.asarray(matches, dtype=np.float32)
    if m.size == 0:
        m = m.reshape(0, 5 if conf is None else 4)
    if m.ndim != 2 or m.shape[1] not in (4, 5):
        raise ValueError(f"点对应为 N×4 或 N×5，现为 {m.shape}")
    if m.shape[1] == 4:
        c = np.full(len(m), np.nan, np.float32) if conf is None else np.asarray(conf, np.float32).reshape(-1)
        if len(c) != len(m):
            raise ValueError(f"conf 有 {len(c)} 个，点对有 {len(m)} 个")
        m = np.c_[m, c]
    elif conf is not None:
        raise ValueError("N×5 点对已含 conf，不要再传 conf=")
    else:
        m = m.copy()
    m[:, :4] += 0.5
    if len(m) > MATCHES_CAP:
        m = m[np.argsort(-m[:, 4], kind="stable")[:MATCHES_CAP]]
    return m


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
