"""实验记录：读（给工作台）与写（给跑实验的代码）。

一个实验一个目录 runs/<id>/，格式（记录格式 v2）见 workbench/README.md：

    runs/<id>/
      exp.toml                              元数据：id、title、parent、init、baseline、date、[methods.<m>]
      notes.md                              可选：实验唯一的自由文本
      preds/<method>/<split>.jsonl          每个 pair 一行：估计仿射（光学 → SAR，2×3）或失败原因
      preds/<method>/<split>.meta.json      写入时的 commit 与 dirty
      preds/<method>/<split>_matches.npz    点对 N×5 (x_opt, y_opt, x_sar, y_sar, conf)，不进 git
      inter/<method>/<name>/meta.json       中间结果的种类、坐标系、说明、单位
      inter/<method>/<name>/<split>.npz     中间结果数据（scalar / points / flow），不进 git
      inter/<method>/<name>/<split>/*.png   中间结果数据（image），不进 git
      tb/<method>/                          TensorBoard 日志，不进 git
      extra/                               附件，Notes 引用时才显示
      metrics.json                          派生物：`python -m workbench eval` 写出，勿手改
"""
from __future__ import annotations

import datetime
import json
import math
import re
import shutil
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
INTER_KINDS = ("scalar", "points", "flow", "image")
INTER_FRAMES = ("opt", "sar")
INTER_META = ("kind", "frame", "desc", "unit")
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

    def splits(self, method: str) -> list[str]:
        """这个方法有 preds 的 split。"""
        return [s for s in SPLITS if self._has(method, s)]

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

    def inters(self, method: str) -> dict[str, dict]:
        """这个方法的中间结果清单 {名称: meta}，来自 inter/<method>/<name>/meta.json（进 git，数据不在本地也有）。
        splits 是写完过的 split；没有这个字段（手写的 meta）时为 None，表示不知道。"""
        d = self.dir / "inter" / method
        out = {}
        for f in sorted(d.glob("*/meta.json")) if d.is_dir() else []:
            meta = json.loads(f.read_text(encoding="utf-8"))
            out[f.parent.name] = {**{k: meta.get(k, "") for k in INTER_META}, "splits": meta.get("splits")}
        return out

    def inter_state(self, method: str, name: str, split: str, pair: str) -> str:
        """一个 pair 的中间结果状态：ok / absent（没写这个 pair 或这个 split）/ not_synced（数据留在服务器上）。"""
        meta = self.inters(method)[name]
        if meta["splits"] is not None and split not in meta["splits"]:
            return "absent"
        d = self.dir / "inter" / method / name
        if not ((d / f"{split}.npz").is_file() or (d / split).is_dir()):
            return "not_synced"
        return "ok" if self.inter(method, name, split, pair, meta["kind"]) is not None else "absent"

    def inter(self, method: str, name: str, split: str, pair: str, kind: str):
        """一个 pair 的中间结果：scalar / points / flow 为数组，image 为 PNG 路径；这个 pair 没有时为 None。"""
        d = self.dir / "inter" / method / name
        if kind == "image":
            f = d / split / f"{_npz_key(pair)}.png"
            return f if f.is_file() else None
        f = d / f"{split}.npz"
        if not f.is_file():
            return None
        with np.load(f) as z:
            key = _npz_key(pair)
            return z[key] if key in z.files else None

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


# ---------------- 实验编辑：CLI 与服务端共用 ----------------
#
# 都是对 exp.toml 的读-改-写：按行改顶层字段，注释、[methods] 和其余字段原样保留。
# 多个写入方各自读-改-写，以后写的为准，不加锁。

ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

EXP_TEMPLATE = """id = {id}
title = {title}
{links}date = "{date}"
"""


class EditError(ValueError):
    """编辑被拒绝（成环、已点亮、编号冲突等）。消息直接给用户看。"""


class NotFound(EditError):
    pass


def _q(v: str) -> str:
    return json.dumps(v, ensure_ascii=False)   # JSON 字符串即合法的 TOML 基本字符串


def _get(root: Path, rid: str) -> Run:
    r = load_runs(root).get(rid)
    if r is None:
        raise NotFound(f"实验 {rid} 不存在")
    return r


def next_id(root: Path) -> str:
    """下一个 E<n>：现有 E<数字> 中最大的加一。"""
    root = Path(root)
    ns = [int(m.group(1)) for d in root.iterdir() if (m := re.fullmatch(r"E(\d+)", d.name))] if root.is_dir() else []
    return f"E{max(ns, default=0) + 1}"


def create_experiment(root: Path, rid: str, title: str = "", parent: str | None = None,
                      init: str | None = None) -> Path:
    """新建未点亮实验：只写 runs/<id>/exp.toml。"""
    root = Path(root)
    if not ID_RE.fullmatch(rid or ""):
        raise EditError(f"编号 {rid!r} 不合法：只能用字母、数字、_ . -，且以字母或数字开头")
    d = root / rid
    if d.exists():
        raise EditError(f"{rid} 已存在")
    if parent or init:
        _check_link(load_runs(root), rid, parent, init)
    links = (f"parent = {_q(parent)}\n" if parent else "") + \
        (f"init = {_q(init)}\n" if init else '# init = "<父实验>/<方法>"   # 可选：从父实验的哪个方法起步\n')
    d.mkdir(parents=True)
    (d / "exp.toml").write_text(EXP_TEMPLATE.format(id=_q(rid), title=_q(title), links=links,
                                                    date=datetime.date.today().isoformat()),
                                encoding="utf-8", newline="\n")
    return d


def _check_link(runs: dict[str, Run], rid: str, parent: str | None, init: str | None):
    if init and not parent:
        raise EditError("有 init 就必须有父实验")
    if not parent:
        return
    if parent not in runs:
        raise EditError(f"父实验 {parent} 不存在")
    seen, cur = set(), parent
    while cur and cur not in seen:
        if cur == rid:
            raise EditError(f"不能把 {rid} 接到{'它自己' if parent == rid else f'它的后代 {parent}'}下面：会成环")
        seen.add(cur)
        cur = runs[cur].parent if cur in runs else None
    if init:
        iid, _, im = init.partition("/")
        if iid != parent or not im:
            raise EditError(f"init 应写成 {parent}/<方法>，现为 {init!r}")
        if runs[parent].lit and im not in runs[parent].methods:
            raise EditError(f"{parent} 没有方法 {im}")


def set_parent(root: Path, rid: str, parent: str | None, init: str | None = None):
    """改父 / init；parent 为空即断开父实验。会成环时拒绝，文件不动。"""
    runs = load_runs(root)
    if rid not in runs:
        raise NotFound(f"实验 {rid} 不存在")
    _check_link(runs, rid, parent or None, init or None)
    _edit_toml(runs[rid].dir / "exp.toml", parent=parent or None, init=init or None)


def set_title(root: Path, rid: str, title: str):
    _edit_toml(_get(root, rid).dir / "exp.toml", title=title)


def rename_experiment(root: Path, old: str, new: str):
    """改编号：只在未点亮时允许。目录改名，并改掉其他实验里指向它的 parent 与 init 前缀。"""
    runs = load_runs(root)
    if old not in runs:
        raise NotFound(f"实验 {old} 不存在")
    if runs[old].lit:
        raise EditError(f"{old} 已点亮，不能改编号")
    if new == old:
        return
    if not ID_RE.fullmatch(new or ""):
        raise EditError(f"编号 {new!r} 不合法：只能用字母、数字、_ . -，且以字母或数字开头")
    if (Path(root) / new).exists():
        raise EditError(f"{new} 已存在")
    d = runs[old].dir.rename(Path(root) / new)   # 先改目录：失败时什么都还没动
    _edit_toml(d / "exp.toml", id=new)
    for r in runs.values():
        if r.id == old:
            continue
        ch = {}
        if r.parent == old:
            ch["parent"] = new
        if r.init and r.init.partition("/")[0] == old:
            ch["init"] = f"{new}/{r.init.partition('/')[2]}"
        if ch:
            _edit_toml(r.dir / "exp.toml", **ch)


def delete_experiment(root: Path, rid: str):
    """删除：只允许未点亮的实验（删目录）。它的子实验断开父实验，成为根；别处指向它的 init 一并删掉。"""
    runs = load_runs(root)
    if rid not in runs:
        raise NotFound(f"实验 {rid} 不存在")
    if runs[rid].lit:
        raise EditError(f"{rid} 已点亮，不能删除")
    for r in runs.values():
        if r.id == rid:
            continue
        if r.parent == rid:
            _edit_toml(r.dir / "exp.toml", parent=None, init=None)
        elif (r.init or "").partition("/")[0] == rid:
            _edit_toml(r.dir / "exp.toml", init=None)
    shutil.rmtree(runs[rid].dir)


def save_notes(root: Path, rid: str, text: str):
    """写 runs/<id>/notes.md：UTF-8、LF 换行；文件第一次保存时才创建。"""
    if not isinstance(text, str):
        raise EditError("notes 应为字符串")
    path = _get(root, rid).dir / "notes.md"
    path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def _edit_toml(path: Path, **fields):
    """按行改 exp.toml 的顶层字符串字段；值为 None 即删掉该行。改完解析核对，不符就报错，不写文件。"""
    text = path.read_text(encoding="utf-8")
    before = tomllib.loads(text)
    lines = text.splitlines()
    top = next((i for i, l in enumerate(lines) if re.match(r"\s*\[", l)), len(lines))
    for key, value in fields.items():
        at = [i for i in range(top) if re.match(rf"\s*{key}\s*=", lines[i])]
        for i in reversed(at[1:] if value is not None else at):
            del lines[i]
            top -= 1
        if value is None:
            continue
        line = f"{key} = {_q(value)}"
        hint = [i for i in range(top) if re.match(rf"#\s*{key}\s*=", lines[i])]   # 模板里注释掉的占位行
        if at or hint:
            lines[(at or hint)[0]] = line
            continue
        rank = FIELDS.index(key)
        pos = [i for i in range(top) for k in FIELDS[:rank] if re.match(rf"\s*{k}\s*=", lines[i])]
        lines.insert(max(pos) + 1 if pos else 0, line)
        top += 1
    new_text = "\n".join(lines) + "\n"
    after = tomllib.loads(new_text)
    expect = {k: v for k, v in before.items() if k not in fields}
    expect.update({k: v for k, v in fields.items() if v is not None})
    if after != expect:
        raise EditError(f"{path} 的写法无法按行安全修改，请手改")
    path.write_text(new_text, encoding="utf-8", newline="\n")


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


class InterWriter:
    """给跑实验的代码用：逐 pair 写一个方法的一个中间结果在一个 split 上的数据，可以只覆盖部分 pair。

        with InterWriter("runs/E3", "main", "certainty", "val", kind="scalar", frame="opt", desc="RoMa certainty") as w:
            for pair in pairs:
                w.write(pair, cert)                    # cert: H×W，分辨率任意

    kind 与数据形状（写入时校验，不符就报错）：
    - scalar：H×W 标量图；
    - points：N×3 (x, y, v)，坐标取 matcher 的约定（整数 = 像素中心），写入端 +0.5，与点对一致；
    - flow：H×W×2，frame 坐标系下每个像素指向另一模态中对应点的位移 (dx, dy)，单位是原始 patch 的 px；
    - image：H×W、H×W×3 或 H×W×4 的 uint8，每个 pair 一张 PNG。
    frame 为 opt / sar：数据所在的坐标系，显示时按它拉伸到 patch 大小。
    构造时写 meta.json（进 git）并清掉这个 split 的旧数据；close 时把 scalar / points / flow 写成 <split>.npz（不进 git），
    再把这个 split 记进 meta.json 的 splits：没写过的 split 读出来是「没有」，不会被当成「数据留在服务器上」。
    with 块里抛异常时丢掉这次写的数据，这个 split 不记。
    """

    def __init__(self, run_dir, method: str, name: str, split: str, *, kind: str, frame: str,
                 desc: str = "", unit: str = ""):
        assert split in SPLITS, split
        if kind not in INTER_KINDS:
            raise ValueError(f"kind 应为 {' / '.join(INTER_KINDS)}，现为 {kind!r}")
        if frame not in INTER_FRAMES:
            raise ValueError(f"frame 应为 {' / '.join(INTER_FRAMES)}，现为 {frame!r}")
        self.dir = Path(run_dir) / "inter" / method / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kind, self.split = kind, split
        old = self.dir / "meta.json"
        splits = (json.loads(old.read_text(encoding="utf-8")).get("splits") or []) if old.exists() else []
        self._meta = {"kind": kind, "frame": frame, "desc": desc, "unit": unit, "splits": sorted(set(splits) - {split})}
        self._save_meta()
        self._discard()
        self._data: dict[str, np.ndarray] = {}

    def _save_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self._meta, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")

    def _discard(self):
        (self.dir / f"{self.split}.npz").unlink(missing_ok=True)
        if (self.dir / self.split).is_dir():
            shutil.rmtree(self.dir / self.split)

    def write(self, pair: str, data):
        a = np.asarray(data)
        ok = {"scalar": a.ndim == 2,
              "points": a.ndim == 2 and a.shape[1] == 3,
              "flow": a.ndim == 3 and a.shape[2] == 2,
              "image": a.dtype == np.uint8 and (a.ndim == 2 or (a.ndim == 3 and a.shape[2] in (3, 4)))}[self.kind]
        if not ok or (self.kind != "points" and a.size == 0):
            shape = {"scalar": "H×W", "points": "N×3", "flow": "H×W×2", "image": "H×W、H×W×3 或 H×W×4 的 uint8"}
            raise ValueError(f"{self.kind} 中间结果应为 {shape[self.kind]}，{pair} 的是 {a.shape} {a.dtype}")
        if self.kind == "image":
            from PIL import Image

            (self.dir / self.split).mkdir(exist_ok=True)
            Image.fromarray(a).save(self.dir / self.split / f"{_npz_key(pair)}.png")
            return
        a = a.astype(np.float32)
        if self.kind == "points":
            a[:, :2] += 0.5
        self._data[_npz_key(pair)] = a

    def close(self):
        if self.kind == "image":
            (self.dir / self.split).mkdir(exist_ok=True)
        else:
            np.savez_compressed(self.dir / f"{self.split}.npz", **self._data)
        self._meta["splits"] = sorted({*self._meta["splits"], self.split})
        self._save_meta()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            self.close()
        else:
            self._discard()


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
