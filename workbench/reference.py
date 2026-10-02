"""参考方法：节点页上「方法 vs 参考方法」的默认值与候选。纯函数，规则见 #37 用户故事 42、43。

参考方法用一个 key 表示：`<实验>/<方法>`；IDENTITY 是未配准（恒等变换）；None 是无参考。
score(rid, method) 给出该方法 Val 上的主指标，没有 Val 结果时为 None。
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import protocol
from .records import SPLITS, Run

IDENTITY = "identity"
Score = Callable[[str, str], "float | None"]


def base_method(method: str) -> str:
    """变体 `<方法>__<变体>` 的基础方法 `<方法>`；不是变体就是它自己。"""
    return method.partition("__")[0]


def ranked(run: Run, score: Score) -> list[str]:
    """按 Val 主指标降序；没有 Val 结果的排最后，同分按方法名。"""
    return sorted(run.methods, key=lambda m: (score(run.id, m) is None, -(score(run.id, m) or 0), m))


def default_ref(run: Run, method: str, runs: dict[str, Run], score: Score) -> str | None:
    """有 init 用 init；父实验有同名基础方法就用它；否则用父实验 Val 主指标最好的方法；
    都没有时，基线无参考，其余对比未配准。"""
    if run.init:
        iid, _, im = run.init.partition("/")
        if iid in runs and im in runs[iid].methods:
            return run.init
    p = runs.get(run.parent) if run.parent else None
    if p is not None and p.lit:
        if base_method(method) in p.methods:
            return f"{p.id}/{base_method(method)}"
        best = [m for m in ranked(p, score) if score(p.id, m) is not None]
        if best:
            return f"{p.id}/{best[0]}"
    return None if run.baseline else IDENTITY


def parse(key: str | None, runs: dict[str, Run]) -> tuple[Run, str] | str | None:
    """key → (实验, 方法)、IDENTITY 或 None；找不到时抛 KeyError。"""
    if key in (None, "", IDENTITY):
        return key or None
    rid, _, m = key.partition("/")
    if rid not in runs or m not in runs[rid].methods:
        raise KeyError(key)
    return runs[rid], m


def label(key: str | None, runs: dict[str, Run]) -> str:
    """给人看的名字：单方法实验只写实验编号，其余写「实验 / 显示名」。"""
    if key is None:
        return "无"
    if key == IDENTITY:
        return "未配准"
    r, m = parse(key, runs)
    return r.id if len(r.methods) == 1 else f"{r.id} / {r.method_info(m)['name']}"


def paired_diff(err: np.ndarray, ref_err: np.ndarray, summary: dict, ref_summary: dict) -> dict:
    """方法 − 参考方法：每个指标的差值；AUC 与 SR 另附 95% CI。

    CI 用配对 bootstrap：两边按同一组 pair 下标重采样（次数与种子同评价协议），取差值的 2.5 / 97.5 百分位。
    做法见 Efron & Tibshirani (1993) *An Introduction to the Bootstrap* 第 13 章（百分位区间），
    以及 Koehn (2004) "Statistical Significance Tests for Machine Translation Evaluation"（配对 bootstrap 重采样）。
    两边误差须按同一 pair 顺序排列（ds.labelled(split)）。
    """
    out = {}
    for k, v in summary.items():
        if k.endswith("_ci") or k == "n":
            continue
        rv = ref_summary.get(k)
        out[k] = v - rv if _finite(v) and _finite(rv) else None
    if len(err) and len(err) == len(ref_err):
        rng = np.random.default_rng(protocol.SEED)
        idx = rng.integers(0, len(err), size=(protocol.BOOTSTRAP, len(err)))
        a, b = err[idx], ref_err[idx]
        for T in protocol.AUC_THRESHOLDS:
            s = np.clip(1.0 - a / T, 0.0, None).mean(axis=1) - np.clip(1.0 - b / T, 0.0, None).mean(axis=1)
            out[f"auc@{T:g}_ci"] = [float(x) for x in np.percentile(s, [2.5, 97.5])]
        for t in protocol.SR_THRESHOLDS:
            s = (a <= t).mean(axis=1) - (b <= t).mean(axis=1)
            out[f"sr@{t:g}_ci"] = [float(x) for x in np.percentile(s, [2.5, 97.5])]
    return out


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def groups(run: Run, runs: dict[str, Run], score: Score) -> list[dict]:
    """候选：父实验的各方法、本实验的方法、其他已点亮实验、未配准与无。
    「本实验」组包含全部方法，当前方法由页面自己排除。每个候选附上它有结果的 split，供页面置灰 Test。"""
    out = []
    p = runs.get(run.parent) if run.parent else None
    order = ([(f"父实验 {p.id}", p)] if p is not None and p.lit else []) + [(f"本实验 {run.id}", run)]
    order += [(f"实验 {r.id}", r) for r in runs.values() if r.lit and r.id != run.id and r is not p]
    for title, r in order:
        if r.methods:
            out.append({"label": title, "options": [{"key": f"{r.id}/{m}", "label": label(f"{r.id}/{m}", runs),
                                                     "splits": r.splits(m)} for m in ranked(r, score)]})
    out.append({"label": "其他", "options": [{"key": IDENTITY, "label": label(IDENTITY, runs), "splits": list(SPLITS)},
                                           {"key": None, "label": label(None, runs), "splits": list(SPLITS)}]})
    return out
