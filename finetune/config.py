"""每个方法的训练配置：`runs/<id>/configs/<方法>.toml`（「训练代码重构」#86）。

    base = "Q4"                 # 可选：先读同一实验里的 configs/Q4.toml，再用本文件覆盖（只能在同一实验内继承）

    [model]                     # 必填 name：loftr | roma（finetune/models/）
    name = "loftr"

    [optim]                     # 优化配方，见 OPTIM
    [run]                       # 种子、存 ckpt 等，见 RUN

    [cexp]                      # 训练成分：出现哪个表就用哪个成分（finetune/parts/），表里只写与默认不同的参数
    r_out = -0.25
    [neg]

展开（expand）= 合并 base → 填上全局默认、模型默认、各成分默认 → 校验。展开后的完整配置写进启动记录和 args.json，
读它就能知道这次训练的全部设置。未知的表或键直接报错，不静默忽略。
"""
from __future__ import annotations

import copy
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:            # 训练环境是 py3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib

OPTIM = {
    "steps": 8000,          # 前向步数（每步一个 batch）
    "batch": 1,
    "lr": 1e-5,
    "wd": 0.0,              # AdamW 的 weight decay
    "warmup": 500,          # 线性 warmup 的步数；0 = 不用
    "warmup_start": 0.1,    # warmup 起点 lr 占 lr 的比例（上游 LoFTR 为 0.1）
    "sched": "cosine",      # warmup 之后：const 恒定 / cosine 降到 lr_min·lr
    "lr_min": 0.0,
    "clip": 0.0,            # 梯度范数裁剪阈值；0 = 不裁
    "accum": 1,             # 梯度累积：每 accum 步更新一次；步数、存 ckpt 仍按前向步计
}
RUN = {
    "seed": 0,
    "workers": 4,           # DataLoader 进程数（A6000 的 /dev/shm 小，不超过 2）
    "save_every": 1000,     # 每多少步存一次 ckpt；0 = 不存
    "save_at": [],          # 额外存 ckpt 的步数
    "save_zero": True,      # 训练前存 ckpt_0
    "limit": 0,             # 只用 Train 的 N 对（均匀取样，调试用）；0 = 全部
    "split": "train",       # 训练用的 split，可用 + 连接；val / test 只用于有标注的上限参考（#96）
}
SPLITS = ("train", "val", "test")
TOP = ("base", "model", "optim", "run")


class ConfigError(ValueError):
    pass


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def read(path) -> dict:
    """读一个方法的配置并解开 base 链（未展开默认值）。base 写同目录下另一个配置的文件名（不带 .toml）。"""
    path = Path(path)
    seen, chain = set(), []
    while True:
        if path.name in seen:
            raise ConfigError(f"base 循环引用：{path}")
        seen.add(path.name)
        if not path.exists():
            raise ConfigError(f"找不到配置 {path}")
        d = tomllib.loads(path.read_text(encoding="utf-8"))
        chain.append(d)
        base = d.get("base")
        if not base:
            break
        if not isinstance(base, str) or "/" in base or "\\" in base or base.endswith(".toml"):
            raise ConfigError(f"{path.name}：base 只能写同一实验里另一个配置的名字（不带 .toml）：{base!r}")
        path = path.with_name(base + ".toml")
    out = {}
    for d in reversed(chain):
        out = _merge(out, {k: v for k, v in d.items() if k != "base"})
    return out


def _fill(name: str, given: dict, defaults: dict) -> dict:
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ConfigError(f"[{name}] 不认识的键 {unknown}，可用 {sorted(defaults)}")
    out = copy.deepcopy(defaults)
    for k, v in given.items():
        d = defaults[k]
        if d is not None and not isinstance(v, type(d)) and not (isinstance(d, float) and isinstance(v, int)):
            raise ConfigError(f"[{name}] {k} 应为 {type(d).__name__}，得到 {v!r}")
        out[k] = float(v) if isinstance(d, float) else v
    return out


def expand(raw: dict) -> dict:
    """未展开的配置 → 完整配置：每个表都带上全部键。"""
    from .models import MODELS
    from .parts import PARTS

    raw = copy.deepcopy(raw)
    raw.pop("base", None)
    m = raw.get("model", {}).get("name")
    if m not in MODELS:
        raise ConfigError(f"[model] name 必填，可选 {sorted(MODELS)}，得到 {m!r}")
    model = MODELS[m]
    unknown = sorted(set(raw) - set(TOP) - set(PARTS))
    if unknown:
        raise ConfigError(f"不认识的表 {unknown}；训练成分可选 {sorted(PARTS)}")
    out = {"model": _fill("model", raw.pop("model"), {"name": m, **model.PARAMS}),
           "optim": _fill("optim", raw.pop("optim", {}), {**OPTIM, **model.OPTIM}),
           "run": _fill("run", raw.pop("run", {}), {**RUN, **model.RUN})}
    bad = [x for x in out["run"]["split"].split("+") if x not in SPLITS]
    if bad:
        raise ConfigError(f"[run] split 只能由 {SPLITS} 用 + 连接，得到 {out['run']['split']!r}")
    for name, part in PARTS.items():      # 按登记顺序：每步也按这个顺序算损失
        if name not in raw:
            continue
        if m not in part.MODELS:
            raise ConfigError(f"训练成分 [{name}] 不支持模型 {m}（支持 {sorted(part.MODELS)}）")
        out[name] = part.fill(_fill(name, raw[name], {**part.PARAMS, **part.MODELS[m]}))
    for name, part in PARTS.items():
        if name in out:
            part.check(out)
    return out


def load(path) -> dict:
    """读 + 展开。"""
    try:
        return expand(read(path))
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def parts_of(cfg: dict) -> list[str]:
    """配置里用到的训练成分，按登记顺序。"""
    from .parts import PARTS
    return [n for n in PARTS if n in cfg]
