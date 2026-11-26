"""实验的数据划分及各主机的数据入口。推理环境兼容 Python 3.8。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

LEGACY = "v1"
CURRENT = "v2"
VERSIONS = (LEGACY, CURRENT)
ROOT_ENV = {LEGACY: "MOON_DATA", CURRENT: "MOON_DATA_V2"}
INLIER_SELECTION = "val_inlier_mean"


def version_of(meta) -> str:
    version = meta.get("data_version", LEGACY)
    if version not in VERSIONS:
        raise ValueError(f"data_version 应为 v1 或 v2，现为 {version!r}")
    return version


def experiment_version(run_dir) -> str:
    path = Path(run_dir) / "exp.toml"
    if not path.is_file():
        return LEGACY
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            from pip._vendor import tomli as tomllib
    with path.open("rb") as f:
        return version_of(tomllib.load(f))


def root_for(version, explicit=None, default=None) -> Path | None:
    root = explicit or os.environ.get(ROOT_ENV[version]) or default
    return Path(root) if root else None


def root_error(version, root, split=None) -> str:
    return f"Moon-{version}{' ' + split.title() if split else ''} 数据目录不可用：{root or '未配置'}（{ROOT_ENV[version]}）"


def required_root(version, explicit=None) -> Path:
    root = root_for(version, explicit)
    if root is None or not root.is_dir():
        raise ValueError(root_error(version, root))
    return root


def manifest_sha256(pairs) -> str:
    return hashlib.sha256("\n".join(sorted(pairs)).encode("utf-8")).hexdigest()


def sha256(path, bufsize=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(bufsize), b""):
            h.update(block)
    return h.hexdigest()


def prediction_sha256(path) -> str:
    """Git 在 Windows 检出时会把 JSONL 的 LF 转成 CRLF。"""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
