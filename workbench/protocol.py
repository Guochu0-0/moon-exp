"""评价协议：检查点误差、pair 误差与 split 级汇总。

口径见「定义评价协议与指标」(#2) 的结论，术语见 CONTEXT.md。
阈值档位见「锁定评价阈值档位」(#9) 的结论：主表 AUC@3/5/10 + SR@3/5/10，选模用 Val AUC@5；
不设 T_粗（无错配率、无 FIRE 式三分）。其余档位照算、照落盘，供附表用。
"""
from __future__ import annotations

import math

import numpy as np

# ---- 档位（#9 锁定） ----
AUC_THRESHOLDS = (3.0, 5.0, 10.0, 20.0)       # 全部计算、落盘
SR_THRESHOLDS = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
TABLE_AUC = (3.0, 5.0, 10.0)                  # 主表档位；其余进附表
TABLE_SR = (3.0, 5.0, 10.0)
MAIN_AUC = 5.0           # 选模与思路树上展示用的主指标 AUC@MAIN_AUC
BOOTSTRAP = 1000
SEED = 0


def pair_error(A, opt_pts: np.ndarray, sar_pts: np.ndarray) -> float:
    """光学检查点经仿射 A（光学 → SAR，2×3）映射后到 SAR 检查点的平均距离。A 为 None 即失败，返回 ∞。"""
    if A is None:
        return math.inf
    A = np.asarray(A, dtype=np.float64).reshape(2, 3)
    proj = opt_pts @ A[:, :2].T + A[:, 2]
    return float(np.linalg.norm(proj - sar_pts, axis=1).mean())


def checkpoint_residuals(A, opt_pts: np.ndarray, sar_pts: np.ndarray) -> np.ndarray | None:
    """逐检查点残差向量（SAR 坐标系，N×2），供逐对可视化画箭头。"""
    if A is None:
        return None
    A = np.asarray(A, dtype=np.float64).reshape(2, 3)
    return opt_pts @ A[:, :2].T + A[:, 2] - sar_pts


def auc(err: np.ndarray, T: float) -> float:
    """误差累积曲线在 [0, T] 下的面积，归一化到 [0, 1]；等价于 mean(max(0, 1 − e/T))，∞ 记 0。"""
    return float(np.clip(1.0 - err / T, 0.0, None).mean()) if len(err) else math.nan


def _point(err: np.ndarray) -> dict:
    n = len(err)
    fail = ~np.isfinite(err)
    out = {
        "n": n,
        "fail_rate": float(fail.mean()) if n else math.nan,
        "median": float(np.median(err)) if n else math.nan,   # 全部 pair，失败计 ∞；过半失败时为 ∞
    }
    for T in AUC_THRESHOLDS:
        out[f"auc@{T:g}"] = auc(err, T)
    for t in SR_THRESHOLDS:
        out[f"sr@{t:g}"] = float((err <= t).mean()) if n else math.nan
    return out


def summarize(err: np.ndarray) -> dict:
    """一个 split 的汇总：全部 pair 等权；AUC 与 SR 附按 pair 重采样的 bootstrap 95% CI。"""
    err = np.asarray(err, dtype=np.float64)
    out = _point(err)
    if len(err):
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(err), size=(BOOTSTRAP, len(err)))
        samples = err[idx]
        keys = [f"auc@{T:g}" for T in AUC_THRESHOLDS] + [f"sr@{t:g}" for t in SR_THRESHOLDS]
        for k in keys:
            kind, v = k.split("@")
            v = float(v)
            if kind == "auc":
                s = np.clip(1.0 - samples / v, 0.0, None).mean(axis=1)
            else:
                s = (samples <= v).mean(axis=1)
            lo, hi = np.percentile(s, [2.5, 97.5])
            out[k + "_ci"] = [float(lo), float(hi)]
    return out


def protocol_info() -> dict:
    return {
        "auc_thresholds": list(AUC_THRESHOLDS),
        "sr_thresholds": list(SR_THRESHOLDS),
        "table_auc": list(TABLE_AUC),
        "table_sr": list(TABLE_SR),
        "main": f"auc@{MAIN_AUC:g}",
        "provisional": False,
    }
