"""难版评测：SAR 侧叠加已知 T 的方法，按扰动后的真值评测和显示（#138，T 的定义见 moonlib/perturb.py）。

实验在 exp.toml 写 `perturb = true`，T 存在 runs/<id>/perturb/<变体>/<split>.json；方法名的最后一段 `__<变体>`
对得上这个目录的，就按该变体评测：检查点 SAR 一侧变为 T(q)，全部检查点都计入，落到扰动后 SAR 画面外的也照算
（#159：误差在原 SAR 坐标系里量，真值精确已知，画面外的点只是要求仿射外推；剔除它们会去掉误差最大的边缘点、抬高重档分数）；
误差把预测换回原 SAR 坐标系（T⁻¹∘A）再量，单位仍是原始像素（见 PerturbedView.original）；
逐对视图里的 SAR 影像同样套上 T。其余方法照常。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .dataset import Dataset


def variant(run, method: str) -> str | None:
    """这个方法的扰动变体；实验没声明 perturb、或方法名没有对应的 T 时为 None。"""
    if run.meta.get("perturb") is not True or "__" not in method:
        return None
    v = method.rpartition("__")[2]
    return v if (run.dir / "perturb" / v).is_dir() else None


def dataset_for(ds: Dataset, run, method: str):
    v = variant(run, method)
    return ds if v is None else view(ds, run.dir / "perturb" / v)


@lru_cache(maxsize=256)
def view(ds: Dataset, vdir: Path) -> "PerturbedView":
    return PerturbedView(ds, vdir)


class PerturbedView:
    """与 Dataset 同接口；只改 labelled / checkpoints，另给 sar_display；其余照转。"""

    def __init__(self, ds: Dataset, vdir: Path):
        self.ds, self.dir, self.key = ds, Path(vdir), f"{Path(vdir).parent.parent.name}/{Path(vdir).name}"
        self._T: dict[str, dict] = {}

    def __getattr__(self, name):          # pairs、numbers、size、optical、root 照转
        return getattr(self.ds, name)

    def T(self, split: str, pair: str) -> np.ndarray | None:
        if split not in self._T:
            f = self.dir / f"{split}.json"
            self._T[split] = ({p: np.asarray(t, np.float64) for p, t in json.loads(f.read_text(encoding="utf-8"))["T"].items()}
                              if f.exists() else {})
        return self._T[split].get(pair)

    @lru_cache(maxsize=None)
    def checkpoints(self, split: str, pair: str):
        """(光学 N×2, 扰动后的 SAR N×2)，全部检查点，含落到 SAR 画面外的；没有检查点或没有 T 时为 None。"""
        cp, T = self.ds.checkpoints(split, pair), self.T(split, pair)
        if cp is None or T is None:
            return None
        opt, sar = cp
        return opt, sar @ T[:, :2].T + T[:, 2]

    def original(self, split: str, pair: str, A):
        """评测用：把扰动坐标系里的仿射 A 换回原 SAR 坐标系（T⁻¹∘A），连同检查点（原坐标）一起返回。
        误差按原 SAR 像素计：缩放变体若在扰动后的网格里量，误差会被放大或缩小 1 ± s 倍，与原协议的阈值不可比。
        平移、旋转是等距变换，两种量法结果相同。"""
        cp, T = self.checkpoints(split, pair), self.T(split, pair)
        if cp is None:
            return A, None
        opt, q = cp
        Li = np.linalg.inv(T[:, :2])
        Ti = np.c_[Li, -Li @ T[:, 2]]
        sar = q @ Ti[:, :2].T + Ti[:, 2]
        if A is not None:
            H = lambda M: np.r_[np.asarray(M, np.float64).reshape(2, 3), [[0, 0, 1]]]
            A = (H(Ti) @ H(A))[:2]
        return A, (opt, sar)

    @lru_cache(maxsize=None)
    def labelled(self, split: str) -> list[str]:
        return [p for p in self.ds.labelled(split) if self.checkpoints(split, p) is not None]

    def outside(self, split: str) -> dict:
        """落到扰动后 SAR 画面外的检查点数（仍计入评测），以及全部检查点都在画面外的 pair 数。"""
        n, pairs = 0, 0
        for p in self.labelled(split):
            q = self.checkpoints(split, p)[1]
            w, h = self.ds.size(split, p, "SAR")
            out = (q[:, 0] < 0) | (q[:, 0] >= w) | (q[:, 1] < 0) | (q[:, 1] >= h)
            n, pairs = n + int(out.sum()), pairs + bool(out.all())
        return {"points": n, "pairs": pairs}

    def sar_display(self, img: np.ndarray, split: str, pair: str) -> np.ndarray:
        """显示用的 SAR（已拉伸到 uint8）套上 T；先拉伸再 warp，补的 0 不影响拉伸。"""
        T = self.T(split, pair)
        return img if T is None else warp(img, T)


def warp(img: np.ndarray, T: np.ndarray) -> np.ndarray:
    """按 T（角点约定，原 → 新）warp 影像，双线性、图外补 0；与 finetune.augment.warp 同口径，只用 numpy。"""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    Li = np.linalg.inv(T[:, :2])
    q = np.stack([x + 0.5, y + 0.5], -1) - T[:, 2]          # 新图像素中心（角点约定）
    p = q @ Li.T - 0.5                                     # 原图上的位置（中心约定）
    px, py = p[..., 0], p[..., 1]
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    fx, fy = px - x0, py - y0
    src = img.astype(np.float64)
    out = np.zeros_like(src)
    if src.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    for dy, dx, wgt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        v = np.zeros_like(src)
        v[ok] = src[yi[ok], xi[ok]]
        out += wgt * v
    inside = (px >= -0.5) & (px <= w - 0.5) & (py >= -0.5) & (py <= h - 0.5)   # 与 cv2 的补 0 边界大致一致
    out[~inside] = 0
    if np.issubdtype(img.dtype, np.floating):
        return out.astype(img.dtype)
    info = np.iinfo(img.dtype)
    return np.clip(np.rint(out), info.min, info.max).astype(img.dtype)
