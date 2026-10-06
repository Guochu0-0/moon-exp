"""E4 共用：6 个模型、Val 数据与标注、带「重要性」记录的推理。在训练环境（/opt/envs/loftr）里跑。

重要性来源（推理时顺带记下）：
- LoFTR：粗级 transformer 每层、每张图上「每个位置被关注的总量」。粗级是线性注意力
  A_ij = φ(q_i)·φ(k_j) / Σ_j' φ(q_i)·φ(k_j')，列和 Σ_i A_ij = φ(k_j)·Σ_i φ(q_i)/Z_i 可闭式算出，不需要显式注意力矩阵。
  8 个头求和后除以 头数×查询数，均值为 1。层顺序同 layer_names（self, cross 交替 ×4）；
  imp[l, 0] 是光学图上被关注的总量，imp[l, 1] 是 SAR 图上的（self 层是同图内的关注，cross 层是另一张图的查询关注它）。
- RoMa：match() 输出的 certainty（symmetric，左半为光学像素、右半为 SAR 像素），面积平均降到 64×64。
- 两者都记 RANSAC（3 px）内点的坐标。

坐标：内点与仿射都取标注（检查点）的约定（角点原点，像素中心在 c + 0.5），与 baselines.fit 相同。
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                         # runs/E4
REPO = RUN.parents[1]                     # worktree 根
sys.path.insert(0, str(REPO))

from baselines.data import Data          # noqa: E402
from moonlib import inputs               # noqa: E402
from moonlib.ransac import fit_affine    # noqa: E402

Y = "/remote-home/xufang/YGC"
MAIN = Path(os.environ.get("MAIN", f"{Y}/moon-exp"))     # ckpt 只在 gpfs 主 checkout 上
DATA = os.environ.get("MOON_DATA", f"{Y}/dataset/Moon")
WEIGHTS = os.environ.get("MOON_WEIGHTS", f"{Y}/weights")
os.environ.setdefault("TORCH_HOME", f"{Y}/weights/torch_home")
HW = 512

# 名字 → 推理配置、ckpt（相对主 checkout；None = 推理配置里的发布权重）、对照用的已有预测（实验/方法）
MODELS = {
    "loftr_zs": dict(family="loftr", config="configs/baselines/anymatch_loftr.json", ckpt=None,
                     ref="B0/anymatch_loftr", name="LoFTR zero-shot"),
    "loftr_q4": dict(family="loftr", config="configs/baselines/anymatch_loftr.json",
                     ckpt="runs/Q/ckpt/Q4/ckpt_1500.pt", ref="Q/Q4", name="LoFTR 无标注训练"),
    "loftr_e1": dict(family="loftr", config="configs/baselines/anymatch_loftr.json",
                     ckpt="runs/E1/ckpt/loftr_lr5e5/ckpt_14000.pt", ref="E1/loftr_lr5e5", name="LoFTR 标注过拟合"),
    "roma_zs": dict(family="roma", config="configs/baselines/anymatch_roma__minmax.json", ckpt=None,
                    ref="B0m/anymatch_roma__minmax", name="RoMa zero-shot"),
    "roma_m4": dict(family="roma", config="configs/baselines/anymatch_roma__minmax.json",
                    ckpt="runs/M/ckpt/M4/ckpt_2000.pt", ref="M/M4", name="RoMa 自训练"),
    "roma_e1": dict(family="roma", config="configs/baselines/anymatch_roma__minmax.json",
                    ckpt="runs/E1/ckpt/roma_vgg/ckpt_16000.pt", ref="E1/roma_vgg", name="RoMa 标注过拟合"),
}
COUPLES = [("loftr_zs", "roma_zs"), ("loftr_q4", "roma_m4"), ("loftr_e1", "roma_e1")]
LAYERS = ["self", "cross"] * 4


def key(pair: str) -> str:
    return pair.replace("/", "__")


def val_pairs() -> list:
    return Data(DATA).pairs("val", labelled_only=True)


def checkpoints(pair: str, split: str = "val"):
    """(光学 N×2, SAR N×2)，角点约定，y 向下；同 workbench.dataset.Dataset.checkpoints。"""
    a = np.loadtxt(Data(DATA).path(split, pair, "Label"), dtype=np.float64, ndmin=2)
    a[:, [1, 3]] *= -1
    return a[:, :2], a[:, 2:4]


def fit_gt(opt_pts, sar_pts) -> np.ndarray:
    """用该对自己的标注点最小二乘拟合的仿射（光学 → SAR，2×3）。"""
    X = np.c_[opt_pts, np.ones(len(opt_pts))]
    return np.linalg.lstsq(X, sar_pts, rcond=None)[0].T


def apply(A, pts):
    A = np.asarray(A, np.float64).reshape(2, 3)
    return pts @ A[:, :2].T + A[:, 2]


def ref_preds(name: str) -> dict:
    exp, m = MODELS[name]["ref"].split("/")
    out = {}
    for line in (REPO / "runs" / exp / "preds" / m / "val.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["pair"]] = None if d.get("A") is None else np.asarray(d["A"], np.float64)
    return out


GRID17 = np.stack(np.meshgrid(np.linspace(0, HW, 17), np.linspace(0, HW, 17)), -1).reshape(-1, 2)


def affine_change(A1, A0) -> float:
    """两个仿射在整张 patch 上（17×17 网格点）映射结果的平均距离，px。任一为 None 返回 nan。"""
    if A1 is None or A0 is None:
        return float("nan")
    return float(np.linalg.norm(apply(A1, GRID17) - apply(A0, GRID17), axis=1).mean())


class Runner:
    """加载一个模型；__call__(opt, sar) → (A 或 None, 内点 N×4 角点约定, 重要性图)。"""

    def __init__(self, name: str, device="cuda", record=True):
        import torch

        from baselines import adapters
        from baselines.match import seed_all

        self.name, self.spec = name, MODELS[name]
        self.family = self.spec["family"]
        cfg = json.loads((REPO / self.spec["config"]).read_text(encoding="utf-8"))
        self.cfg, self.seed, self.seed_all, self.torch = cfg, int(cfg.get("seed", 0)), seed_all, torch
        weights = str(MAIN / self.spec["ckpt"]) if self.spec["ckpt"] else str(Path(WEIGHTS) / cfg["weights"])
        self.weights = weights
        self.map_opt = inputs.get("optical", cfg["input"]["optical"])
        self.map_sar = inputs.get("sar", cfg["input"]["sar"])
        seed_all(self.seed)
        self.ad = adapters.load(cfg["adapter"])(repo=REPO / cfg["repo"], weights=weights, device=device,
                                                **cfg.get("params", {}))
        self.record = record
        self._buf = []
        if self.family == "loftr" and record:
            for layer in self.ad.model.loftr_coarse.layers:
                layer.attention.forward = types.MethodType(self._make_rec(layer.attention.forward), layer.attention)

    def _make_rec(self, orig):
        torch, buf = self.torch, self._buf

        def rec(att, queries, keys, values, q_mask=None, kv_mask=None):
            out = orig(queries, keys, values, q_mask, kv_mask)
            assert q_mask is None and kv_mask is None
            Q = torch.nn.functional.elu(queries.float()) + 1
            K = torch.nn.functional.elu(keys.float()) + 1
            Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + att.eps)
            w = torch.einsum("nlhd,nlh->nhd", Q, Z)
            col = torch.einsum("nshd,nhd->ns", K, w)               # 各头求和：每头列和之和 = 查询数
            buf.append((col[0] / (Q.shape[2] * Q.shape[1]) * K.shape[1]).cpu().numpy())
            return out
        return rec

    def load_pair(self, pair: str, split="val"):
        d = Data(DATA)
        return self.map_opt(d.optical(split, pair)), self.map_sar(d.sar(split, pair))

    def __call__(self, opt, sar):
        self.seed_all(self.seed)
        self._buf.clear()
        imp = None
        if self.family == "loftr":
            kp0, kp1, conf = self.ad.match(opt, sar)
            if self.record:
                n = int(round(len(self._buf[0]) ** 0.5))
                b = [x.reshape(n, n) for x in self._buf]
                imp = np.stack([np.stack([b[2 * l], b[2 * l + 1]]) if t == "self" else
                                np.stack([b[2 * l + 1], b[2 * l]]) for l, t in enumerate(LAYERS)]).astype(np.float16)
            dense = None
        else:
            kp0, kp1, conf, imp, dense = self._roma(opt, sar)
        M = np.c_[kp0, kp1, conf].astype(np.float32) if len(kp0) else np.zeros((0, 5), np.float32)
        A, inl, _ = fit_affine(M, 3.0)
        pts = (M[inl, :4].astype(np.float64) + 0.5) if inl is not None else np.zeros((0, 4))
        # 供 refit 用的对应：LoFTR 为全部匹配（权重 1）；RoMa 为稠密对应（光学侧每 4 个 864 网格像素取一个，权重 certainty）
        corr = dense if dense is not None else (M[:, :2] + 0.5, M[:, 2:4] + 0.5, np.ones(len(M)))
        return A, pts.astype(np.float32), imp, len(M), corr

    def _roma(self, opt, sar):
        import cv2

        from baselines.adapters.base import to_original

        ad, torch = self.ad, self.torch
        p0, g0 = ad._pil(opt)
        p1, g1 = ad._pil(sar)
        with torch.no_grad():
            warp, cert = ad.model.match(p0, p1, batched=False, device=ad.device)
            m, conf = ad.model.sample(warp, cert)
            k0, k1 = ad.model.to_pixel_coordinates(m, g0[2], g0[3], g1[2], g1[3])
        kp0 = to_original(k0.float().cpu().numpy() - 0.5, *g0)
        kp1 = to_original(k1.float().cpu().numpy() - 0.5, *g1)
        c = cert.float().cpu().numpy()
        w = c.shape[1] // 2
        imp = None
        if self.record:
            imp = np.stack([cv2.resize(c[:, :w], (64, 64), interpolation=cv2.INTER_AREA),
                            cv2.resize(c[:, w:], (64, 64), interpolation=cv2.INTER_AREA)]).astype(np.float16)
        # 稠密对应：光学侧像素 → SAR，归一化坐标 x_n 换成原网格角点约定 u = 256·(x_n + 1)
        wa = warp[:, :w][::4, ::4].float().cpu().numpy().reshape(-1, 4)
        dense = (HW / 2 * (wa[:, :2] + 1), HW / 2 * (wa[:, 2:] + 1), c[:, :w][::4, ::4].reshape(-1).astype(np.float64))
        return kp0, kp1, conf.float().cpu().numpy(), imp, dense


def refit(corr, A0, it=3, thr=3.0):
    """以 A0 为起点：取残差 < thr 的对应做（加权）最小二乘，迭代 it 次。不随机抽样，结果随输入连续变化。
    遮挡实验里用它代替 RANSAC：RANSAC 在相近的内点集合之间跳动，微小噪声就能让仿射变 1–3 px，盖过遮挡的效果。"""
    src, dst, w = corr
    A = np.asarray(A0, np.float64)
    for _ in range(it):
        m = (np.linalg.norm(apply(A, src) - dst, axis=1) < thr) & (w > 0)
        if m.sum() < 3:
            return None
        sw = np.sqrt(w[m])[:, None]
        A = np.linalg.lstsq(np.c_[src[m], np.ones(m.sum())] * sw, dst[m] * sw, rcond=None)[0].T
    return A


# ---- 遮挡实验的 60 对 ----

def offsets(name: str, pairs) -> dict:
    """模型已有预测相对标注的整体偏移（各检查点误差向量的平均，SAR 坐标系，px）；失败的对不在结果里。"""
    P, out = ref_preds(name), {}
    for p in pairs:
        if P.get(p) is not None:
            o, s = checkpoints(p)
            out[p] = (apply(P[p], o) - s).mean(0)
    return out


def pick_occlusion_pairs(n_per=20, seed=0) -> dict:
    """按两个 zero-shot 模型的整体偏移大小（两者模长的平均）三等分，各取 20 对；大偏移里 x 向同号、异号各一半。"""
    pairs = val_pairs()
    oL, oR = offsets("loftr_zs", pairs), offsets("roma_zs", pairs)
    ok = [p for p in pairs if p in oL and p in oR]
    mag = np.array([(np.linalg.norm(oL[p]) + np.linalg.norm(oR[p])) / 2 for p in ok])
    t1, t2 = np.quantile(mag, [1 / 3, 2 / 3])
    rng = np.random.default_rng(seed)
    grp = {"small": [p for p, m in zip(ok, mag) if m <= t1],
           "medium": [p for p, m in zip(ok, mag) if t1 < m <= t2]}
    large = [p for p, m in zip(ok, mag) if m > t2]
    same = [p for p in large if np.sign(oL[p][0]) == np.sign(oR[p][0])]
    diff = [p for p in large if np.sign(oL[p][0]) != np.sign(oR[p][0])]
    sel = {g: sorted(rng.choice(v, n_per, replace=False).tolist()) for g, v in grp.items()}
    k = min(n_per // 2, len(diff))
    pick_d = rng.choice(diff, k, replace=False).tolist() if k else []
    pick_s = rng.choice(same, n_per - k, replace=False).tolist()
    sel["large_same_sign"], sel["large_opposite_sign"] = sorted(pick_s), sorted(pick_d)
    return {"groups": sel, "thresholds_px": [float(t1), float(t2)], "n_candidates": len(ok),
            "n_large_same": len(same), "n_large_opposite": len(diff)}
