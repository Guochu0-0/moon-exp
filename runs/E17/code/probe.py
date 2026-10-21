"""粗匹配诊断（「【RoMa·RL】诊断：E15 的粗匹配到了什么水平」#187）：只读模型，不训练。

    python runs/E17/code/probe.py <任务>      # 任务见 TASKS；产物写进 runs/E17/raw/<任务>.*

模型（MODELS）：zero-shot AnyMatch-RoMa（zs）；E15 三个正常配置在 Val 上选出的存档（loo_t05 第 2000 步、loo_t03 第 3000 步、
none_t03 第 2000 步，sweep/*/peak.json）；E9 noisy student 第 2 轮的 Val 最好存档（rns_r2 第 6000 步）；RoMa 原版室外权重
（roma_outdoor，只测 MegaDepth）。

coarse_<数据>_<模型>（第 1、2 项）：粗级（尺度 16，560 输入 40×40 格）每格对 64×64 = 4096 个候选位置的 softmax(gm_cls)
（温度 1），只看第一张图 → 第二张图这一个方向（同 E14、E16 的 rloo.coarse）。逐格算真值落点 x（第二张图上的归一化坐标），
只留有真值的格子：
- ring：最高分位置与离 x 最近的候选位置在候选网格上相差几圈（max(|Δ列|, |Δ行|)；0 = 正是最近的那个）；
- p_near：离 x 一个候选间距（归一化 2/64，月球原网格 8 px）以内的候选位置的概率之和；
- rank：离 x 最近的候选位置按概率在 4096 个里排第几（1 = 最高；并列按严格大于计）。
数据：
- moon：Val 全部 652 对有标注的，不加扰动。真值仿射 = 标注点最小二乘拟合（光学 → SAR，同 E16）；落点在 SAR 画面内的格子才算。
- mega：MegaDepth-1500 全部 1500 对（LoFTR 的 megadepth_test_1500 顺序），RGB 直接缩放到 560×560（不保长宽比，同 E14）。
  格子中心换到原图像素，按自带深度图与相机位姿投到第二张图；去掉没有深度、落点出画面、深度前后对不上
  （投影深度与第二张图该处深度的相对差 ≥ 0.2，同 LoFTR 的 warp_kpts）的格子。

gtref_<模型>（第 3 项，只做月球 Val 652 对）：正常评测（baselines.match + baselines.fit + workbench.protocol，同一个适配器、
同一套输入映射、逐对重置种子、560 → 864、symmetric、抽 10000 点、RANSAC 3 px）各跑两遍：
- normal：不改；
- gt：560 那一遍里，romatch 用 cls_to_flow_refine 把 gm_cls 换成粗 warp 的地方，改为真值仿射把每个格子中心映射到的连续位置
  （光学 → SAR 用标注点拟合的仿射，SAR → 光学用它的逆；落点出画面的格子也照换）。粗级 certainty、细化、864 那一遍、抽点、
  RANSAC 与误差都不变。
误差 = 光学检查点经预测仿射映射后到 SAR 检查点的平均距离（原始 px，失败记 ∞），与工作台评测同一个函数。

产物：raw/coarse_*.npz（逐格 pair、ring、p_near、rank）与 .json（逐格去除计数）；raw/gtref_*.csv（逐对）。汇总见 summary.py。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO))

from finetune import config as C                    # noqa: E402
from finetune.data import PairSet                   # noqa: E402
from finetune.geom import inv_affine                # noqa: E402
from finetune.models import roma                    # noqa: E402
from finetune.parts import rloo                     # noqa: E402
from finetune.parts.pseudo import affine_norm, grid  # noqa: E402

MAIN = "/remote-home/xufang/YGC/moon-exp/runs"
ANYMATCH = "configs/baselines/anymatch_roma__minmax.json"
MODELS = {   # 名 → (推理配置, 存档；空 = 推理配置里的权重)
    "zs": (ANYMATCH, ""),
    "loo_t05": (ANYMATCH, f"{MAIN}/E15/ckpt/loo_t05/ckpt_2000.pt"),
    "loo_t03": (ANYMATCH, f"{MAIN}/E15/ckpt/loo_t03/ckpt_3000.pt"),
    "none_t03": (ANYMATCH, f"{MAIN}/E15/ckpt/none_t03/ckpt_2000.pt"),
    "rns_r2": (ANYMATCH, f"{MAIN}/E9/ckpt/rns_r2/ckpt_6000.pt"),
    "roma_outdoor": ("runs/E17/code/roma_outdoor.json", ""),
}
MOON_MODELS = ("zs", "loo_t05", "loo_t03", "none_t03", "rns_r2")
MEGA_MODELS = ("zs", "roma_outdoor")
TASKS = ([f"coarse_moon_{m}" for m in MOON_MODELS] + [f"gtref_{m}" for m in MOON_MODELS]
         + [f"coarse_mega_{m}" for m in MEGA_MODELS])
MEGA_ROOT = Path("/remote-home/xufang/YGC/dataset/megadepth_test_1500")
MEGA_INFO = REPO / "third_party/LoFTR/assets/megadepth_test_1500_scene_info"
DEPTH_TOL = 0.2      # LoFTR warp_kpts 的深度一致性阈值（相对差）
RES = 64             # 候选位置网格边长（4096 = 64²）


# ---------- 真值 ----------

def labels(data, split, pair):
    """(光学 N×2, 原 SAR N×2)，与 workbench/dataset.py checkpoints 同口径（Label 里 y 存为负值）。"""
    a = np.loadtxt(data.path(split, pair, "Label"), dtype=np.float64, ndmin=2)
    a[:, [1, 3]] *= -1
    return a[:, :2], a[:, 2:4]


def lstsq_affine(p, q):
    X = np.c_[p, np.ones(len(p))]
    return np.linalg.lstsq(X, q, rcond=None)[0].T          # 2×3


def amap_t(An, g):
    """归一化仿射 An (2×3 张量) 作用在网格 g (h,w,2) 上。"""
    return torch.einsum("ij,hwj->hwi", An[:, :2], g) + An[:, 2]


def mega_pairs():
    """MegaDepth-1500：按 megadepth_test_1500.txt 的顺序拼起 5 个场景文件的 1500 对（同 E14）。"""
    out = []
    for stem in (MEGA_INFO / "megadepth_test_1500.txt").read_text().split():
        d = np.load(MEGA_INFO / f"{stem}.npz", allow_pickle=True)
        for (i0, i1), *_ in d["pair_infos"]:
            out.append((stem, d, int(i0), int(i1)))
    return out


def mega_view(d, i):
    """(RGB 影像 (1,3,560,560) [0,1], 深度图 (H,W), K, 世界 → 相机 4×4)。影像与深度在同一个 dense 目录下。"""
    import h5py
    from PIL import Image
    dp = MEGA_ROOT / d["depth_paths"][i]
    img = dp.parent.parent / "imgs" / Path(d["image_paths"][i]).name
    im = Image.open(img).convert("RGB")
    with h5py.File(dp, "r") as f:
        depth = np.asarray(f["depth"], np.float64)
    if depth.shape != (im.height, im.width):
        raise ValueError(f"{dp} 深度 {depth.shape} 与影像 {(im.height, im.width)} 尺寸不同")
    x = np.asarray(im.resize((roma.RES, roma.RES), Image.BICUBIC), np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1)[None], depth, np.asarray(d["intrinsics"][i], np.float64), \
        np.asarray(d["poses"][i], np.float64)


def mega_gt(v0, v1, h, w):
    """第一张图 h×w 格的中心 → 第二张图上的真值落点（归一化坐标 (h,w,2)）与有效掩码，以及各去除原因的格数。
    像素约定：整数 = 像素中心；深度取最近像素。"""
    _, D0, K0, P0 = v0
    _, D1, K1, P1 = v1
    H0, W0 = D0.shape
    H1, W1 = D1.shape
    g = grid(h, w, "cpu").numpy().astype(np.float64).reshape(-1, 2)
    u = (g + 1) / 2 * [W0, H0] - 0.5
    ui = np.clip(np.round(u).astype(int), 0, [W0 - 1, H0 - 1])
    z0 = D0[ui[:, 1], ui[:, 0]]
    X0 = (np.linalg.inv(K0) @ np.c_[u, np.ones(len(u))].T) * z0                     # 3×n
    T = P1 @ np.linalg.inv(P0)
    X1 = T[:3, :3] @ X0 + T[:3, 3:]
    p = K1 @ X1
    with np.errstate(divide="ignore", invalid="ignore"):
        u1 = (p[:2] / p[2]).T
    xt = (u1 + 0.5) / [W1, H1] * 2 - 1
    has_d = z0 > 0
    inside = has_d & (X1[2] > 0) & (np.abs(xt) < 1).all(1)
    vi = np.clip(np.round(np.nan_to_num(u1)).astype(int), 0, [W1 - 1, H1 - 1])
    z1 = D1[vi[:, 1], vi[:, 0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        consist = inside & (z1 > 0) & (np.abs(X1[2] - z1) / z1 < DEPTH_TOL)
    cnt = {"cells": int(len(g)), "no_depth": int((~has_d).sum()), "outside": int((has_d & ~inside).sum()),
           "inconsistent": int((inside & ~consist).sum()), "kept": int(consist.sum())}
    return (torch.from_numpy(np.nan_to_num(xt)).float().reshape(h, w, 2), torch.from_numpy(consist).reshape(h, w),
            cnt)


# ---------- 第 1、2 项 ----------

def load_train_model(name, weights_root):
    cfg_path, ckpt = MODELS[name]
    return roma.Model(C.expand({"model": {"name": "roma", "config": cfg_path, "init": ckpt}}), weights_root)


def cell_stats(cls, xt, valid):
    """cls (C,h,w) gm_cls；xt (h,w,2) 真值落点；valid (h,w) → 有效格子的 (ring, p_near, rank)。"""
    C_ = cls.shape[0]
    P = cls.float().softmax(0).reshape(C_, -1).T[valid.reshape(-1)]                  # (n,C)
    x = xt.reshape(-1, 2)[valid.reshape(-1)].to(P)                                   # (n,2)
    near_xy = ((x + 1) * RES / 2 - 0.5).round().clamp(0, RES - 1).long()             # (列, 行)
    near = near_xy[:, 1] * RES + near_xy[:, 0]
    mode = P.argmax(1)
    ring = torch.maximum((mode % RES - near_xy[:, 0]).abs(), (mode // RES - near_xy[:, 1]).abs())
    G, _ = rloo.anchors(C_, P.device)                                                # (C,2)，k = 行·64 + 列
    p_near = (P * (torch.cdist(x, G) <= 2 / RES + 1e-6)).sum(1)
    rank = (P > P.gather(1, near[:, None])).sum(1) + 1
    return ring.cpu().numpy(), p_near.cpu().numpy(), rank.cpu().numpy()


def run_coarse(data_kind, name, a, out):
    model = load_train_model(name, a.weights_root)
    rows = {"pair": [], "ring": [], "p_near": [], "rank": []}
    removed, names, t0 = {}, [], time.time()

    def add(i, cls, xt, valid):
        r, pn, rk = cell_stats(cls[0], xt.to(cls.device), valid.to(cls.device))
        rows["pair"].append(np.full(len(r), i, np.int32))
        rows["ring"].append(r.astype(np.int16))
        rows["p_near"].append(pn.astype(np.float32))
        rows["rank"].append(rk.astype(np.int16))

    with torch.no_grad():
        if data_kind == "moon":
            ds = PairSet(a.data, "val", model.resize, **model.input, seed=0)
            ds.pairs = [q for q in ds.pairs if ds.data.path(ds.src[q], q, "Label").exists()][: a.limit or None]
            for i in range(len(ds)):
                it = ds[i]
                po, qo = labels(ds.data, ds.src[it["pair"]], it["pair"])
                i0, i1 = it["image0"][None].cuda(), it["image1"][None].cuda()
                f1, f2 = rloo.features(model, i0, i1)
                cls, _ = rloo.coarse(model, f1, f2)
                h, w = cls.shape[-2:]
                An = torch.from_numpy(affine_norm(lstsq_affine(po, qo))).float()
                xt = amap_t(An, grid(h, w, "cpu"))
                valid = (xt.abs() < 1).all(-1)
                for k, v in {"cells": h * w, "outside": int((~valid).sum()), "kept": int(valid.sum())}.items():
                    removed[k] = removed.get(k, 0) + v
                add(i, cls, xt, valid)
                names.append(it["pair"])
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(ds)} {time.time() - t0:.0f}s", flush=True)
        else:
            for i, (stem, d, j0, j1) in enumerate(mega_pairs()[: a.limit or None]):
                v0, v1 = mega_view(d, j0), mega_view(d, j1)
                f1, f2 = rloo.features(model, v0[0].cuda(), v1[0].cuda())
                cls, _ = rloo.coarse(model, f1, f2)
                h, w = cls.shape[-2:]
                xt, valid, cnt = mega_gt(v0, v1, h, w)
                for k, v in cnt.items():
                    removed[k] = removed.get(k, 0) + v
                add(i, cls, xt, valid)
                names.append(f"{stem}/{Path(d['image_paths'][j0]).stem}-{Path(d['image_paths'][j1]).stem}")
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/1500 {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out.with_suffix(".npz"), **{k: np.concatenate(v) for k, v in rows.items()})
    meta = {"data": data_kind, "model": name, "config": MODELS[name][0], "ckpt": MODELS[name][1],
            "n_pairs": len(names), "cells": removed, "sec": round(time.time() - t0, 1), "pairs": names}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- 第 3 项 ----------

def run_gtref(name, a, out):
    from baselines.adapters.romatch import RomatchAdapter
    from baselines.data import Data
    from baselines.match import seed_all
    from moonlib import inputs
    from moonlib.ransac import fit_affine
    from workbench import protocol

    cfg_path, ckpt = MODELS[name]
    cfg = json.loads((REPO / cfg_path).read_text(encoding="utf-8"))
    weights = ckpt or str(Path(a.weights_root) / cfg["weights"])
    seed = int(cfg.get("seed", 0))
    seed_all(seed)
    matcher = RomatchAdapter(REPO / cfg["repo"], weights, device="cuda", **cfg.get("params", {}))
    import romatch.models.matcher as MM                 # 适配器已把 repo 放进 sys.path
    orig = MM.cls_to_flow_refine
    state = {"gt": None}

    def patched(cls):
        flow = orig(cls)                                 # (B,h,w,2)；symmetric 时 B = 2：光学 → SAR、SAR → 光学
        if state["gt"] is None:
            return flow
        B, _, h, w = cls.shape
        if B != len(state["gt"]):
            raise RuntimeError(f"gm_cls batch {B} 与真值仿射个数 {len(state['gt'])} 不符")
        g = grid(h, w, cls.device)
        return torch.stack([amap_t(torch.from_numpy(An).to(cls.device).float(), g) for An in state["gt"]]).to(flow.dtype)

    MM.cls_to_flow_refine = patched
    map_opt, map_sar = inputs.get("optical", cfg["input"]["optical"]), inputs.get("sar", cfg["input"]["sar"])
    data = Data(a.data)
    pairs = data.pairs("val")[: a.limit or None]
    rows, t0 = [], time.time()
    for i, pair in enumerate(pairs):
        opt, sar = map_opt(data.optical("val", pair)), map_sar(data.sar("val", pair))
        po, qo = labels(data, "val", pair)
        A_lab = lstsq_affine(po, qo)
        row = {"pair": pair, "n_labels": len(po),
               "label_fit_res": float(np.linalg.norm(po @ A_lab[:, :2].T + A_lab[:, 2] - qo, axis=1).mean())}
        for kind, gt in (("normal", None), ("gt", [affine_norm(A_lab), affine_norm(inv_affine(A_lab))])):
            state["gt"] = gt
            seed_all(seed)
            kp0, kp1, conf = matcher.match(opt, sar)
            M = np.c_[np.asarray(kp0, np.float64).reshape(-1, 2), np.asarray(kp1, np.float64).reshape(-1, 2),
                      np.asarray(conf, np.float64).reshape(-1)].astype(np.float32)
            A, inl, _ = fit_affine(M, 3.0)
            row[f"err_{kind}"] = protocol.pair_error(A, po, qo)
            row[f"n_inl_{kind}"] = 0 if inl is None else int(inl.sum())
        state["gt"] = None
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 对（冒烟测试），产物写到 raw/_smoke_<任务>")
    a = ap.parse_args(argv)
    os.environ.setdefault("TORCH_HOME", roma.TORCH_HOME)
    out = RUN / "raw" / (f"_smoke_{a.task}" if a.limit else a.task)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.task.startswith("gtref_"):
        run_gtref(a.task[len("gtref_"):], a, out)
    else:
        _, kind, name = a.task.split("_", 2)
        run_coarse(kind, name, a, out)


if __name__ == "__main__":
    main()
