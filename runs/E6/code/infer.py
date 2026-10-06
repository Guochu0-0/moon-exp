"""对一个模型在 Val 全部有标注的对上推理一次，存下全图的对应（「在陨石坑附近拟合仿射是否更准」#114）。

    GPU=1 /opt/envs/loftr/bin/python runs/E6/code/infer.py roma_zs [--limit N]

模型、ckpt、读图与推理配置沿用 E4（runs/E4/code/common.py）。坐标全部是评测口径（原 512 网格、角点原点，像素中心在 c + 0.5）。
产物 runs/E6/raw/<模型>/（不进 git）：
- pairs.json：对的顺序；A.npy：(P,2,3) 模型仿射（与评测同样的采样 + RANSAC 3 px），失败为 nan；
- LoFTR：matches.npz，M__<pair> 为全部匹配 N×4（光学 x,y、SAR x,y），RANSAC 之前；
- RoMa：D.npy (P,256,256,2) float32，光学侧网格点 (2j+1, 2i+1) 处稠密对应的位移（SAR − 光学）；
        C.npy (P,256,256) float16，同一网格上的 certainty。由网络 warp 双线性取样得到，后续再插值到任意位置。
- val.jsonl：匹配数、耗时、与已有预测的仿射差（17×17 网格平均距离）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                                   # runs/E6
REPO = RUN.parents[1]
sys.path.insert(0, str(REPO / "runs" / "E4" / "code"))

from common import HW, MODELS, Runner, affine_change, key, ref_preds, val_pairs  # noqa: E402
from moonlib.ransac import fit_affine  # noqa: E402

from workbench.launch import begin, end  # noqa: E402

G = 256                                             # RoMa 稠密对应的存储网格
GRID_UV = 2 * np.arange(G) + 1.0                    # 网格点的原网格坐标（角点约定）


def roma_dense(r: Runner, opt, sar):
    """一次 match：评测同样的采样 + RANSAC 得模型仿射；warp 与 certainty 在 G×G 网格上双线性取样。"""
    import torch
    import torch.nn.functional as F

    from baselines.adapters.base import to_original

    ad = r.ad
    p0, g0 = ad._pil(opt)
    p1, g1 = ad._pil(sar)
    r.seed_all(r.seed)
    with torch.no_grad():
        warp, cert = ad.model.match(p0, p1, batched=False, device=ad.device)
        m, conf = ad.model.sample(warp, cert)
        k0, k1 = ad.model.to_pixel_coordinates(m, g0[2], g0[3], g1[2], g1[3])
        W = warp.shape[1] // 2                      # symmetric：左半是光学网格上的 光学 → SAR
        xn = torch.tensor(GRID_UV / (HW / 2) - 1, dtype=torch.float32, device=warp.device)
        gy, gx = torch.meshgrid(xn, xn, indexing="ij")
        grid = torch.stack([gx, gy], -1)[None]      # 1×G×G×2，(x, y)
        ab = warp[:, :W, 2:4].permute(2, 0, 1)[None].float()
        q = F.grid_sample(ab, grid, mode="bilinear", align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
        c = F.grid_sample(cert[:, :W][None, None].float(), grid, mode="bilinear",
                          align_corners=False)[0, 0].cpu().numpy()
    kp0 = to_original(k0.float().cpu().numpy() - 0.5, *g0)
    kp1 = to_original(k1.float().cpu().numpy() - 0.5, *g1)
    M = np.c_[kp0, kp1, conf.float().cpu().numpy()].astype(np.float32)
    A, _, _ = fit_affine(M, 3.0)
    sar_uv = HW / 2 * (q.astype(np.float64) + 1)
    U, V = np.meshgrid(GRID_UV, GRID_UV)
    D = (sar_uv - np.stack([U, V], -1)).astype(np.float32)
    return A, D, c.astype(np.float16), len(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    tag = f"infer_{a.model}"
    begin(RUN, tag, ["infer.py", a.model] + (["--limit", str(a.limit)] if a.limit else []),
          entry="runs/E6/code/infer.py", model=MODELS[a.model])
    status = "fail"
    try:
        out = RUN / "raw" / a.model
        out.mkdir(parents=True, exist_ok=True)
        pairs = val_pairs()[: a.limit or None]
        ref = ref_preds(a.model)
        r = Runner(a.model, record=False)
        roma = r.family == "roma"
        As = np.full((len(pairs), 2, 3), np.nan)
        if roma:
            Ds = np.zeros((len(pairs), G, G, 2), np.float32)
            Cs = np.zeros((len(pairs), G, G), np.float16)
        Ms, t0 = {}, time.time()
        with open(out / "val.jsonl", "w", encoding="utf-8") as log:
            for i, p in enumerate(pairs):
                t = time.time()
                opt, sar = r.load_pair(p)
                if roma:
                    A, Ds[i], Cs[i], n = roma_dense(r, opt, sar)
                else:
                    A, _, _, n, corr = r(opt, sar)
                    Ms[f"M__{key(p)}"] = np.c_[corr[0], corr[1]].astype(np.float32)
                if A is not None:
                    As[i] = A
                rec = {"pair": p, "n_matches": n, "sec": round(time.time() - t, 3),
                       "diff_vs_ref_px": affine_change(A, ref.get(p)), "fail": A is None,
                       "ref_fail": ref.get(p) is None}
                log.write(json.dumps(rec) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"{a.model} {i + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
        (out / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
        np.save(out / "A.npy", As)
        if roma:
            np.save(out / "D.npy", Ds)
            np.save(out / "C.npy", Cs)
        else:
            np.savez(out / "matches.npz", **Ms)
        status = "ok"
    finally:
        end(RUN, tag, status)


if __name__ == "__main__":
    main()
