"""E46（#254）同模态自配对的准确度：固定一组 Val 图、每对固定一个已知仿射 T3，分别做 SAR 对 SAR、光学对光学，
看各模型在同模态上离满分多远、训练后变了多少。标注不参与。

    python runs/E46/code/selfpair_eval.py --name zeroshot|<名字> [--ckpt <ckpt_*.pt>] --out runs/E46/extra/data/selfpair

图：Val 有标注的 652 对按名字排序后每 3 对取 1 对（218 对）。每对两种模态：X = 光学图（/255）或 SAR（minmax），
即正式评测的同一套输入映射。T3 用训练扰动同一抽法（finetune.augment.sample_T：平移在半径 128 px 的圆盘内均匀、
旋转 ±20°、各向同性缩放 ±20%），随机数只由（模态, pair）决定（crc32 作种子），各模型完全相同；T3 X 在原网格（512）上
warp、图外补 0（同正式评测的扰动档）。推理照 baselines.match：同一个 romatch 适配器（AnyMatch-RoMa 配置），每次推理前
随机种子重置为 0；拟合照 baselines.fit（RANSAC 3 px）。
配准误差：X 上 8×8 个均匀格点（原网格 px，格心）经拟合仿射与经 T3 映射到的位置之间的平均距离，与正式评测的
"检查点经仿射映射后的平均距离"同一种量，只是检查点换成格点、真值换成 T3。拟合失败记 ∞。
每个模型写 <out>/<name>.jsonl：一行一对一种模态（pair、mod、err、n_inliers）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from baselines import adapters  # noqa: E402
from baselines.data import Data  # noqa: E402
from baselines.match import seed_all  # noqa: E402
from finetune.augment import sample_T, warp  # noqa: E402
from moonlib import inputs  # noqa: E402
from moonlib.ransac import fit_affine  # noqa: E402

CFG = json.loads((REPO / "configs/baselines/anymatch_roma__minmax.json").read_text(encoding="utf-8"))
HW = 512
EQ = dict(shift=128.0, rot=20.0, scale=0.2, disk=True)       # 同 E41 [aug]（难版）
RANSAC = 3.0
_g = (np.arange(8) + 0.5) * (HW / 8)
GRID = np.stack(np.meshgrid(_g, _g), -1).reshape(-1, 2)        # 8×8 格心，(x, y)


def pairs(data):
    return sorted(data.pairs("val", labelled_only=True))[::3]


def T_for(mod, pair):
    return sample_T(np.random.default_rng(zlib.crc32(f"E46/{mod}/{pair}".encode())), hw=HW, **EQ)


def grid_error(A, T):
    if A is None:
        return float("inf")
    A, T = np.asarray(A, np.float64), np.asarray(T, np.float64)
    return float(np.linalg.norm(GRID @ (A[:, :2] - T[:, :2]).T + (A[:, 2] - T[:, 2]), axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    a = ap.parse_args()
    data = Data(a.data)
    ps = pairs(data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f_out = out / f"{a.name}.jsonl"
    done = set()
    if f_out.exists():
        done = {(r["pair"], r["mod"]) for r in map(json.loads, f_out.read_text(encoding="utf-8").splitlines()) if r}
    w = Path(a.ckpt).resolve() if a.ckpt else Path(a.weights_root) / CFG["weights"]
    seed_all(CFG["seed"])
    matcher = adapters.load(CFG["adapter"])(repo=REPO / CFG["repo"], weights=w, device="cuda", **CFG["params"])
    maps = {"opt": inputs.get("optical", CFG["input"]["optical"]), "sar": inputs.get("sar", CFG["input"]["sar"])}
    print(f"{a.name}: {len(ps)} pairs, weights {w}", flush=True)
    t0 = time.time()
    with open(f_out, "a", encoding="utf-8", newline="\n") as f:
        for i, pair in enumerate(ps):
            for mod in ("sar", "opt"):
                if (pair, mod) in done:
                    continue
                raw = data.sar("val", pair) if mod == "sar" else data.optical("val", pair)
                x = maps[mod](raw).astype(np.float32)
                T = T_for(mod, pair)
                seed_all(CFG["seed"])
                kp0, kp1, conf = matcher.match(x, warp(x, T))
                M = np.c_[np.asarray(kp0).reshape(-1, 2), np.asarray(kp1).reshape(-1, 2), np.asarray(conf).reshape(-1)]
                A, inl, _ = fit_affine(M, RANSAC)
                f.write(json.dumps({"pair": pair, "mod": mod, "err": round(grid_error(A, T), 4),
                                    "n_inliers": int(inl.sum()) if inl is not None else 0}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}  {time.time() - t0:.0f}s", flush=True)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
