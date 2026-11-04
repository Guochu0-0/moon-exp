"""E40（#242）：对 Val 652 对有标注的图，每对推理 18 次（原输入、交换输入、8 个已知扰动、8 个随机遮挡），
每次各自 RANSAC 拟合仿射并换算回「原光学 → 原 SAR」坐标，写 raw/rows_s<k>.jsonl（不进 git，由 fit.py 汇总）。
标注不参与这里的任何一步。

    python runs/E40/code/infer.py --ckpt <E37 all 的 ckpt_7907.pt> --shard 0/2 --out runs/E40/raw

推理照 baselines.match：同一个 romatch 适配器（AnyMatch-RoMa 配置，SAR minmax、光学 /255），每次推理前把随机种子
重置为 0；拟合照 baselines.fit（moonlib.ransac.fit_affine，阈值 3 px）。所以 orig 一项与 E37 `all` 的正式预测相同。

各视角（view）的做法：
- orig：原输入。
- swap：交换光学与 SAR 再推理（适配器 swap=True，输出仍按（光学, SAR）顺序），按 SAR → 光学拟合（阈值量在光学图上），再求逆。
- eq0–7：SAR 按已知仿射 T warp（图外补 0，同 baselines.match --perturb），T 用训练时难版扰动的抽法
  （finetune.augment.sample_T：平移在半径 128 px 的圆盘内均匀、旋转 ±20°、各向同性缩放 ±20%），
  拟合得到 光学 → 扰动后 SAR 的 A'，换回原坐标 A = T⁻¹∘A'。
- oc0–7：在光学图上取一块边长 229 px 的正方形（约 20% 面积），左上角在图内均匀抽；SAR 上遮住同一地面位置，
  即这块正方形经本对 orig 仿射映射到 SAR 上的四边形（orig 失败时用恒等映射）。被遮处填高斯噪声，均值、标准差取
  这张图自己的全图均值、标准差，截到 [0, 1]。

T、遮挡位置、噪声的随机数只由 (视角, pair) 决定（crc32 作种子），每对固定、与分片无关。

--mode resample（抽样噪声对照，写 raw/resample_s<k>.jsonl）：原输入不变，推理前把随机种子分别设为 1–8，各推理一次（rs1–rs8）；
网络的前向计算相同，变的只是从稠密对应里抽哪 10000 对点（以及因此 RANSAC 拿到的点）。另用种子 0 重跑一次（rs0），
只用来核对与 constraints 模式的 orig 是否相同。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import cv2
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
N_EQ = N_OC = N_RS = 8
EQ = dict(shift=128.0, rot=20.0, scale=0.2, disk=True)       # 同 E37 [aug]（难版，同 E9）
OC_SIDE = int(round(np.sqrt(0.2) * HW))                        # 229 px，面积约 20%
RANSAC = 3.0


def rng_for(view, pair):
    return np.random.default_rng(zlib.crc32(f"E40/{view}/{pair}".encode()))


def h3(A):
    return np.vstack([np.asarray(A, np.float64), [0.0, 0.0, 1.0]])


def occlude(img, poly, rng):
    """poly：角点约定的四边形顶点（4×2）。cv2 的像素 i 在角点约定里是 i + 0.5，所以减 0.5 后按 1/16 px 精度填。"""
    mask = np.zeros(img.shape, np.uint8)
    pts = np.round((np.asarray(poly, np.float64) - 0.5) * 16).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=4)
    noise = np.clip(rng.normal(img.mean(), img.std(), img.shape), 0.0, 1.0).astype(np.float32)
    out = img.copy()
    out[mask > 0] = noise[mask > 0]
    return out, float(mask.mean())


def fit(M, inverse=False):
    """返回 (A 光学 → SAR, n_inliers, fail)。inverse：点对按 SAR → 光学拟合后求逆。"""
    if inverse:
        A, inl, fail = fit_affine(M[:, [2, 3, 0, 1, 4]], RANSAC)
        if A is not None:
            A = cv2.invertAffineTransform(A)
    else:
        A, inl, fail = fit_affine(M, RANSAC)
    return A, (int(inl.sum()) if inl is not None else 0), fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--mode", default="constraints", choices=("constraints", "resample"))
    a = ap.parse_args()

    k, n = map(int, a.shard.split("/"))
    data = Data(a.data)
    pairs = data.pairs("val", labelled_only=True)[k::n]
    if a.limit:
        pairs = pairs[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / (f"rows_s{k}.jsonl" if a.mode == "constraints" else f"resample_s{k}.jsonl")
    done = set()
    if rows_f.exists():
        done = {json.loads(l)["pair"] for l in rows_f.read_text(encoding="utf-8").splitlines() if l.strip()}

    map_opt = inputs.get("optical", CFG["input"]["optical"])
    map_sar = inputs.get("sar", CFG["input"]["sar"])
    seed_all(CFG["seed"])
    matcher = adapters.load(CFG["adapter"])(repo=REPO / CFG["repo"], weights=Path(a.ckpt).resolve(), device="cuda",
                                            **CFG["params"])
    print(f"shard {k}/{n}: {len(pairs)} pairs, {len(done)} done; lora merged {len(matcher.lora_merged)}", flush=True)

    def run(opt, sar, swap=False, seed=CFG["seed"]):
        matcher.swap = swap
        seed_all(seed)
        kp0, kp1, conf = matcher.match(opt, sar)
        return np.c_[np.asarray(kp0).reshape(-1, 2), np.asarray(kp1).reshape(-1, 2), np.asarray(conf).reshape(-1)]

    t0 = time.time()
    if a.mode == "resample":
        with open(rows_f, "a", encoding="utf-8", newline="\n") as f:
            for i, pair in enumerate(p for p in pairs if p not in done):
                t = time.time()
                opt = map_opt(data.optical("val", pair)).astype(np.float32)
                sar = map_sar(data.sar("val", pair)).astype(np.float32)
                views = {}
                for s in range(N_RS + 1):            # rs0 = 种子 0 重跑一次，只用来核对与 orig 是否逐位相同，不参与平均
                    A, ni, fl = fit(run(opt, sar, seed=s))
                    views[f"rs{s}"] = {"A": None if A is None else np.round(A, 8).tolist(), "n_inliers": ni}
                    if fl:
                        views[f"rs{s}"]["fail"] = fl
                f.write(json.dumps({"pair": pair, "sec": round(time.time() - t, 2), "views": views}) + "\n")
                f.flush()
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}  {time.time() - t0:.0f}s", flush=True)
        print("done", time.time() - t0, flush=True)
        return
    orig_store = {}
    with open(rows_f, "a", encoding="utf-8", newline="\n") as f:
        for i, pair in enumerate(p for p in pairs if p not in done):
            t = time.time()
            opt = map_opt(data.optical("val", pair)).astype(np.float32)
            sar = map_sar(data.sar("val", pair)).astype(np.float32)
            views = {}

            def put(name, A, ninl, fail, **extra):
                views[name] = {"A": None if A is None else np.round(A, 8).tolist(), "n_inliers": ninl, **extra}
                if fail:
                    views[name]["fail"] = fail

            M0 = run(opt, sar)
            orig_store[pair.replace("/", "__")] = M0.astype(np.float32)
            A0, ni, fl = fit(M0)
            put("orig", A0, ni, fl, n_matches=len(M0))

            A, ni, fl = fit(run(opt, sar, swap=True), inverse=True)
            put("swap", A, ni, fl)

            for j in range(N_EQ):
                T = sample_T(rng_for(f"eq{j}", pair), hw=HW, **EQ)
                A1, ni, fl = fit(run(opt, warp(sar, T)))
                A = None if A1 is None else (np.linalg.inv(h3(T)) @ h3(A1))[:2]
                put(f"eq{j}", A, ni, fl, T=np.round(T, 8).tolist())

            Am = A0 if A0 is not None else np.c_[np.eye(2), np.zeros(2)]
            for j in range(N_OC):
                rng = rng_for(f"oc{j}", pair)
                x, y = rng.integers(0, HW - OC_SIDE + 1, 2)
                sq = np.array([[x, y], [x + OC_SIDE, y], [x + OC_SIDE, y + OC_SIDE], [x, y + OC_SIDE]], np.float64)
                o2, fo = occlude(opt, sq, rng)
                s2, fs = occlude(sar, sq @ Am[:, :2].T + Am[:, 2], rng)
                A, ni, fl = fit(run(o2, s2))
                put(f"oc{j}", A, ni, fl, box=[int(x), int(y), OC_SIDE], frac_opt=round(fo, 4), frac_sar=round(fs, 4))

            f.write(json.dumps({"pair": pair, "sec": round(time.time() - t, 2), "views": views}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out / f"orig_matches_s{k}_{len(done)}.npz", **orig_store)    # 续跑时另存一份，不覆盖
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
