"""E48（#256）：擦过的 Val 上的可信度与配准。对 Val 652 对有标注的图，SAR 照训练的擦除方式擦一份（finetune.augment.erase：
1–3 块、共占 10%–25% 面积、填全图均值；随机数只由 pair 决定，crc32("E48/erase/<pair>") 作种子，每个存档用同一份），
用存档推理一次，写 <out>（jsonl，每对一行）：
- A、n_inliers：从稠密对应抽点、RANSAC 拟合的仿射（照 baselines.match / baselines.fit：同一个 romatch 适配器，
  AnyMatch-RoMa 配置，SAR minmax、光学 /255，推理前随机种子重置为 0，抽 10000 对、RANSAC 3 px），err 为配准误差
  （workbench.protocol.pair_error，光学检查点经 A 映射后到 SAR 检查点的平均距离，失败为 null）；
- cert_e、cert_u：光学一侧稠密可信度（romatch match() 输出的 certainty 左半，即 RANSAC 抽点所依据的那张图，
  上采样分辨率网格）在被擦处与未擦处的平均。光学像素是否对到被擦处，用标注（真实仿射：检查点最小二乘拟合）把像素中心
  映到 SAR 上、查擦除掩码；映到 SAR 画面外的像素两边都不算。frac_e 为被擦处占画面内像素的比例。
标注只用于评测（误差与上面的映射），不参与推理。

    python runs/E48/code/erased_val.py --ckpt <ckpt.pt> --out runs/E48/raw/erased/<名字>.jsonl [--shard 0/2]
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
from finetune.augment import erase  # noqa: E402
from moonlib import inputs  # noqa: E402
from moonlib.ransac import fit_affine  # noqa: E402
from workbench.protocol import pair_error  # noqa: E402

CFG = json.loads((REPO / "configs/baselines/anymatch_roma__minmax.json").read_text(encoding="utf-8"))
HW = 512
ERASE = dict(n_max=3, a_min=0.10, a_max=0.25)      # 同训练 [aug] erase 的默认（erase_n、erase_min、erase_max）
RANSAC = 3.0


def checkpoints(data, pair):
    """(光学 N×2, SAR N×2)，像素坐标，y 向下；同 workbench.dataset.Dataset.checkpoints。"""
    a = np.loadtxt(data.path("val", pair, "Label"), dtype=np.float64, ndmin=2)
    a[:, [1, 3]] *= -1
    return a[:, :2], a[:, 2:4]


def gt_affine(po, ps):
    X = np.c_[po, np.ones(len(po))]
    return np.linalg.lstsq(X, ps, rcond=None)[0].T          # 2×3，光学 → SAR


def erased_on_optical(A, mask, h, w):
    """光学 h×w 网格（覆盖整幅 512 图）各像素中心经 A 映到 SAR、查 mask（512×512）→ (被擦, 画面内) 两个 (h, w) bool。"""
    xs = (np.arange(w) + 0.5) * HW / w - 0.5
    ys = (np.arange(h) + 0.5) * HW / h - 0.5
    gx, gy = np.meshgrid(xs, ys)
    u = A[0, 0] * gx + A[0, 1] * gy + A[0, 2]
    v = A[1, 0] * gx + A[1, 1] * gy + A[1, 2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (ui >= 0) & (ui < HW) & (vi >= 0) & (vi < HW)
    e = np.zeros((h, w), bool)
    e[inside] = mask[vi[inside], ui[inside]] > 0
    return e, inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    a = ap.parse_args()

    import torch

    k, n = map(int, a.shard.split("/"))
    data = Data(a.data)
    pairs = data.pairs("val", labelled_only=True)[k::n]
    if a.limit:
        pairs = pairs[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["pair"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    map_opt = inputs.get("optical", CFG["input"]["optical"])
    map_sar = inputs.get("sar", CFG["input"]["sar"])
    seed_all(CFG["seed"])
    mt = adapters.load(CFG["adapter"])(repo=REPO / CFG["repo"], weights=Path(a.ckpt).resolve(), device="cuda",
                                       **CFG["params"])
    print(f"shard {k}/{n}: {len(pairs)} pairs, {len(done)} done; lora merged {len(mt.lora_merged)}", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for i, pair in enumerate(p for p in pairs if p not in done):
            opt = map_opt(data.optical("val", pair)).astype(np.float32)
            sar = map_sar(data.sar("val", pair)).astype(np.float32)
            sar_e, mask = erase(sar, np.random.default_rng(zlib.crc32(f"E48/erase/{pair}".encode())), **ERASE)
            seed_all(CFG["seed"])
            p0, g0 = mt._pil(opt)
            p1, g1 = mt._pil(sar_e)
            with torch.no_grad():                            # 同 RomatchAdapter.match（未交换），另留下稠密可信度
                warp, cert = mt.model.match(p0, p1, batched=False, device=mt.device)
                m, conf = mt.model.sample(warp, cert)
                k0, k1 = mt.model.to_pixel_coordinates(m, g0[2], g0[3], g1[2], g1[3])
            from baselines.adapters.base import to_original
            kp0 = to_original(k0.float().cpu().numpy() - 0.5, *g0)
            kp1 = to_original(k1.float().cpu().numpy() - 0.5, *g1)
            A, inl, fail = fit_affine(np.c_[kp0, kp1, conf.float().cpu().numpy()], RANSAC)
            po, ps = checkpoints(data, pair)
            c = cert.float().cpu().numpy()
            c = c[:, :c.shape[1] // 2]                       # 左半：光学一侧
            e, inside = erased_on_optical(gt_affine(po, ps), mask, *c.shape)
            u = inside & ~e
            row = {"pair": pair, "A": None if A is None else np.round(A, 8).tolist(),
                   "n_inliers": int(inl.sum()) if inl is not None else 0,
                   "err": None if A is None else round(pair_error(A, po, ps), 4),
                   "frac_e": round(float((e & inside).sum() / max(1, inside.sum())), 4),
                   "cert_e": round(float(c[e].mean()), 5) if e.any() else None,
                   "cert_u": round(float(c[u].mean()), 5) if u.any() else None,
                   "frac_sar": round(float(mask.mean()), 4)}
            if fail:
                row["fail"] = fail
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}  {time.time() - t0:.0f}s", flush=True)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
