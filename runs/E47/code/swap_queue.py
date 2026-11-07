"""E47 只用（#255）：反向推理（交换输入，SAR 当第一张图、光学当第二张）的逐存档 Val 评测，供「正反向互逆残差」与
「反向本身的准确度」。在空卡上跑，与 eval_queue.py 并行：

    GPU=<空卡> nohup /opt/envs/loftr/bin/python runs/E47/code/swap_queue.py > runs/E47/ckpt/swap_<host>_g<GPU>.log 2>&1 < /dev/null &

领取顺序：E41 门槛 0.8 的存档（gpfs 主 checkout 里 runs/E41/ckpt/main__screen0.8/ 现存的，清理后只剩第 7907 次），
然后是本实验 main 训练中陆续写出的存档（写完至少 SETTLE 秒、开头 1 MB 不是全零）。靠 mkdir runs/E47/.swapclaims/<名> 分工。
每对图照 E40 code/infer.py 的 swap 视角：同一个 romatch 适配器（AnyMatch-RoMa 配置），swap=True，推理前随机种子重置为 0，
点对按 SAR → 光学 RANSAC 拟合（阈值 3 px，量在光学图上）再求逆，得到「光学 → SAR」的仿射，写成与预测相同格式的
extra/data/swap/<名>.jsonl（pair、A、n_inliers）。标注不参与这里的任何一步。训练写了 train.done 且本实验的存档都评完就退出。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
E41 = Path("/remote-home/xufang/YGC/moon-exp/runs/E41/ckpt/main__screen0.8")
SETTLE = 120


def now():
    f = R / ".swapclaims" / ".now"
    f.touch()
    return f.stat().st_mtime


def zeros(p):
    with open(p, "rb") as f:
        b = f.read(1 << 20)
    return not b.strip(b"\0")


def todo():
    """[(名, ckpt 路径)]，先 E41 后本实验。"""
    out = [(f"E41_step{c.stem[5:]}", c) for c in sorted(E41.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:]))]
    ck = R / "ckpt" / "main"
    out += [(f"main_step{c.stem[5:]}", c) for c in sorted(ck.glob("ckpt_*.pt"), key=lambda p: int(p.stem[5:]))]
    return out


def infer(ckpt, out_f):
    import numpy as np
    import cv2
    from baselines import adapters
    from baselines.data import Data
    from baselines.match import seed_all
    from moonlib import inputs
    from moonlib.ransac import fit_affine

    cfg = json.loads((REPO / "configs/baselines/anymatch_roma__minmax.json").read_text(encoding="utf-8"))
    data = Data(os.environ.get("MOON_DATA", "/remote-home/xufang/YGC/dataset/Moon"))
    pairs = data.pairs("val", labelled_only=True)
    map_opt = inputs.get("optical", cfg["input"]["optical"])
    map_sar = inputs.get("sar", cfg["input"]["sar"])
    seed_all(cfg["seed"])
    m = adapters.load(cfg["adapter"])(repo=REPO / cfg["repo"], weights=Path(ckpt).resolve(), device="cuda", **cfg["params"])
    m.swap = True
    tmp = out_f.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for pair in pairs:
            opt = map_opt(data.optical("val", pair)).astype(np.float32)
            sar = map_sar(data.sar("val", pair)).astype(np.float32)
            seed_all(cfg["seed"])
            kp0, kp1, conf = m.match(opt, sar)
            M = np.c_[np.asarray(kp0).reshape(-1, 2), np.asarray(kp1).reshape(-1, 2), np.asarray(conf).reshape(-1)]
            B, inl, fail = fit_affine(M[:, [2, 3, 0, 1, 4]], 3.0)
            A = None if B is None else np.round(cv2.invertAffineTransform(B), 6).tolist()
            f.write(json.dumps({"pair": pair, "A": A, "n_inliers": int(inl.sum()) if inl is not None else 0}) + "\n")
    tmp.replace(out_f)
    del m


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ["GPU"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    claims = R / ".swapclaims"
    claims.mkdir(exist_ok=True)
    out = R / "extra" / "data" / "swap"
    out.mkdir(parents=True, exist_ok=True)
    while True:
        for name, c in todo():
            if (out / f"{name}.jsonl").exists() or now() - c.stat().st_mtime < SETTLE:
                continue
            if zeros(c):
                print(f"{time.strftime('%H:%M:%S')} ZERO {name}", flush=True)
                continue
            try:
                (claims / name).mkdir()
            except FileExistsError:
                continue
            t = time.time()
            print(f"{time.strftime('%H:%M:%S')} SWAP {name}", flush=True)
            try:
                infer(c, out / f"{name}.jsonl")
                print(f"{time.strftime('%H:%M:%S')} OK {name} {time.time() - t:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{time.strftime('%H:%M:%S')} FAIL {name}: {e!r}", flush=True)
                (claims / name).rmdir()
            break
        else:
            ck = R / "ckpt" / "main"
            if (ck / "train.done").exists() and all((out / f"{n}.jsonl").exists() for n, _ in todo()):
                print("all done", flush=True)
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
