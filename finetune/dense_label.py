"""稠密自标签的老师 warp（#122；SMURF 2105.07014 §3.2.1 用模型自己在干净输入上的最终输出作目标）。

用老师 ckpt 按评测口径（RomatchAdapter.match：560 → 864、symmetric、attenuate_cert）在 Train 上推理，存每对两个方向的
落点与 certainty，供 [pseudo] dense 使用：

    python -m finetune.dense_label runs/T7/ckpt/main__ref/ckpt_8000.pt --labels <伪标签 jsonl> --top 0.5 \
        --out runs/T11/raw/teacher

每对一个 <out>/<ROI>/<stem>.npy，float16 (2, 3, S, S)：[0] 光学上每个像素 → SAR 的落点 (x, y)（RoMa 归一化坐标）与
certainty，[1] SAR → 光学。864 网格按面积平均缩到 S（默认 288）。certainty 是 match() 的输出（sigmoid 后，落点出界的记 0）。
<out>/meta.json 记老师、commit、参数。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines.data import Data
from baselines.match import git_head
from moonlib import inputs
from workbench.launch import require_clean

from .label import load_labels

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/baselines/anymatch_roma__minmax.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="老师 ckpt（{'model': ...}）")
    ap.add_argument("--labels", required=True, help="只给这份伪标签里 keep 的对打（与训练用的同一份）")
    ap.add_argument("--top", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    require_clean(REPO)

    from baselines.adapters.romatch import RomatchAdapter

    cfg = json.loads((REPO / CONFIG).read_text(encoding="utf-8"))
    ad = RomatchAdapter(REPO / cfg["repo"], args.weights, device=args.device, weights_key="model",
                        long_side=cfg["params"].get("long_side", 640))
    m_opt, m_sar = inputs.get("optical", cfg["input"]["optical"]), inputs.get("sar", cfg["input"]["sar"])
    labels = load_labels(args.labels, args.top)
    pairs = sorted(q for q, A in labels.items() if A is not None)
    if args.limit:
        pairs = pairs[:args.limit]
    data, out = Data(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(
        {"teacher": str(args.weights), "labels": args.labels, "top": args.top, "size": args.size, "n": len(pairs),
         "commit": git_head(REPO), "config": CONFIG}, ensure_ascii=False, indent=1), encoding="utf-8")
    t0 = time.time()
    for it, q in enumerate(pairs):
        f = out / f"{q}.npy"
        if f.exists():
            continue
        p0, _ = ad._pil(m_opt(data.optical("train", q)).astype(np.float32))
        p1, _ = ad._pil(m_sar(data.sar("train", q)).astype(np.float32))
        with torch.no_grad():
            warp, cert = ad.model.match(p0, p1, batched=False, device=args.device)
        n = warp.shape[0]
        ab = torch.cat([warp[:, :n, 2:4], cert[:, :n, None]], -1)      # 光学 → SAR
        ba = torch.cat([warp[:, n:, 0:2], cert[:, n:, None]], -1)      # SAR → 光学
        x = torch.stack([ab, ba]).permute(0, 3, 1, 2).float()           # (2,3,864,864)
        x = F.interpolate(x, size=(args.size, args.size), mode="area")
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, x.cpu().numpy().astype(np.float16))
        if (it + 1) % 200 == 0:
            print(f"{it + 1}/{len(pairs)} {(time.time() - t0) / 60:.1f}min", flush=True)
    print(f"done: {len(pairs)} pairs")


if __name__ == "__main__":
    main()
