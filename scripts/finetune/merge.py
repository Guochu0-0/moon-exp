"""不训练、只合并或重估权重，得到可直接评测的 RoMa ckpt（「第一批实验」#120）。产物格式同训练（{'model': state_dict}），
baselines.match --weights 直接能读。

    python scripts/finetune/merge.py avg   <out.pt> <ckpt> <ckpt> ...            # 逐参数均匀平均（Model soups 2203.05482）
    python scripts/finetune/merge.py wise  <out.pt> --zero <底座> --ft <ckpt> --alpha 0.5
                                                                                 # (1−α)·底座 + α·微调（WiSE-FT 2109.01903）
    python scripts/finetune/merge.py adabn <out.pt> --init <ckpt> [--n 2000]     # 在 Train 上重估 VGG 的 BN 统计（AdaBN 1603.04779）

- 浮点张量（参数与 BN 的 running mean / var）参与平均或插值；整数张量（num_batches_tracked）取第一个 / 微调的。
- wise 的底座权重可以多出 DINOv2 等微调 ckpt 里没有的键：只对两者共有的键插值，输出的键与微调 ckpt 相同。
- adabn：只把 VGG（encoder.cnn）里的 BN 切到训练态、清零统计、momentum=None（累计平均），在 Train 均匀取 n 对，
  光学与 SAR 各按训练同口径缩放到 560 后过一遍 VGG；其余模块和参数不变。要 GPU 与 loftr 环境。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _sd(path):
    d = torch.load(path, map_location="cpu")
    return d["model"] if isinstance(d, dict) and "model" in d else d


def average(sds):
    out = {}
    for k, v in sds[0].items():
        out[k] = torch.stack([sd[k].float() for sd in sds]).mean(0).to(v.dtype) if v.is_floating_point() else v.clone()
    return out


def wise(zero, ft, alpha):
    out = {}
    for k, v in ft.items():
        if v.is_floating_point() and k in zero:
            if zero[k].shape != v.shape:
                raise ValueError(f"{k}: 形状不同 {tuple(zero[k].shape)} vs {tuple(v.shape)}")
            out[k] = ((1 - alpha) * zero[k].float() + alpha * v.float()).to(v.dtype)
        else:
            out[k] = v.clone()
    return out, sorted(k for k in ft if k not in zero)


def adabn(init, n, data, weights_root, device="cuda"):
    from finetune import config as C
    from finetune.data import PairSet
    from finetune.models import MODELS

    mod = MODELS["roma"]
    os.environ.setdefault("TORCH_HOME", mod.TORCH_HOME)
    cfg = C.expand({"model": {"name": "roma", "init": str(init)}})
    model = mod.Model(cfg, weights_root, device=device)
    cnn = model.model.encoder.cnn
    bns = [m for m in cnn.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise RuntimeError("encoder.cnn 里没有 BN 层")
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    ds = PairSet(data, "train", model.resize, **model.input)
    idx = range(0, len(ds), max(1, len(ds) // n))[:n]
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        for i, j in enumerate(idx):
            b = ds[j]
            for x in (b["image0"], b["image1"]):
                x = x[None].to(device).expand(-1, 3, -1, -1)
                cnn((x - mean.to(x)) / std.to(x))
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(idx)}", flush=True)
    for m in bns:
        m.eval()
    return model.state_dict()["model"], len(bns), len(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("avg"); a.add_argument("out"); a.add_argument("ckpts", nargs="+")
    w = sub.add_parser("wise"); w.add_argument("out"); w.add_argument("--zero", required=True)
    w.add_argument("--ft", required=True); w.add_argument("--alpha", type=float, required=True)
    b = sub.add_parser("adabn"); b.add_argument("out"); b.add_argument("--init", required=True)
    b.add_argument("--n", type=int, default=2000)
    b.add_argument("--data", default=os.environ.get("MOON_DATA"))
    b.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "avg":
        sd = average([_sd(p) for p in args.ckpts])
        print(f"平均 {len(args.ckpts)} 个 ckpt，{len(sd)} 个键")
    elif args.cmd == "wise":
        sd, extra = wise(_sd(args.zero), _sd(args.ft), args.alpha)
        print(f"α={args.alpha}，{len(sd)} 个键；底座里没有、照搬微调值的键 {len(extra)} 个：{extra[:5]}")
    else:
        sd, nb, ni = adabn(args.init, args.n, args.data, args.weights_root)
        print(f"重估 {nb} 个 BN 层，用 Train {ni} 对（光学、SAR 各一张）")
    torch.save({"model": sd}, out)


if __name__ == "__main__":
    main()
