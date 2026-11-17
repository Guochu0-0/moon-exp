"""正式运行前核对真实模型共同增强、BN、可训练范围和梯度；不保存训练权重。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tvf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baselines.match import env_info, seed_all
from finetune.config import load
from finetune.data import PairSet
from finetune.models.roma import Model, _norm
from finetune.optim import Optim
from finetune.parts.pseudo import Term
from finetune.train import Step
from workbench.launch import require_clean


def main():
    require_clean(ROOT)
    seed_all(0)
    cfg = load(ROOT / "runs/E50/configs/main.toml")
    out = ROOT / "runs/E50/ckpt/probe"
    out.mkdir(parents=True, exist_ok=True)
    model = Model(cfg, "/remote-home/xufang/YGC/weights")
    ds = PairSet("/remote-home/xufang/YGC/dataset/Moon", "train", model.resize, limit=4, **model.input)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
    labels = out / "labels.jsonl"
    labels.write_text("".join(json.dumps({"pair": q, "A": [[1, .01, 2], [0, 1, -3]],
                     "keep": True, "n_inliers": 100}) + "\n" for q in ds.pairs))
    term = Term({**cfg["pseudo"], "labels": str(labels)}, model, ds)
    optim = Optim(model.params, cfg["optim"], model.amp, model.groups)
    captures = []
    hook = model.model.register_forward_pre_hook(lambda mod, args: captures.append(args[0]))
    specs = []
    original = model.forward

    def capture(a, b, common=None):
        specs.append((a.detach().clone(), b.detach().clone(), common))
        return original(a, b, common=common)

    model.forward = capture
    bn = [m for m in model.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    before = [m.running_mean.clone() for m in bn]
    t0 = time.monotonic()
    loss, stats, active = term(Step(model, "roma", batch, "cuda", False, np.random.default_rng(0), optim))
    optim.backward(loss)
    optim.scaler.unscale_(optim.opt)
    grad = {"vgg": [], "coarse": [], "refine": []}
    from finetune.models.roma import _part
    for n, p in model.model.named_parameters():
        if p.grad is not None and _part(n) in grad:
            grad[_part(n)].append(p.grad)
    assert active and torch.isfinite(loss)
    finite = {k: all(bool(torch.isfinite(g).all()) for g in gs) for k, gs in grad.items()}
    nonzero = {k: any(bool((g != 0).any()) for g in gs) for k, gs in grad.items()}
    assert all(nonzero.values())
    a, b, transforms = specs[0]
    for key, img in (("im_A", a), ("im_B", b)):
        normalized = _norm(img)
        expected = []
        for im, (dx, dy, flip) in zip(normalized, transforms):
            x = tvf.affine(im, angle=0., translate=[dx, dy], scale=1., shear=[0., 0.])
            expected.append(x.flip(-1) if flip else x)
        assert torch.equal(captures[0][key], torch.stack(expected))
    hook.remove()
    assert all(m.training for m in bn)
    assert any(not torch.equal(m.running_mean, old) for m, old in zip(bn, before))
    dino = model.model.encoder.dinov2_vitl14[0]
    # DINOv2 不在网络注册参数内，上游在 no_grad 下提特征。
    assert not any(p.grad is not None for p in dino.parameters())
    report = {"env": env_info(), "config": cfg, "batch": batch["pair"], "loss": float(loss), "stats": stats,
              "finite_grad": finite, "nonzero_grad": nonzero, "bn_count": len(bn),
              "lr_groups": [cfg["optim"]["lr"] * g["lr_scale"] for g in model.groups],
              "sec": time.monotonic() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
              "initial_scale": optim.scaler.get_scale(), "common_specs": transforms}
    (ROOT / "runs/E50/extra").mkdir(exist_ok=True)
    (ROOT / "runs/E50/extra/probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
