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
    outputs = []
    output_hook = model.model.register_forward_hook(lambda mod, args, out: outputs.append(out))
    specs = []
    original = model.forward

    def capture(a, b, common=None):
        specs.append((a.detach().clone(), b.detach().clone(), common))
        return original(a, b, common=common)

    model.forward = capture
    bn = [m for m in model.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    before = [m.running_mean.clone() for m in bn]
    t0 = time.monotonic()
    from finetune.models.roma import _part
    attempts = []
    rng = np.random.default_rng(0)
    for attempt in range(16):
        captures.clear()
        outputs.clear()
        specs.clear()
        state_before = rng.bit_generator.state
        loss, stats, active = term(Step(model, "roma", batch, "cuda", False, rng, optim))
        optim.backward(loss)
        grad = {"vgg": [], "coarse": [], "refine": []}
        for n, p in model.model.named_parameters():
            if p.grad is not None and _part(n) in grad:
                grad[_part(n)].append(p.grad)
        assert active and torch.isfinite(loss)
        finite = {k: all(bool(torch.isfinite(g).all()) for g in gs) for k, gs in grad.items()}
        nonzero = {k: any(bool((g != 0).any()) for g in gs) for k, gs in grad.items()}
        scale_before = optim.scaler.get_scale()
        optim.step(1)
        attempts.append({"scale": scale_before, "finite": finite, "skipped": optim.skipped})
        if not optim.skipped:
            break
    assert all(finite.values()) and not optim.skipped
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
    output_hook.remove()
    assert all(m.training for m in bn)
    assert any(not torch.equal(m.running_mean, old) for m, old in zip(bn, before))
    dino = model.model.encoder.dinov2_vitl14[0]
    # DINOv2 不在网络注册参数内，上游在 no_grad 下提特征。
    assert not any(p.grad is not None for p in dino.parameters())
    # 对真实前向的输出，使用同一固定仿射目标，核对上游损失和输出梯度。
    from finetune.parts.common import sample, targets
    from finetune.parts.pseudo import affine_norm, roma_loss
    from finetune.geom import compose, inv_affine
    import romatch
    # 环境不装远程日志客户端；原损失只用 wandb.log，诊断中将日志明确替换为空操作。
    import types
    try:
        import wandb
    except ModuleNotFoundError:
        wandb = types.ModuleType("wandb")
        wandb.log = lambda *a, **kw: None
        sys.modules["wandb"] = wandb
    import romatch.losses.robust_loss as upstream
    check_rng = np.random.default_rng()
    check_rng.bit_generator.state = state_before
    sw = check_rng.random(4) < cfg["pseudo"]["swap"]
    _, Gs = sample(check_rng, 4, cfg["common"])
    A = np.array([[1., .01, 2.], [0., 1., -3.]])
    transformed = [compose(compose(G, A), inv_affine(G)) for G in Gs]
    An = torch.tensor(np.stack([affine_norm(inv_affine(x) if s else x) for x, s in zip(transformed, sw)]),
                      dtype=torch.float32, device="cuda")
    Gi = torch.tensor(np.stack([affine_norm(inv_affine(G)) for G in Gs]), dtype=torch.float32, device="cuda")
    gt = targets(An, Gi)
    leaves = {s: {k: v.detach().float().requires_grad_(True) for k, v in c.items() if torch.is_tensor(v)}
              for s, c in outputs[0].items()}
    upstream.get_gt_warp = lambda *a, H=None, W=None, **kw: gt(H, W)
    upstream.wandb.log = lambda *a, **kw: None
    romatch.GLOBAL_STEP = 0
    criterion = upstream.RobustLosses(ce_weight=.01, local_dist={1: 4, 2: 4, 4: 8, 8: 8},
                                     local_largest_scale=8, alpha=.5, c=1e-4)
    up_loss = criterion(leaves, dict.fromkeys(("im_A_depth", "im_B_depth", "T_1to2", "K1", "K2")))
    own_loss, _ = roma_loss(leaves, An, gt=gt)
    variables = [v for c in leaves.values() for v in c.values()]
    up_grad = torch.autograd.grad(up_loss, variables, allow_unused=True, retain_graph=True)
    own_grad = torch.autograd.grad(own_loss, variables, allow_unused=True)
    assert torch.allclose(up_loss, own_loss, rtol=1e-5, atol=1e-6), (up_loss, own_loss)
    differences = []
    for x, y in zip(up_grad, own_grad):
        assert (x is None) == (y is None)
        if x is not None:
            assert torch.allclose(x, y, rtol=1e-4, atol=1e-6)
            differences.append(float((x-y).abs().max()))
    report = {"env": env_info(), "config": cfg, "batch": batch["pair"], "loss": float(loss), "stats": stats,
              "finite_grad": finite, "nonzero_grad": nonzero, "bn_count": len(bn),
              "lr_groups": [cfg["optim"]["lr"] * g["lr_scale"] for g in model.groups],
              "sec": time.monotonic() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
              "final_scale": optim.scaler.get_scale(), "common_specs": transforms, "attempts": attempts,
              "upstream_loss": float(up_loss), "local_loss": float(own_loss),
              "loss_output_grad_max_diff": max(differences)}
    report["resolved_inference"] = getattr(model.adapter, "resolved_params", None)
    report["upstream_wandb_logging"] = "disabled for local loss comparison"
    (ROOT / "runs/E50/extra").mkdir(exist_ok=True)
    (ROOT / "runs/E50/extra/probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
