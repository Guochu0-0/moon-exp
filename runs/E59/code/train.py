"""Adapt the pinned RIPE training loop to unlabeled lunar optical–SAR pairs."""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset

from baselines.adapters.base import to_original
from baselines.data import Data
from moonlib import inputs
from moonlib.ransac import fit_affine
from workbench.tbwriter import ScalarWriter


class MoonTrain(Dataset):
    def __init__(self, data_path, cfg):
        self.data = Data(data_path)
        self.pairs = self.data.pairs("train", labelled_only=False)
        if len(self.pairs) < 2:
            raise ValueError("RIPE training needs at least two Train patches")
        self.optical_map = inputs.get("optical", cfg["optical"])
        self.sar_map = inputs.get("sar", cfg["sar"])
        self.original_size = cfg["original_size"]
        self.train_size = cfg["train_size"]

    def resize(self, image):
        if image.shape != (self.original_size, self.original_size):
            raise ValueError(f"Unexpected Moon patch size {image.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None]
        return interpolate(tensor, size=(self.train_size, self.train_size), mode="bilinear",
                           align_corners=False, antialias=True)[0]

    def __getitem__(self, indices):
        optical_id, sar_id = (self.pairs[index] for index in indices)
        optical = self.resize(self.optical_map(self.data.optical("train", optical_id)))
        sar = self.resize(self.sar_map(self.data.sar("train", sar_id)))
        return optical, sar, optical_id, sar_id


def pair_batches(count, batch_size, seed, start, stop):
    """Shuffle one positive and one independently sampled negative per optical patch each epoch."""
    batches_per_epoch = 2 * count // batch_size
    batch = start
    while batch < stop:
        epoch, within_epoch = divmod(batch, batches_per_epoch)
        rng = np.random.default_rng([seed, epoch])
        order = rng.permutation(2 * count)
        draws = rng.integers(count - 1, size=count)
        for index in range(within_epoch, min(batches_per_epoch, within_epoch + stop - batch)):
            samples = order[index * batch_size:(index + 1) * batch_size]
            result = []
            for sample in samples:
                optical, negative = divmod(int(sample), 2)
                sar = int(draws[optical])
                result.append((optical, sar + (sar >= optical) if negative else optical))
            yield result
            batch += 1


def affine_mask(src, dst, threshold, original_size, train_size):
    src = to_original(src.float().cpu().numpy(), original_size, original_size, train_size, train_size)
    dst_np = to_original(dst.float().cpu().numpy(), original_size, original_size, train_size, train_size)
    matches = np.column_stack((src, dst_np, np.ones(len(src))))
    matrix, inliers, _ = fit_affine(matches, threshold)
    mask = inliers if inliers is not None else np.zeros(len(matches), dtype=bool)
    return matrix, torch.as_tensor(mask, device=dst.device)


def reward_sum(logp_src, logp_dst, matches, inliers, positive, fp_penalty):
    """Sum the official dense reward times log probability without allocating its grid product."""
    logp_src, logp_dst = logp_src.float(), logp_dst.float()
    sign = 1 if positive else -1
    background = sign * fp_penalty
    selected = matches[inliers]
    dense_sum = len(logp_dst) * logp_src.sum() + len(logp_src) * logp_dst.sum()
    inlier_sum = (logp_src[selected[:, 0]] + logp_dst[selected[:, 1]]).sum()
    return background * dense_sum + (sign - background) * inlier_sum


def features(model, images):
    padding = torch.ones_like(images, dtype=torch.bool)
    xy, logp, selected, valid, _, out = model(images, padding, training=True)
    desc = model.get_descs(out["coarse_descs"], images, xy, *images.shape[-2:])
    return xy, logp, selected & valid, desc


def loss_for_batch(model, matcher, descriptor_loss, batch, signal, alpha):
    optical, sar, optical_ids, sar_ids = batch
    xy0, lp0, selected0, desc0 = features(model, optical.cuda(non_blocking=True))
    xy1, lp1, selected1, desc1 = features(model, sar.cuda(non_blocking=True))
    _, matches, masks, matrices = matcher(xy0, xy1, desc0, desc1, selected0, selected1,
                                         signal["inlier_threshold_px"])
    # All square Train patches have entirely valid padding masks.
    policy, keypoints, descriptors = [], [], []
    rows = []
    for index, (src_id, dst_id) in enumerate(zip(optical_ids, sar_ids)):
        positive = src_id == dst_id
        row = {"optical": src_id, "sar": dst_id, "positive": positive,
               "keypoints": [int(selected0[index].sum()), int(selected1[index].sum())]}
        rows.append(row)
        if matches[index] is None:
            row.update(candidates=None, inliers=None, geometry="few_keypoints", descriptor_fallback=None)
            continue
        pairs, mask = matches[index], masks[index]
        candidates, inliers = len(pairs), int(mask.sum())
        geometry = "ok"
        if candidates < signal["min_matches"]:
            geometry = "few_matches"
        elif matrices[index] is None:
            geometry = "fit_failed"
        row.update(candidates=candidates, inliers=inliers, geometry=geometry,
                   descriptor_fallback=inliers < signal["descriptor_fallback_below"])
        src, dst = lp0[index], lp1[index]
        policy.append(reward_sum(src, dst, pairs, mask, positive, signal["false_positive_penalty"] * alpha))
        keypoints.append(src[selected0[index]].float().mean() + dst[selected1[index]].float().mean())
        descriptors.append(descriptor_loss(desc0[index], desc1[index], pairs, mask, positive))

    if not policy:
        return None, rows
    policy_loss = -torch.stack(policy).sum() / (len(policy) * lp0.shape[1] * lp1.shape[1])
    kp_loss = -signal["keypoint_penalty"] * alpha * torch.stack(keypoints).mean()
    desc_loss = torch.stack(descriptors).mean()
    return (policy_loss + kp_loss + signal["descriptor_weight"] * desc_loss,
            policy_loss, kp_loss, desc_loss), rows


def save_atomic(obj, path):
    pending = path.with_suffix(".tmp")
    torch.save(obj, pending)
    pending.replace(path)


def prepare_log(path, completed_batches):
    """Keep only committed training history if a process died between resume snapshots."""
    if not path.exists():
        return
    pending = path.with_suffix(".tmp")
    with path.open(encoding="utf-8") as old, pending.open("w", encoding="utf-8") as kept:
        for line in old:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break
            if entry["batch"] > completed_batches:
                break
            kept.write(line)
    pending.replace(path)


def train(run_dir, data_path, weights, weight_hash, cfg, *, smoke, seconds):
    from kornia.feature import DescriptorMatcher
    from ripe import vgg_hyper
    from ripe.losses.contrastive_loss import ContrastiveLoss
    from ripe.matcher.concurrent_matcher import ConcurrentMatcher
    from ripe.scheduler.linear_with_plateaus import LinearWithPlateaus
    from ripe.scheduler.linearLR import StepLinearLR

    options, signal, optim = cfg["train"], cfg["signal"], cfg["optim"]
    if options["precision"] != "bf16-mixed" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The official bf16-mixed recipe requires native CUDA BF16")
    torch.set_num_threads(1)
    torch.manual_seed(options["seed"])
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = vgg_hyper(model_path=weights).cuda().train()
    if not any(isinstance(layer, torch.nn.InstanceNorm2d) for layer in model.modules()):
        raise RuntimeError("The released RIPE model must retain InstanceNorm")
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim["lr"], betas=tuple(optim["betas"]),
                                  eps=optim["eps"], weight_decay=optim["weight_decay"])
    alpha = LinearWithPlateaus(0.0, 1.0, options["batches"], rel_length_end_plateu=signal["alpha_end_plateau"])
    accumulation = options["grad_accumulation"]
    checkpoint_dir = run_dir / "ckpt" / ("preflight" if smoke else "main")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume = checkpoint_dir / "resume.pt"
    batch_count, update_count = 0, 0
    if not smoke and resume.exists():
        saved = torch.load(resume, map_location="cpu")
        if saved["config"] != cfg or saved["weights_sha256"] != weight_hash:
            raise RuntimeError("Resume checkpoint belongs to a different training recipe or initial weight")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        batch_count, update_count = saved["batch"], saved["updates"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    scheduler = StepLinearLR(optimizer, batch_count, options["batches"], optim["lr"], optim["final_lr"])
    dataset = MoonTrain(data_path, cfg["input"])
    stop = accumulation * 2 if smoke else options["batches"]
    if batch_count >= stop:
        return {"batches": batch_count, "updates": update_count, "complete": True}
    batches = pair_batches(len(dataset.pairs), options["batch_size"], options["seed"], batch_count, stop)
    workers = options["workers"]
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=workers, pin_memory=True,
                        prefetch_factor=1 if workers else None, generator=torch.Generator().manual_seed(0))
    estimator = partial(affine_mask, original_size=cfg["input"]["original_size"],
                        train_size=cfg["input"]["train_size"])
    matcher = ConcurrentMatcher(DescriptorMatcher("mnn"), estimator, min_num_matches=signal["min_matches"])
    descriptor_loss = ContrastiveLoss(signal["positive_margin"], signal["negative_margin"])
    log_path = checkpoint_dir / "log.jsonl"
    prepare_log(log_path, batch_count)
    started = time.monotonic()
    first_batch = batch_count
    status = Counter()
    check_update = smoke
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log, ScalarWriter(run_dir / "tb" / ("preflight" if smoke else "main") / "scalars") as tb:
            for data in loader:
                scheduler.step()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses, rows = loss_for_batch(model, matcher, descriptor_loss, data, signal, alpha(batch_count))
                batch_count += 1
                row = {"batch": batch_count, "updates": update_count, "lr": scheduler.get_last_lr(), "pairs": rows}
                if losses is None or not torch.isfinite(losses[0]).item():
                    row["failure"] = "no_eligible_pair" if losses is None else "nonfinite_loss"
                    log.write(json.dumps(row) + "\n")
                    raise RuntimeError(f"RIPE batch {batch_count}: {row['failure']}")
                loss, policy, keypoint, descriptor = losses
                (loss / accumulation).backward()
                row.update(loss=float(loss.detach()), policy=float(policy.detach()),
                           keypoint=float(keypoint.detach()), descriptor=float(descriptor.detach()))
                if batch_count % accumulation == 0:
                    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, math.inf, error_if_nonfinite=True))
                    if check_update and grad_norm > 0:
                        probe = max(parameters, key=lambda p: float(p.grad.abs().max()))
                        before = probe.detach().clone()
                        optimizer.step()
                        row["parameter_max_change"] = float((probe.detach() - before).abs().max())
                        if row["parameter_max_change"] == 0:
                            raise RuntimeError("Preflight optimizer step did not change weights")
                        check_update = False
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_count += 1
                    row.update(updates=update_count, grad_norm=grad_norm)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                for pair in rows:
                    status[(pair["positive"], pair["geometry"])] += 1
                if (batch_count - first_batch) % 50 == 0 or smoke:
                    for key in ("loss", "policy", "keypoint", "descriptor", "lr"):
                        tb.add(f"train/{key}", row[key], batch_count)
                    print(f"batch={batch_count} updates={update_count} loss={row['loss']:.6g} "
                          f"sec/batch={(time.monotonic() - started) / (batch_count - first_batch):.3f} "
                          f"peak_GiB={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)
                if batch_count % accumulation:
                    continue
                budget_reached = time.monotonic() - started >= seconds
                if not smoke:
                    val_due = update_count % options["val_every_updates"] == 0
                    if val_due:
                        save_atomic(model.state_dict(), checkpoint_dir / f"ckpt_{update_count}.pt")
                    if val_due or budget_reached or update_count % options["resume_every_updates"] == 0:
                        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                     "batch": batch_count, "updates": update_count, "config": cfg,
                                     "weights_sha256": weight_hash, "cpu_rng": torch.get_rng_state(),
                                     "cuda_rng": torch.cuda.get_rng_state()}, resume)
                if budget_reached:
                    break
    finally:
        matcher.executor.shutdown()
    if smoke and check_update:
        raise RuntimeError("Preflight did not observe a finite, nonzero update")
    return {"batches": batch_count, "updates": update_count, "complete": batch_count == stop,
            "sec_per_batch": (time.monotonic() - started) / (batch_count - first_batch),
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "geometry_counts": {f"{'positive' if positive else 'negative'}/{name}": n
                                for (positive, name), n in sorted(status.items())}}
