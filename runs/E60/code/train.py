"""Run resumable training segments and record the real exposure and optimizer counts."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from baselines.data import Data
from moonlib import inputs
from runs.E60.code import signal
from workbench.tbwriter import ScalarWriter


class MoonTrain(Dataset):
    def __init__(self, root, config):
        self.data = Data(root)
        self.pairs = self.data.pairs("train", labelled_only=False)
        self.maps = tuple(inputs.get(mode, config["input"][mode]) for mode in ("optical", "sar"))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        optical = self.maps[0](self.data.optical("train", pair))
        sar = self.maps[1](self.data.sar("train", pair))
        return pair, optical, sar


class ExtractorBatchWithoutKeypoints(RuntimeError):
    def __init__(self, rows):
        super().__init__("RIPE++ extractor batch has no pair with 16 sampled keypoints on both sides")
        self.rows = rows


def batches(count, batch_size, seed, start, end):
    """Shuffle each full pass and pack a continuous stream into optimizer batches."""
    position = start
    last_epoch = None
    while position < end:
        size = min(batch_size, end - position)
        indices = []
        while len(indices) < size:
            epoch, offset = divmod(position, count)
            if epoch != last_epoch:
                order = np.random.default_rng([seed, epoch]).permutation(count)
                last_epoch = epoch
            take = min(size - len(indices), count - offset)
            indices.extend(order[offset:offset + take].tolist())
            position += take
        yield indices


def atomic_save(value, path):
    pending = path.with_suffix(".pending")
    torch.save(value, pending)
    pending.replace(path)


def trim_log(path, pairs):
    if not path.exists():
        return
    pending = path.with_suffix(".pending")
    with path.open(encoding="utf-8") as source, pending.open("w", encoding="utf-8") as target:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["pairs_seen"] > pairs:
                break
            target.write(line)
    pending.replace(path)


def occupancy():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdigit():
        return False
    result = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
                            text=True, capture_output=True, check=True, timeout=8)
    return len(result.stdout.split()) > 1


def extractor_batch(adapter, data, cfg, ramp, bf16):
    from lightglue.utils import ImagePreprocessor

    prepare = ImagePreprocessor(**adapter.extractor.preprocess_conf)
    terms, entropies, rows = [], [], []
    for pair, optical, sar in data:
        features, grids = [], []
        for image in (optical, sar):
            tensor, grid = adapter._tensor(image)
            tensor, _ = prepare(tensor)
            features.append(signal.sampled_features(adapter.extractor, tensor, bf16))
            grids.append((*grid[:2], *tensor.shape[-2:]))
        losses, entropy, row = signal.extractor_pair(*features, *grids, cfg, ramp)
        rows.append(dict(row, pair=pair))
        entropies.append(entropy)
        if losses is not None:
            terms.append(losses)

    if not terms:
        raise ExtractorBatchWithoutKeypoints(rows)
    policy, keypoints, description = (torch.stack(values).mean() for values in zip(*terms))
    sums0, sums1, counts0, counts1 = zip(*entropies)
    entropy = (torch.stack(sums0).sum() / max(1, sum(counts0)) +
               torch.stack(sums1).sum() / max(1, sum(counts1))) / 2
    return {"policy": policy, "keypoint_penalty": keypoints, "descriptor": description * cfg["descriptor_weight"],
            "entropy": entropy * cfg["entropy_weight"]}, rows


def matcher_sample(adapter, data, cfg, ramp):
    pair, optical, sar = data
    features, grids = [], []
    with torch.no_grad():
        for image in (optical, sar):
            tensor, grid = adapter._tensor(image)
            features.append(adapter.extractor.extract(tensor))
            grids.append(grid)
    terms, row = signal.matcher_pair(adapter.matcher, *features, *grids, cfg, ramp)
    return dict(zip(("policy", "dustbin_penalty", "confidence"), terms)), dict(row, pair=pair)


def batch_step(stage, adapter, data, cfg, ramp, divisor, bf16):
    if stage == "extractor_final":
        terms, rows = extractor_batch(adapter, data, cfg, ramp, bf16)
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite extractor loss")
        (loss / divisor).backward()
        return {name: float(value.detach()) for name, value in terms.items()}, rows

    sums, rows = {}, []
    for sample in data:
        terms, row = matcher_sample(adapter, sample, cfg, ramp)
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite matcher loss")
        (loss / (divisor * len(data))).backward()
        for name, value in terms.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) / len(data)
        rows.append(row)
    return sums, rows


def evidence(model, names):
    result = {}
    params = dict(model.named_parameters())
    for name in names:
        parameter = params[name]
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().max() > 0:
            raise RuntimeError(f"Missing finite nonzero gradient in {name}")
        result[name] = (float(gradient.abs().max()), parameter.detach().clone())
    return result


def snapshots(count, options):
    width = options["batch_size"] * options["accumulation"]
    return {math.ceil(n * count / width): n for n in options["observation_passes"]}


def train(stage, adapter, run, data_root, baseline, cfg, identity, *, smoke, seconds, stop_requested):
    options, settings = cfg["train"], cfg["optim"]
    extractor = stage == "extractor_final"
    model = adapter.extractor if extractor else adapter.matcher
    frozen = None if extractor else {k: x.cpu().clone() for k, x in adapter.extractor.state_dict().items()}
    if extractor:
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
        probes = [f"{head}.weight" for head in ("conv4b", "convPb", "convDb")]
    else:
        adapter.extractor.eval().requires_grad_(False)
        for layer in model.transformers:
            layer.self_attn.inner_attn.enable_flash = cfg["signal"]["flash"]
            if not cfg["signal"]["flash"]:
                layer.cross_attn.flash = None
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
        probes = ["posenc.Wr.weight", "log_assignment.0.matchability.weight", "log_assignment.8.final_proj.weight",
                  "token_confidence.0.token.0.weight"]
    model.train()
    torch.manual_seed(options["seed"])
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    bf16 = extractor and torch.cuda.get_device_capability()[0] >= 8
    accumulation = options["accumulation"]
    batch_size = options["batch_size"]
    dataset = MoonTrain(data_root, baseline)
    target = len(dataset) * options["epochs"]
    maximum = batch_size * accumulation * options.get("preflight_updates", 1) if smoke else target
    if not len(dataset):
        raise RuntimeError("No Train pairs")
    observations = snapshots(len(dataset), options)
    directory = run / "ckpt" / (f"{stage}_preflight" if smoke else stage)
    directory.mkdir(parents=True, exist_ok=True)
    resume = directory / "resume.pt"
    seen = microbatches = updates = 0
    if not smoke and resume.is_file():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["config"] != cfg or saved["identity"] != identity or saved["precision"] != ("bf16" if bf16 else "fp32"):
            raise RuntimeError("Saved training uses different inputs, settings or precision")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        seen, microbatches, updates = (saved[key] for key in ("pairs_seen", "microbatches", "updates"))
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    sampler = batches(len(dataset), batch_size, options["seed"], seen, maximum)
    workers = options["workers"]
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, collate_fn=list,
                        prefetch_factor=1 if workers else None, generator=torch.Generator().manual_seed(0))
    log_path = directory / "log.jsonl"
    trim_log(log_path, seen)
    start = time.monotonic()
    next_occupancy = start + 10
    start_seen = seen
    reason = "complete"
    probe_result = {}
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)

    def stop_state():
        nonlocal next_occupancy
        now = time.monotonic()
        if stop_requested.is_set():
            return "withdrawal_signal"
        if now - start >= seconds:
            return "segment_budget"
        if now >= next_occupancy:
            next_occupancy = now + 10
            if occupancy():
                return "shared_gpu"
        return None

    with log_path.open("a", encoding="utf-8", buffering=1) as log, ScalarWriter(run / "tb" / directory.name / "scalars") as tb:
        group_stop = None
        for data in loader:
            if microbatches % accumulation == 0:
                optimizer_batches = min(accumulation, math.ceil((maximum - seen) / batch_size))
            if extractor:
                progress = microbatches / max(1, math.ceil(target / batch_size) - 1)
                optimizer.param_groups[0]["lr"] = settings["lr"] + (settings["final_lr"] - settings["lr"]) * progress
            ramp = min(1.0, seen / (target * cfg["signal"]["ramp_fraction"]))
            try:
                terms, rows = batch_step(stage, adapter, data, cfg["signal"], ramp, optimizer_batches, bf16)
            except ExtractorBatchWithoutKeypoints as exc:
                log.write(json.dumps({"pairs_seen": seen + len(data), "microbatches": microbatches + 1,
                                      "updates": updates, "pairs": exc.rows, "failure": str(exc)}) + "\n")
                raise
            seen += len(data)
            microbatches += 1
            row = {"pairs_seen": seen, "microbatches": microbatches, "updates": updates, "pairs": rows,
                   "loss": sum(terms.values()), "lr": optimizer.param_groups[0]["lr"], **terms}
            group_stop = group_stop or stop_state()
            if microbatches % accumulation == 0 or seen == maximum:
                expected = evidence(model, probes) if smoke else {}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                row.update(updates=updates, grad_norm=float(norm))
                for name, (grad, previous) in expected.items():
                    delta = float((dict(model.named_parameters())[name].detach() - previous).abs().max())
                    if not delta > 0:
                        raise RuntimeError(f"Optimizer did not update {name}")
                    probe_result[name] = {"gradient_max": grad, "parameter_change_max": delta}

                if not smoke and updates in observations:
                    atomic_save(model.state_dict(), directory / f"pass_{observations[updates]}.pt")

                log.write(json.dumps(row, allow_nan=False) + "\n")
                if not smoke and (group_stop or seen == maximum or updates in observations or
                                  updates % options["checkpoint_every"] == 0):
                    atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg,
                                 "identity": identity, "precision": "bf16" if bf16 else "fp32",
                                 "pairs_seen": seen, "microbatches": microbatches, "updates": updates,
                                 "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, resume)
                if group_stop and seen < maximum:
                    reason = group_stop
                    break
            else:
                log.write(json.dumps(row, allow_nan=False) + "\n")

            if smoke or microbatches % 10 == 0 or seen == maximum:
                for name, value in {**terms, "loss": row["loss"], "lr": row["lr"]}.items():
                    tb.add(f"train/{name}", value, microbatches)
                tb.add("train/candidates_mean", sum(r["candidates"] for r in rows) / len(rows), microbatches)
                tb.add("train/inliers_mean", sum(r["inliers"] for r in rows) / len(rows), microbatches)
                tb.add("train/affine_fit_rate", sum(r["geometry"] == "ok" for r in rows) / len(rows), microbatches)
                if extractor:
                    tb.add("train/descriptor_fallbacks", sum(r["fallback"] is True for r in rows), microbatches)
                else:
                    tb.add("train/positive_geometry_rate", sum(r["inliers"] > 0 for r in rows) / len(rows), microbatches)
                print(f"{stage} pairs={seen}/{maximum} batches={microbatches} updates={updates} "
                      f"loss={row['loss']:.6g} pairs/s={(seen - start_seen) / max(.01, time.monotonic() - start):.3f}", flush=True)

    if frozen is not None:
        if any(p.grad is not None for p in adapter.extractor.parameters()) or any(
                not torch.equal(frozen[k], tensor.cpu()) for k, tensor in adapter.extractor.state_dict().items()):
            raise RuntimeError("Frozen SuperPoint changed during matcher training")
    if smoke and (not probe_result or seen != maximum):
        raise RuntimeError("Training preflight did not complete a full optimizer update")
    return {"pairs_seen": seen, "train_pairs": len(dataset), "microbatches": microbatches, "updates": updates,
            "complete": seen == maximum, "reason": reason, "gradients_and_updates": probe_result,
            "extractor_frozen": frozen is not None, "precision": "bf16" if bf16 else "fp32",
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "elapsed_seconds": time.monotonic() - start}
