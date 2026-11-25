"""Replay the first optimizer update and compare sampled counts on the same next patches."""
from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace

import cv2
import torch

from runs.E60.code import signal
from runs.E60.code.run import BASE, BASELINE, ROOT, RUN, load_model, model_files, read_config
from runs.E60.code.train import MoonTrain, batch_step, batches
from workbench import launch


def check_counts(adapter, examples, bf16):
    from lightglue.utils import ImagePreprocessor

    prepare = ImagePreprocessor(**adapter.extractor.preprocess_conf)
    rows = []
    with torch.no_grad():
        for pair, optical, sar in examples:
            counts = []
            for image in (optical, sar):
                pixels, _ = adapter._tensor(image)
                pixels, _ = prepare(pixels)
                _, _, selected, _, _ = signal.sampled_features(adapter.extractor, pixels, bf16)
                counts.append(int(selected.sum()))
            rows.append({"pair": pair, "keypoints": counts})
    return rows


def replay(adapter, cfg, dataset):
    options, settings = cfg["train"], cfg["optim"]
    model = adapter.extractor.train()
    torch.manual_seed(options["seed"])
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    initial = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    target = len(dataset) * options["epochs"]
    width = options["batch_size"]
    accumulation = options["accumulation"]
    stop = width * (accumulation + 5)
    stream = batches(len(dataset), width, options["seed"], 0, stop)
    optimizer.zero_grad(set_to_none=True)

    first_run_log = RUN / "ckpt/extractor_final/log.jsonl"
    recorded = [json.loads(line) for line in first_run_log.read_text(encoding="utf-8").splitlines()][:accumulation]
    losses = []
    for batch in range(accumulation):
        samples = [dataset[i] for i in next(stream)]
        ramp = min(1.0, batch * width / (target * cfg["signal"]["ramp_fraction"]))
        progress = batch / (math.ceil(target / width) - 1)
        optimizer.param_groups[0]["lr"] = settings["lr"] + (settings["final_lr"] - settings["lr"]) * progress
        terms, rows = batch_step("extractor_final", adapter, samples, cfg["signal"], ramp, accumulation, bf16)
        loss = sum(terms.values())
        if rows != recorded[batch]["pairs"]:
            # Rows serialized to JSON have a list in place of the keypoint tuple.
            if json.loads(json.dumps(rows)) != recorded[batch]["pairs"]:
                raise RuntimeError(f"Replay selection differs in microbatch {batch + 1}")
        if not math.isclose(loss, recorded[batch]["loss"], rel_tol=1e-5):
            raise RuntimeError(f"Replay loss differs in microbatch {batch + 1}")
        losses.append(loss)

    torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
    optimizer.step()
    following = [[dataset[i] for i in indices] for indices in stream]
    cuda_state = torch.cuda.get_rng_state()
    after = [check_counts(adapter, batch, bf16) for batch in following]
    model.load_state_dict(initial, strict=True)
    torch.cuda.set_rng_state(cuda_state)
    before = [check_counts(adapter, batch, bf16) for batch in following]
    return {"completed_optimizer_updates": 1, "replayed_microbatch_losses": losses,
            "minimum_keypoints_per_side": cfg["signal"]["min_keypoints"],
            "before_update": before, "after_update": after}


def main():
    data_root, weights_root = BASE / "dataset/Moon", BASE / "weights"
    baseline = json.loads((ROOT / BASELINE).read_text(encoding="utf-8"))
    path, cfg = read_config("extractor_final")
    repo, files, hashes = model_files(SimpleNamespace(weights=weights_root), baseline)
    name = "first_update_diagnosis"
    launch.begin(RUN, name, [sys.executable, "-m", "runs.E60.code.diagnose_first_update"],
                 config_file=path, args=cfg, config=baseline, hashes=hashes, data_root=str(data_root))
    try:
        result = replay(load_model(repo, files, baseline), cfg, MoonTrain(data_root, baseline))
        out = RUN / "extra/first_update_diagnosis.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    except BaseException as exc:
        launch.end(RUN, name, "fail", error=str(exc))
        raise
    threshold = result["minimum_keypoints_per_side"]
    for state in ("before_update", "after_update"):
        counts = [sum(min(row["keypoints"]) >= threshold for row in batch) for batch in result[state]]
        print(f"{state}: pairs with at least {threshold} points per side in each batch: {counts}", flush=True)
    launch.end(RUN, name, "ok", replayed_microbatches=4, diagnostic_pairs=30)


if __name__ == "__main__":
    main()
