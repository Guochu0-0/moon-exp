"""在实验服务器上对比同一批次的共享内存与管道传输（只读训练样本，CPU）。

    python runs/E53/code/probe_batch_transport.py /path/to/Moon [--stress-pipe]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from itertools import islice
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import torch

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN.parents[1]))

from finetune.config import load
from finetune.data import PairSet, collate_for_pipe
from finetune.models.roma import Model
from finetune.parts.aug import data_kw
from finetune.trace import Loader


def probe(data, count, rank, pipe):
    config = load(RUN / "configs" / "k0.toml")
    run = config["run"]
    seed, workers, world = run["seed"], run["workers"], run["ddp"]
    resize = partial(Model.resize, SimpleNamespace(long_side=640))
    dataset = PairSet(data, run["split"], resize, sar="minmax", seed=seed, **data_kw(config["aug"]))
    generator = torch.Generator().manual_seed(seed + 1000 * rank)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world, rank=rank, seed=seed)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, generator=generator, num_workers=workers,
                                     drop_last=True, persistent_workers=True, collate_fn=collate_for_pipe if pipe else None)
    base_shm = peak_shm = shutil.disk_usage("/dev/shm").used
    hashes = {}
    size = 0
    t0 = time.monotonic()
    batches = Loader(dl, generator)
    for batch in islice(batches, count):
        if pipe:
            batch = torch.utils.data.default_collate(batch)
        peak_shm = max(peak_shm, shutil.disk_usage("/dev/shm").used)
        for key, value in batch.items():
            if torch.is_tensor(value):
                payload = value.numpy().tobytes()
                size += len(payload)
            else:
                payload = json.dumps(value).encode()
            hashes.setdefault(key, hashlib.sha256()).update(payload)
    return {
        "rank": rank,
        "transport": "pipe" if pipe else "shared_memory",
        "fields": {k: h.hexdigest() for k, h in hashes.items()},
        "sample_count": count,
        "array_bytes": size,
        "seconds": time.monotonic() - t0,
        "sampled_shared_memory_delta_bytes": peak_shm - base_shm,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--stress-pipe", action="store_true", help="另用四个 rank 并行传送")
    args = ap.parse_args()
    for rank in range(4):
        shared = probe(args.data, args.count, rank, False)
        pipe = probe(args.data, args.count, rank, True)
        if shared["fields"] != pipe["fields"]:
            raise ValueError(f" rank {rank} 批次内容不同")
        print(json.dumps({"exact": True, "runs": [shared, pipe]}), flush=True)
    if args.stress_pipe:
        with ProcessPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(probe, args.data, args.count, rank, True) for rank in range(4)]
            for task in tasks:
                print(json.dumps({"parallel_probe": task.result()}), flush=True)


if __name__ == "__main__":
    main()
