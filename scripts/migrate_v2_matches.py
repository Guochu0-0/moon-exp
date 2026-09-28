"""一次性：把 B0 / B0m 的原始层点对转成记录格式 v2 的 `_matches.npz` 与 preds meta（#39）。在服务器上跑。

    cd /remote-home/xufang/YGC/moon-exp
    /opt/envs/wb/bin/python scripts/migrate_v2_matches.py

原始层 `YGC/results/baselines/<m>/`（B0）、`YGC/results/baselines_ablation/<m>/`（B0m）是 baselines.match 的产物：
<split>.npz（N×5，整数 = 像素中心）、<split>.jsonl（逐 pair 日志）、<split>.meta.json。
对 runs/<id>/preds/<m>/<split>.jsonl 里的每个 pair，按 baselines.fit 的同一判定：
- 原始日志没有这个 pair（not_run）或记了 error：不写 key；
- 否则取原始点对，没有就是 0×5；交给 PredWriter 同一个打包函数做 +0.5 与按 conf 截断到 2000。
<split>.meta.json：commit 取原始 meta 去掉 -dirty 后缀，dirty 看有没有这个后缀，另写 migrated_from；
原始层缺 meta 的不写。preds 的 jsonl 不动，所以 eval 结果不变。提交 meta.json 后删掉本脚本。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench.records import SPLITS, _npz_key, _pack_matches  # noqa: E402

RESULTS = Path("/remote-home/xufang/YGC/results")
SOURCES = {"B0": "baselines", "B0m": "baselines_ablation"}


def read_log(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["pair"]] = d
    return out


def migrate(pred_dir: Path, raw: Path, split: str) -> str:
    preds = read_log(pred_dir / f"{split}.jsonl")
    log = read_log(raw / f"{split}.jsonl")
    store = {}
    with np.load(raw / f"{split}.npz") as z:
        for pair in preds:
            rec, k = log.get(pair), _npz_key(pair)
            if rec is None or "error" in rec:
                continue
            store[k] = _pack_matches(z[k] if k in z.files else np.zeros((0, 5), np.float32))
    np.savez_compressed(pred_dir / f"{split}_matches.npz", **store)

    meta_path = raw / f"{split}.meta.json"
    if not meta_path.exists():
        return f"{len(store)} 个 pair，原始层无 meta，不写 meta.json"
    commit = json.loads(meta_path.read_text(encoding="utf-8")).get("commit")
    dirty = commit.endswith("-dirty") if commit else None
    meta = {"commit": commit.removesuffix("-dirty") if commit else None, "dirty": dirty,
            "migrated_from": str(raw / f"{split}.npz")}
    (pred_dir / f"{split}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return f"{len(store)} 个 pair，commit {meta['commit'] and meta['commit'][:8]} dirty={dirty}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args(argv)

    for rid, src in SOURCES.items():
        for pred_dir in sorted(p for p in (Path(args.runs) / rid / "preds").iterdir() if p.is_dir()):
            raw = Path(args.results) / src / pred_dir.name
            for split in SPLITS:
                if not (pred_dir / f"{split}.jsonl").exists():
                    continue
                if not (raw / f"{split}.npz").exists() or not (raw / f"{split}.jsonl").exists():
                    print(f"{rid}/{pred_dir.name}/{split}: 原始层缺 {raw}/{split}.npz 或 .jsonl，跳过")
                    continue
                print(f"{rid}/{pred_dir.name}/{split}: {migrate(pred_dir, raw, split)}")


if __name__ == "__main__":
    main()
