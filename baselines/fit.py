"""原始点对 → 仿射，写成工作台记录（runs/<id>/preds/<method>/<split>.jsonl）。在工作台环境里跑（需要 cv2）。

    python -m baselines.fit $MOON_RESULTS/baselines/loftr --split val --run runs/B0 [--ransac 3] [--name loftr]

口径（「定义评价协议与指标」#2）：RANSAC 阈值 3 px，实现在 moonlib/ransac.py（与微调代码共用）；
点对少于 3 或内点少于 3 记失败。有标注但没跑到（或匹配阶段报错）的 pair 也写一行失败。

坐标：matcher 输出「整数 = 像素中心」，而标注（workbench 的检查点）是 ArcGIS 角点原点，像素中心在 c + 0.5，
所以两侧都 +0.5 再估计，得到的 A 直接和检查点同一约定。点对原样交给 PredWriter（RANSAC 前的全部点和 conf），
+0.5 与截断由写入端负责。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from workbench.dataset import Dataset
from workbench.protocol import RANSAC_PX
from workbench.records import PredWriter

from moonlib.ransac import fit_affine
from moonlib.data_version import CURRENT, LEGACY, experiment_version, manifest_sha256, required_root


def fit_pairs(raw, name, split, run, pairs, ransac, provenance):
    log = {}
    for line in (raw / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if provenance and d["pair"] in log:
                raise ValueError(f"匹配结果含重复 patch 对：{d['pair']}")
            log[d["pair"]] = d
    if provenance and set(log).difference(pairs):
        raise ValueError("匹配结果含评测清单之外的 patch 对")
    counts = Counter()
    with np.load(raw / f"{split}.npz") as z, PredWriter(run, name, split, provenance=provenance) as w:
        if provenance and set(z.files).difference(p.replace("/", "__") for p in pairs):
            raise ValueError("原始点对含评测清单之外的 patch 对")
        for pair in pairs:
            rec = log.get(pair)
            k = pair.replace("/", "__")
            A, M, extra = None, None, {}
            if rec is None:
                fail = "not_run"
            elif "error" in rec:
                fail, extra = "error: " + rec["error"], {"sec": rec.get("sec")}
            elif provenance and k not in z.files:
                fail = "not_run"
            else:
                M = z[k] if k in z.files else np.zeros((0, 5), np.float32)
                extra = {"n_matches": len(M), "sec": rec.get("sec")}
                try:
                    if provenance and (M.shape != (rec.get("n"), 5) or not np.isfinite(M).all()):
                        raise ValueError("原始点对的条数或数值无效")
                    A, inl, fail = fit_affine(M, ransac)
                    if inl is not None:
                        extra["n_inliers"] = int(inl.sum())
                except Exception as e:
                    if not provenance:
                        raise
                    fail = f"error: {type(e).__name__}: {e}"
            w.write(pair, A, fail=fail, matches=M, **extra)
            counts[fail.split(":")[0] if fail else "ok"] += 1
    print(f"[{name}/{split}] ransac={ransac}", dict(counts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="baselines.match 的 <out>/<method> 目录")
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--run", required=True, help="实验目录，例如 runs/B0")
    ap.add_argument("--name", help="方法名，默认取 raw 目录名")
    ap.add_argument("--ransac", type=float, default=RANSAC_PX)
    ap.add_argument("--data", help="覆盖本机的实验数据根")
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    name = args.name or raw.name
    run, split = Path(args.run), args.split
    version = experiment_version(run)
    ds = Dataset(required_root(version, args.data))
    meta_path = raw / f"{split}.meta.json"
    if version != CURRENT:
        if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")).get("data_version", LEGACY) != version:
            raise ValueError(f"{raw}/{split} 的匹配来源与 Moon-{version} 不符")
        fit_pairs(raw, name, split, run, ds.labelled(split), args.ransac, None)
        return

    if args.ransac != RANSAC_PX:
        ap.error(f"Moon-v2 使用 {RANSAC_PX:g} px 的 RANSAC 阈值")
    pairs = ds.pairs(split) if split == "val" else ds.labelled(split)
    expected = {"data_version": version, "split": split, "manifest_sha256": manifest_sha256(pairs)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("finished") or any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{raw}/{split} 的匹配来源与当前评测清单不符")
    provenance = {**expected, "ransac_px": args.ransac,
                  **{k: meta[k] for k in ("config_sha256", "weights_sha256")}}
    if name.startswith("step") and name[4:].isdecimal():
        provenance["step"] = int(name[4:])
    # 先在目标文件系统写完一套预测。元数据最后发布，是重试验证使用的完成标记。
    with TemporaryDirectory(dir=run) as tmp:
        fit_pairs(raw, name, split, tmp, pairs, args.ransac, provenance)
        source = Path(tmp) / "preds" / name
        output = run / "preds" / name
        output.mkdir(parents=True, exist_ok=True)
        for suffix in (".jsonl", "_matches.npz", ".meta.json"):
            src, dst = source / f"{split}{suffix}", output / f"{split}{suffix}"
            if src.exists():
                src.replace(dst)
            else:
                dst.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
