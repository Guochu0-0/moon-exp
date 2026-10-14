"""打包工作台用的精简数据集：只含 Val、Test，SAR 只留工作台显示用的那个波段。

    python scripts/slim_dataset.py <原数据集根目录> <输出目录> [--workers 8]

原数据集根目录可以是含 Processed_Data 的那一层，也可以是 Processed_Data 本身。输出：

    <输出目录>/Processed_Data/{Val,Test}/ROI_xxx/
        Optical/patch_<n>.tif   原样复制
        SAR/patch_<n>.tif       只留第 SAR_DISPLAY_BAND 个波段（float32，deflate，按 strip），其余波段工作台不读
        Label/patch_<n>.txt     原样复制；.prj 不要

工作台只用 SAR 做显示（dataset.to_display 对三维 SAR 取 SAR_DISPLAY_BAND，二维的原样用），指标只用 Label，
所以换成精简版后指标和显示都不变。每张 SAR 写完用 dataset.read_tif 读回，与原波段逐像素比较。
read_tif 只认按 strip 存、没有 predictor 的 deflate，写出时就按这个格式。

输出目录里已有且能读回一致的 SAR 跳过，中断后重跑接着做。全部做完后统计各 split 的 pair 数、有标注 pair 数和大小。
配置指南见 docs/workbench-setup.md。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.dataset import SAR_DISPLAY_BAND, SPLIT_DIRS, Dataset, read_tif  # noqa: E402


def slim_sar(src: Path, dst: Path) -> None:
    band = np.ascontiguousarray(read_tif(src)[..., SAR_DISPLAY_BAND])
    if dst.exists():
        try:
            if np.array_equal(read_tif(dst), band, equal_nan=True):
                return
        except Exception:
            pass
    tmp = dst.with_suffix(".tmp")
    tifffile.imwrite(tmp, band, compression="zlib", predictor=False, photometric="minisblack")
    if not np.array_equal(read_tif(tmp), band, equal_nan=True):
        raise RuntimeError(f"读回不一致：{dst}")
    tmp.replace(dst)


def copy(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="原数据集根目录")
    ap.add_argument("out", help="输出目录（会在其下建 Processed_Data/）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ds = Dataset(args.src)
    out_root = Path(args.out) / "Processed_Data"
    jobs = []
    for split, name in SPLIT_DIRS.items():
        for pair in ds.pairs(split):
            roi, patch = pair.split("/")
            for kind in ("Optical", "SAR", "Label"):
                (out_root / name / roi / kind).mkdir(parents=True, exist_ok=True)
            jobs.append((slim_sar, ds._path(split, pair, "SAR"), out_root / name / roi / "SAR" / f"{patch}.tif"))
            jobs.append((copy, ds._path(split, pair, "Optical"), out_root / name / roi / "Optical" / f"{patch}.tif"))
            lab = ds._path(split, pair, "Label")
            if lab.exists():
                jobs.append((copy, lab, out_root / name / roi / "Label" / f"{patch}.txt"))

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for _ in pool.map(lambda j: j[0](j[1], j[2]), jobs):
            done += 1
            if done % 500 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}", flush=True)

    slim = Dataset(out_root)
    total = 0
    for split, name in SPLIT_DIRS.items():
        assert slim.pairs(split) == ds.pairs(split) and slim.labelled(split) == ds.labelled(split), split
        size = sum(f.stat().st_size for f in (out_root / name).rglob("*") if f.is_file())
        total += size
        print(f"{name}: {len(slim.pairs(split))} 个 pair，{len(slim.labelled(split))} 个有标注，{size / 2**30:.2f} GiB")
    print(f"合计 {total / 2**30:.2f} GiB → {out_root}")


if __name__ == "__main__":
    main()
