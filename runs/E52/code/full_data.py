"""Enumerate all Moon and official WHU Test inputs on the native patch grids."""

from contextlib import ExitStack, contextmanager
from functools import cache
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from baselines.data import Data
from moonlib import inputs
from runs.E51.code.extract_moon import MODALITY_DIRS, SIZE
from runs.E51.code.extract_whu import map_input, validate_grid
from .repeatability import affine

SOURCES = Path(__file__).with_name("whu_test_sources.json")
DOMAINS = ("moon", "whu")
MODALITIES = tuple(MODALITY_DIRS)
MOON_MAPPINGS = {"optical": inputs.get("optical", "div255"), "sar": inputs.get("sar", "p2p98")}


def axis_windows(length):
    if length < SIZE:
        raise ValueError(f"Source dimension {length} is smaller than {SIZE}")
    return sorted(set(range(0, length - SIZE + 1, SIZE)) | {length - SIZE})


def reference(labels):
    points = np.loadtxt(labels, dtype=np.float64, ndmin=2)[:, :4]
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"Invalid ground-truth points in {labels}")
    points[:, [1, 3]] *= -1
    source, target = points[:, :2], points[:, 2:]
    linear, shift = affine(source, target)
    if np.linalg.matrix_rank(linear) < 2:
        raise ValueError(f"Noninvertible evaluation geometry in {labels}")
    holdout = []
    for index in range(len(source)):
        keep = np.arange(len(source)) != index
        try:
            lhs, rhs = affine(source[keep], target[keep])
        except ValueError:
            continue
        holdout.append(float(np.linalg.norm(source[index] @ lhs + rhs - target[index])))
    return {
        "affine": np.vstack((linear, shift)).T.tolist(),
        "checkpoints": len(source),
        "leave_one_out_mean_px": float(np.mean(holdout)) if holdout else None,
        "leave_one_out_count": len(holdout),
    }


def moon_pairs(root):
    data = Data(root)
    pairs = data.pairs("test", labelled_only=False)
    for pair in pairs:
        paths = {m: str(data.path("test", pair, kind)) for m, kind in MODALITY_DIRS.items()}
        if not all(Path(path).is_file() for path in paths.values()):
            raise FileNotFoundError(paths)
        labels = data.path("test", pair, "Label")
        yield {"domain": "moon", "pair": pair, "group": pair.split("/")[0], "paths": paths,
               "window_xywh": [0, 0, SIZE, SIZE], "reference": reference(labels) if labels.is_file() else None}


@cache
def whu_sources():
    return json.loads(SOURCES.read_text(encoding="utf-8"))["test"]


def whu_pairs(root):
    for filename in whu_sources():
        scene = Path(filename).stem
        paths = {m: str(root / m / filename) for m in MODALITIES}
        with open_whu(paths) as datasets:
            xs = axis_windows(datasets["optical"].width)
            ys = axis_windows(datasets["optical"].height)
        for y in ys:
            for x in xs:
                yield {"domain": "whu", "pair": f"{scene}/x{x}_y{y}", "group": scene, "paths": paths,
                       "window_xywh": [x, y, SIZE, SIZE], "reference": {"affine": [[1, 0, 0], [0, 1, 0]]}}


@contextmanager
def open_whu(paths):
    with ExitStack() as stack:
        datasets = {m: stack.enter_context(rasterio.open(path)) for m, path in paths.items()}
        validate_grid(**datasets)
        if datasets["optical"].shape != (3704, 5556):
            raise ValueError(f"Unexpected WHU source dimensions: {paths}")
        yield datasets


@contextmanager
def image_reader(domain, paths):
    if domain == "moon":
        from baselines.tif import read_tif

        def read(row):
            return {m: MOON_MAPPINGS[m](read_tif(path)) for m, path in row["paths"].items()}

        yield read
    else:
        with open_whu(paths) as datasets:
            def read(row):
                window = Window(*row["window_xywh"])
                return {m: map_input(source, m, window)[0] for m, source in datasets.items()}

            yield read
