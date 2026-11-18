"""Prepare a visually selected WHU building gallery and export both detectors."""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from baselines.adapters.redfeat import ReDFeatAdapter
from baselines.match import env_info, seed_all, sha256
from runs.E51.code.extract_moon import CONFIG, GITLINK, ROOT, RUN, SIZE, extract, write_json
from runs.E51.code.extract_superpoint import OriginalSuperPoint, WEIGHTS_SHA256 as SP_SHA, read_whu
from runs.E51.code.extract_whu import MODALITIES, WEIGHTS_SHA256 as RED_SHA, describe, map_input, validate_grid
from runs.E51.code.plot_detections import plt, render
from workbench import launch

SELECTION = RUN / "code" / "whu_urban_selection.json"
DATA = RUN / "extra" / "data"
FIGURES = RUN / "extra" / "whu_urban"
MANIFEST = DATA / "whu_urban_manifest.json"
PREFIX = "whu_urban"
LIGHTGLUE = "third_party/LightGlue"


def window_from_preview(source, pixel, preview_size):
    x = round((pixel[0] + 0.5) * source.width / preview_size[0] - 0.5) - SIZE // 2
    y = round((pixel[1] + 0.5) * source.height / preview_size[1] - 0.5) - SIZE // 2
    if not (0 <= x <= source.width - SIZE and 0 <= y <= source.height - SIZE):
        raise ValueError(f"Crop exceeds the source image at ({x}, {y})")
    return Window(x, y, SIZE, SIZE)


def gallery(rows, previews):
    fig, axes = plt.subplots(4, 4, figsize=(12.8, 12.8), dpi=125)
    fig.subplots_adjust(left=.018, right=.982, top=.93, bottom=.012, wspace=.045, hspace=.155)
    fig.text(.02, .979, "WHU-OPT-SAR · Building candidates", va="top", fontsize=19, weight="bold")
    try:
        for ax, row in zip(axes.flat, rows):
            ax.imshow(previews[row["pair"]], interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(f"{row['id']}  ·  {row['subject']}", fontsize=10, pad=5)
        target = FIGURES / "gallery.jpg"
        with BytesIO() as buffer:
            fig.savefig(buffer, format="jpeg", facecolor="white",
                        pil_kwargs={"quality": 92, "subsampling": 0, "optimize": True})
            target.with_suffix(".tmp").write_bytes(buffer.getbuffer())
        target.with_suffix(".tmp").replace(target)
    finally:
        plt.close(fig)


def prepare(data):
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))
    rows, previews, provenance = [], {}, {}
    for scene in selected["scenes"]:
        stem = Path(scene["filename"]).stem
        paths = {m: data / m / scene["filename"] for m in MODALITIES}
        with rasterio.open(paths["optical"]) as optical, rasterio.open(paths["sar"]) as sar:
            sources = {"optical": optical, "sar": sar}
            validate_grid(**sources)
            provenance[stem] = {m: describe(sources[m], paths[m]) for m in MODALITIES}
            for m in MODALITIES:
                for key in ("bytes", "sha256"):
                    if provenance[stem][m][key] != scene["files"][m][key]:
                        raise ValueError(f"Transfer integrity differs for {m}/{stem} ({key})")
            for i, item in enumerate(scene["candidates"], 1):
                sample = f"{scene['group']}{i}"
                pair = f"{stem}/{sample}"
                spot = window_from_preview(optical, item["preview_xy"], selected["preview_size"])
                mappings = {m: map_input(sources[m], m, spot)[1] for m in MODALITIES}
                previews[pair] = np.moveaxis(optical.read(mappings["optical"]["bands"], window=spot), 0, -1)
                rows.append({
                    "id": sample, "group": scene["group"], "domain": "earth", "dataset": "WHU-OPT-SAR",
                    "scene": stem, "pair": pair, "subject": item["subject"], "preview_xy": item["preview_xy"],
                    "window_xywh": [spot.col_off, spot.row_off, SIZE, SIZE],
                    "paths": {m: str(paths[m]) for m in MODALITIES}, "input": mappings,
                })
    if len(rows) != 16 or len({row["pair"] for row in rows}) != 16:
        raise ValueError("Expected sixteen distinct urban candidates")
    FIGURES.mkdir(parents=True, exist_ok=True)
    gallery(rows, previews)
    write_json(MANIFEST, rows)
    write_json(DATA / "whu_urban_sources.json", provenance)
    print(f"Prepared {len(provenance)} matched scenes and {len(rows)} fixed crops", flush=True)


def verify_source(relative):
    expected = subprocess.check_output(["git", "rev-parse", f"HEAD:{relative}"], text=True).strip()
    actual = subprocess.check_output(["git", "-C", str(ROOT / relative), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"Unexpected upstream commit for {relative}")
    return actual


def detect(base, smoke):
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    red_weights = base / "weights/official/redfeat/VIS_SAR.pth"
    sp_weights = base / "weights/official/superpoint/superpoint_v1.pth"
    hub = base / "weights/supplement268_torch_home/hub"
    cached = hub / "checkpoints/superpoint_v1.pth"
    if sha256(red_weights) != RED_SHA or sha256(sp_weights) != SP_SHA or sha256(cached) != SP_SHA:
        raise ValueError("Unexpected detector weight checksum")
    upstream = {relative: verify_source(relative) for relative in (GITLINK, LIGHTGLUE)}
    torch.hub.set_dir(str(hub))
    seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    redfeat = ReDFeatAdapter(ROOT / GITLINK, red_weights, **config["params"])
    superpoint = OriginalSuperPoint(ROOT / LIGHTGLUE, "cuda")

    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    output = DATA
    if smoke:
        rows = [rows[0], rows[-1]]
        output = DATA / "smoke"
    images = {(row["pair"], m): read_whu(row, m) for row in rows for m in MODALITIES}

    def read_image(row, modality):
        return images[row["pair"], modality]

    extract(redfeat, rows, read_image, output, PREFIX, "redfeat")
    extract(superpoint, rows, read_image, output, PREFIX, "superpoint_original")
    if not smoke:
        groups = defaultdict(list)
        for row in rows:
            groups[row["group"]].append(row)
        for group, group_rows in groups.items():
            render(PREFIX, f"{group} · {group_rows[0]['scene']}", group_rows, read_image,
                   target=FIGURES / f"{group}.jpg")
    return {
        "upstream": upstream, "weights_sha256": {"redfeat": RED_SHA, "superpoint_original": SP_SHA},
        "redfeat": config["params"], "superpoint_original": vars(superpoint.model.conf),
        "superpoint_resize": superpoint.model.preprocess_conf["resize"], "tf32": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "detect"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    method = f"whu_urban_{args.stage}"
    launch.begin(RUN, method, [sys.executable, "-m", "runs.E51.code.urban_candidates", *sys.argv[1:]],
                 args={"stage": args.stage, "smoke": args.smoke, "base": str(args.base)})
    try:
        metadata = {}
        if args.stage == "prepare":
            prepare(args.base / "dataset/WHU-OPT-SAR")
        else:
            metadata = detect(args.base, args.smoke)
    except BaseException:
        launch.end(RUN, method, "failed")
        raise
    launch.end(RUN, method, "ok", env=env_info(), **metadata)


if __name__ == "__main__":
    main()
