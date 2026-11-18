"""Export ReDFeat detections on fixed 512px windows of the first WHU scenes."""
import argparse
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from baselines.adapters.redfeat import ReDFeatAdapter
from baselines.match import env_info, seed_all, sha256
from runs.E51.code.extract_moon import CONFIG, GITLINK, ROOT, RUN, SIZE, extract, write_json
from workbench import launch

MODALITIES = ("optical", "sar")
WEIGHTS_SHA256 = "6294dcbd7b116c02bba4b7704ed8495ead1666a81494b14630945014ffaeea24"
RGB_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def describe(dataset, path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "width": dataset.width,
        "height": dataset.height,
        "band_count": dataset.count,
        "dtypes": dataset.dtypes,
        "nodata": [repr(v) if v is not None and not np.isfinite(v) else v for v in dataset.nodatavals],
        "color_interpretation": [color.name for color in dataset.colorinterp],
        "scales": dataset.scales,
        "offsets": dataset.offsets,
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "transform": list(dataset.transform)[:6],
        "gcp_count": len(dataset.gcps[0]),
        "has_rpcs": dataset.rpcs is not None,
        "mask_flags": [[flag.name for flag in band] for band in dataset.mask_flag_enums],
    }


def validate_grid(optical, sar):
    if (optical.shape != sar.shape or optical.crs != sar.crs
            or not optical.transform.almost_equals(sar.transform, precision=1e-8)):
        raise ValueError("Optical and SAR source grids differ")


def paired_windows(optical, sar):
    validate_grid(optical, sar)
    for y in (optical.height // 4, 3 * optical.height // 4):
        for x in (optical.width // 4, 3 * optical.width // 4):
            window = Window(x - SIZE // 2, y - SIZE // 2, SIZE, SIZE)
            if (window.col_off < 0 or window.row_off < 0
                    or window.col_off + SIZE > optical.width or window.row_off + SIZE > optical.height):
                raise ValueError("Quarter-position window exceeds the shared grid")
            yield window


def map_input(dataset, modality, window):
    if modality == "optical":
        colors = [color.name for color in dataset.colorinterp]
        bands = [colors.index(name) + 1 for name in ("red", "green", "blue")]
        if any(dataset.dtypes[b - 1] != "uint8" for b in bands):
            raise ValueError("Expected uint8 RGB optical bands")
    else:
        if dataset.count != 1:
            raise ValueError("Expected single-band SAR")
        bands = [1]
    values = dataset.read(bands, window=window, masked=True)
    if np.ma.getmaskarray(values).any() or not np.isfinite(values.data).all():
        raise ValueError(f"{dataset.name} has invalid pixels in {window}")
    if modality == "optical":
        gray = (RGB_LUMA[:, None, None] * values.data).sum(axis=0) / np.float32(255)
        return gray, {"bands": bands, "mapping": "RGB luminance / 255", "luma": [0.299, 0.587, 0.114]}
    intensity = values.data[0]
    lo, hi = np.percentile(intensity, [2.0, 98.0])
    mapped = np.clip((intensity - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)
    return mapped, {
        "bands": bands, "mapping": "raw intensity percentile stretch", "percentiles": [2.0, 98.0],
        "lo": float(lo), "hi": float(hi),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = RUN / "extra" / "data"
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    selection = json.loads((output / "whu_download_manifest.json").read_text(encoding="utf-8"))
    launch.begin(RUN, "whu_redfeat", [sys.executable, "-m", "runs.E51.code.extract_whu", *sys.argv[1:]],
                 args={"data": str(args.data), "weights": str(args.weights), "smoke": args.smoke},
                 config_file=CONFIG, config=config)
    try:
        repo = ROOT / GITLINK
        pinned = subprocess.check_output(["git", "rev-parse", f"HEAD:{GITLINK}"], text=True).strip()
        upstream = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        weights_hash = sha256(args.weights)
        if pinned != upstream or weights_hash != WEIGHTS_SHA256:
            raise ValueError("Unexpected ReDFeat source commit or VIS_SAR.pth checksum")

        seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        detector = ReDFeatAdapter(repo, args.weights, **config["params"])

        images = {}
        manifest = []
        provenance = {}
        with ExitStack() as stack:
            for scene in selection["scenes"]:
                stem = Path(scene["filename"]).stem
                paths = {m: args.data.resolve() / m / scene["filename"] for m in MODALITIES}
                datasets = {m: stack.enter_context(rasterio.open(paths[m])) for m in MODALITIES}
                provenance[stem] = {m: describe(datasets[m], paths[m]) for m in MODALITIES}
                for m in MODALITIES:
                    if provenance[stem][m]["bytes"] != scene["expected_bytes"][m]:
                        raise ValueError(f"Unexpected file size for {m}/{scene['filename']}")
                for i, window in enumerate(paired_windows(**datasets)):
                    pair = f"{stem}/patch_{i}"
                    mapped = {m: map_input(datasets[m], m, window) for m in MODALITIES}
                    images[pair] = {m: mapped[m][0] for m in MODALITIES}
                    manifest.append({
                        "domain": "earth", "dataset": "WHU-OPT-SAR", "scene": stem,
                        "pair": pair,
                        "window_xywh": [window.col_off, window.row_off, SIZE, SIZE],
                        "paths": {m: str(paths[m]) for m in MODALITIES},
                        "input": {m: mapped[m][1] for m in MODALITIES},
                    })

            if len(manifest) != 16 or len({row["pair"] for row in manifest}) != 16:
                raise ValueError("The WHU batch must contain four windows in each of four scenes")
            write_json(output / "whu_manifest.json", manifest)
            write_json(output / "whu_source_metadata.json", provenance)

            def read_image(item, modality):
                return images[item["pair"]][modality]

            if args.smoke:
                output = output / "smoke"
                output.mkdir(exist_ok=True)
                manifest = [manifest[0], manifest[-1]]
            extract(detector, manifest, read_image, output, "whu")
    except BaseException:
        launch.end(RUN, "whu_redfeat", "failed")
        raise
    launch.end(RUN, "whu_redfeat", "ok", env=env_info(), weights_sha256=weights_hash, repo_commit=upstream,
               tf32=False, rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__)


if __name__ == "__main__":
    main()
