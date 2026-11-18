"""Export the first lunar batch using the B0 ReDFeat detector."""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from baselines.adapters.redfeat import ReDFeatAdapter
from baselines.data import Data
from baselines.match import env_info, seed_all, sha256
from moonlib import inputs
from workbench import launch

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
CONFIG = "configs/baselines/redfeat.json"
GITLINK = "third_party/ReDFeat"
MAPPINGS = {"optical": "Optical", "sar": "SAR"}
SIZE = 512


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_manifest(data):
    pairs = defaultdict(list)
    for pair in data.pairs("val", labelled_only=False):
        if data.path("val", pair, "Optical").is_file():
            pairs[pair.split("/")[0]].append(pair)
    rois = sorted(pairs)
    if len(rois) < 4:
        raise ValueError(f"At least four paired ROIs are required, found {rois}")
    chosen = [rois[(i * (len(rois) - 1) + 1) // 3] for i in range(4)]
    manifest = []
    for roi in chosen:
        available = pairs[roi]
        if len(available) < 4:
            raise ValueError(f"{roi} has fewer than four paired patches")
        for i in range(4):
            pair = available[(2 * i + 1) * len(available) // 8]
            manifest.append({
                "domain": "moon", "split": "val", "roi": roi, "pair": pair,
                "window_xywh": [0, 0, SIZE, SIZE],
                "paths": {m: str(data.path("val", pair, kind)) for m, kind in MAPPINGS.items()},
            })
    return manifest


def extract(detector, data, mappings, manifest, directory):
    points_by_image = {}
    records = []
    for item in manifest:
        for modality in MAPPINGS:
            key = item["pair"].replace("/", "__") + "__" + modality
            try:
                image = mappings[modality](getattr(data, modality)("val", item["pair"]))
                if image.shape != (SIZE, SIZE) or not np.isfinite(image).all():
                    raise ValueError(f"Invalid mapped image of shape {image.shape}")
                xy, score = detector.detect(image, modality)
                points = np.column_stack((xy, score)).astype(np.float32)
                if (not np.isfinite(points).all()
                        or (points[:, :2] < 0).any() or (points[:, :2] >= SIZE).any()
                        or (points[:, 2] < detector.threshold).any()
                        or len(points) > detector.num_features):
                    raise ValueError("Invalid ReDFeat points")
                status = {"status": "ok", "count": len(points), "key": key}
                points_by_image[key] = points
            except Exception as exc:
                status = {"status": "failed", "count": None, "key": None,
                          "error": f"{type(exc).__name__}: {exc}"}
            records.append({"pair": item["pair"], "modality": modality, **status})

    archive = directory / "moon_redfeat.npz"
    temporary = directory / "moon_redfeat.tmp.npz"
    np.savez_compressed(temporary, **points_by_image)
    with np.load(temporary, allow_pickle=False) as saved:
        if saved.files != list(points_by_image):
            raise ValueError("Archive keys do not match the image manifest")
        for key, expected in points_by_image.items():
            np.testing.assert_array_equal(saved[key], expected)
    temporary.replace(archive)
    write_json(directory / "moon_redfeat_counts.json", records)
    failures = [row for row in records if row["status"] == "failed"]
    print(f"{len(manifest)} pairs, {len(points_by_image)} images, {len(failures)} failures", flush=True)
    if failures:
        raise RuntimeError(str(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    rec = launch.begin(RUN, "main", [sys.executable, "-m", "runs.E51.code.extract_moon", *sys.argv[1:]],
                       args={"data": str(args.data), "weights": str(args.weights), "smoke": args.smoke},
                       config_file=CONFIG, config=config)
    try:
        output = RUN / "extra" / "data"
        output.mkdir(parents=True, exist_ok=True)
        data = Data(args.data.resolve())
        manifest_file = output / "moon_manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        else:
            manifest = make_manifest(data)
            write_json(manifest_file, manifest)
        for item in manifest:
            for modality, kind in MAPPINGS.items():
                if item["paths"][modality] != str(data.path("val", item["pair"], kind)):
                    raise ValueError(f"Data root does not match the fixed manifest for {item['pair']}")

        repo = ROOT / GITLINK
        pinned = subprocess.check_output(["git", "rev-parse", f"HEAD:{GITLINK}"], text=True).strip()
        upstream = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if pinned != upstream:
            raise ValueError(f"Expected official ReDFeat commit {pinned}, got {upstream}")
        weights_hash = sha256(args.weights)
        if weights_hash != "6294dcbd7b116c02bba4b7704ed8495ead1666a81494b14630945014ffaeea24":
            raise ValueError(f"Unexpected official VIS_SAR.pth checksum {weights_hash}")
        seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        detector = ReDFeatAdapter(repo, args.weights, **config["params"])
        mappings = {m: inputs.get(m, config["input"][m]) for m in MAPPINGS}
        if args.smoke:
            output = output / "smoke"
            output.mkdir(exist_ok=True)
            manifest = [manifest[0], manifest[-1]]
        extract(detector, data, mappings, manifest, output)
    except BaseException:
        launch.end(RUN, "main", "failed")
        raise
    launch.end(RUN, "main", "ok", env=env_info(), weights_sha256=weights_hash,
               repo_commit=upstream, code_commit=rec["commit"], tf32=False)


if __name__ == "__main__":
    main()
