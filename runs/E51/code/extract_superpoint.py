"""Export original SuperPoint detections on the fixed Earth and Moon batches."""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from baselines.data import Data
from baselines.match import env_info, seed_all, sha256
from moonlib import inputs
from runs.E51.code.extract_moon import MODALITY_DIRS, ROOT, RUN, extract, validate_manifest
from runs.E51.code.extract_whu import map_input
from workbench import launch

METHOD = "superpoint_original"
CONFIG = "configs/baselines/splg.json"
WEIGHTS_SHA256 = "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"


class OriginalSuperPoint:
    def __init__(self, repo, device):
        sys.path.insert(0, str(repo))
        from lightglue import SuperPoint

        self.device = device
        self.model = SuperPoint().eval().to(device)
        self.threshold = self.model.conf.detection_threshold
        self.num_features = self.model.conf.max_num_keypoints

    def detect(self, image, _modality):
        tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            result = self.model.extract(tensor)
        return result["keypoints"][0].cpu().numpy(), result["keypoint_scores"][0].cpu().numpy()


def read_whu(item, modality):
    with rasterio.open(item["paths"][modality]) as source:
        image, mapping = map_input(source, modality, Window(*item["window_xywh"]))
    if mapping != item["input"][modality]:
        raise ValueError(f"Input mapping differs from the fixed WHU manifest for {item['pair']}/{modality}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moon", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    output = RUN / "extra" / "data"
    manifests = {domain: output / f"{domain}_manifest.json" for domain in ("moon", "whu")}
    launch.begin(RUN, METHOD, [sys.executable, "-m", "runs.E51.code.extract_superpoint", *sys.argv[1:]],
                 args={"moon": str(args.moon), "weights": str(args.weights), "torch_home": str(args.torch_home),
                       "smoke": args.smoke, "device": args.device},
                 config_file=CONFIG, input_manifests={domain: str(path) for domain, path in manifests.items()})
    try:
        repo = ROOT / config["repo"]
        pinned = subprocess.check_output(["git", "rev-parse", f"HEAD:{config['repo']}"], text=True).strip()
        upstream = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        hub = args.torch_home / "hub"
        cached_weights = hub / "checkpoints" / "superpoint_v1.pth"
        if pinned != upstream or sha256(args.weights) != WEIGHTS_SHA256 or sha256(cached_weights) != WEIGHTS_SHA256:
            raise ValueError("Unexpected LightGlue source commit or original SuperPoint weight checksum")
        torch.hub.set_dir(str(hub))
        seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        detector = OriginalSuperPoint(repo, args.device)

        moon = Data(args.moon.resolve())
        mappings = {m: inputs.get(m, config["input"][m]) for m in MODALITY_DIRS}

        def read_moon(item, modality):
            return mappings[modality](getattr(moon, modality)("val", item["pair"]))

        lunar, whu = [json.loads(path.read_text(encoding="utf-8")) for path in manifests.values()]
        validate_manifest(moon, lunar)
        scenes = Counter(item["scene"] for item in whu)
        if len(scenes) != 4 or set(scenes.values()) != {4} or len({item["pair"] for item in whu}) != 16:
            raise ValueError("Expected the fixed 16-pair WHU manifest")
        if args.smoke:
            output = output / "smoke"
            output.mkdir(exist_ok=True)
            lunar, whu = [lunar[0], lunar[-1]], [whu[0], whu[-1]]
        extract(detector, lunar, read_moon, output, "moon", METHOD)
        extract(detector, whu, read_whu, output, "whu", METHOD)
    except BaseException:
        launch.end(RUN, METHOD, "failed")
        raise
    launch.end(RUN, METHOD, "ok", env=env_info(), weights_sha256=WEIGHTS_SHA256, repo_commit=upstream,
               torch_hub_file=str(cached_weights), detector_config=vars(detector.model.conf),
               resize=detector.model.preprocess_conf["resize"], tf32=False)


if __name__ == "__main__":
    main()
