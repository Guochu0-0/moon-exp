"""Extract E51-configured detectors on the new lunar texture sample."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import torch

from baselines.adapters.redfeat import ReDFeatAdapter
from baselines.data import Data
from baselines.match import sha256
from moonlib import inputs
from runs.E51.code.extract_moon import CONFIG, GITLINK, MODALITY_DIRS, ROOT, extract, write_json
from runs.E51.code.extract_superpoint import OriginalSuperPoint, WEIGHTS_SHA256


RED_SHA256 = "6294dcbd7b116c02bba4b7704ed8495ead1666a81494b14630945014ffaeea24"


def verified_commit(repo, gitlink):
    expected = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"HEAD:{gitlink}"], text=True).strip()
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if expected != actual:
        raise ValueError(f"Expected {gitlink} commit {expected}, got {actual}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("redfeat", type=Path)
    parser.add_argument("lightglue", type=Path)
    parser.add_argument("superpoint_weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    data = Data(args.dataset)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.smoke:
        manifest = [manifest[0], manifest[-1]]

    red_weights = args.redfeat / "Pretrained" / "VIS_SAR.pth"
    if sha256(red_weights) != RED_SHA256 or sha256(args.superpoint_weights) != WEIGHTS_SHA256:
        raise ValueError("The detector checkpoint hashes do not match E51")
    red_commit = verified_commit(args.redfeat, GITLINK)
    sp_commit = verified_commit(args.lightglue, "third_party/LightGlue")
    hub = args.output / "hub"
    hub_weights = hub / "checkpoints" / "superpoint_v1.pth"
    hub_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.superpoint_weights, hub_weights)
    torch.hub.set_dir(str(hub))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mappings = {modality: inputs.get(modality, config["input"][modality]) for modality in MODALITY_DIRS}
    images = {
        (row["pair"], modality): mapping(getattr(data, modality)("val", row["pair"]))
        for row in manifest for modality, mapping in mappings.items()
    }

    def read_image(row, modality):
        return images[row["pair"], modality]

    redfeat = ReDFeatAdapter(str(args.redfeat), red_weights, device=args.device, **config["params"])
    extract(redfeat, manifest, read_image, args.output, "texture", "redfeat")
    del redfeat
    detector = OriginalSuperPoint(str(args.lightglue), args.device)
    extract(detector, manifest, read_image, args.output, "texture", "superpoint_original")
    write_json(args.output / "provenance.json", {
        "device": args.device, "torch": str(torch.__version__), "input": config["input"],
        "redfeat_commit": red_commit, "redfeat_sha256": RED_SHA256, "redfeat_config": config["params"],
        "superpoint_commit": sp_commit, "superpoint_sha256": WEIGHTS_SHA256,
        "superpoint_config": vars(detector.model.conf), "superpoint_resize": detector.model.preprocess_conf["resize"],
    })


if __name__ == "__main__":
    main()
