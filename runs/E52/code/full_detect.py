"""Export before-matching detections for an entire Test split, with per-pair resume."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
import torch

from baselines.adapters.base import long_side_size, to_original
from baselines.adapters.comir import CoMIRAdapter
from baselines.adapters.redfeat import ReDFeatAdapter
from baselines.adapters.splg import SP_CONF
from baselines.match import sha256
from runs.E51.code.extract_moon import ROOT, SIZE, write_json
from runs.E51.code.extract_superpoint import OriginalSuperPoint, WEIGHTS_SHA256
from workbench import launch
from .full_data import DOMAINS, MODALITIES, image_reader, moon_pairs, whu_pairs
from .repeatability import METHODS, RUN, TEST_COUNTS
from .texture_detect import RED_SHA256, verified_commit

def config(method):
    name = "splg" if method == "superpoint_original" else method
    return json.loads((ROOT / f"configs/baselines/{name}.json").read_text(encoding="utf-8"))


def resized(image, long_side, df=1):
    import cv2

    h, w = image.shape
    hn, wn = long_side_size(h, w, long_side, df=df)
    if (hn, wn) != (h, w):
        image = cv2.resize(image, (wn, hn))
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    return tensor, (h, w, hn, wn)


def point_model(method, cfg, device):
    repo = ROOT / cfg["repo"]
    sys.path.insert(0, str(repo))
    params = cfg["params"]
    if method == "spsg":
        from models.superpoint import SuperPoint

        model = SuperPoint({k: params[k] for k in ("max_keypoints", "nms_radius", "keypoint_threshold")})
        df = 1
    else:
        from lightglue import SuperPoint

        model = SuperPoint(**SP_CONF)
        df = 8
    model.eval().to(device)

    @torch.inference_mode()
    def detect(image, _modality):
        tensor, grid = resized(image, params["long_side"], df=df)
        if method == "spsg":
            out = model({"image": tensor.to(device)})
            xy, score = out["keypoints"][0], out["scores"][0]
        else:
            out = model.extract(tensor.to(device))
            xy, score = out["keypoints"][0], out["keypoint_scores"][0]
        return to_original(xy.cpu().numpy(), *grid), score.cpu().numpy()

    return detect


def detector(args, cfg):
    method = args.method
    repo = ROOT / cfg["repo"]
    commit = verified_commit(repo, cfg["repo"])
    info = {"upstream_commit": commit}
    if method in ("superpoint_original", "minima_splg"):
        path = Path(torch.hub.get_dir()) / "checkpoints" / "superpoint_v1.pth"
        info["weights_sha256"] = sha256(path)
        if info["weights_sha256"] != WEIGHTS_SHA256:
            raise ValueError(f"Unexpected SuperPoint weights: {path}")
    if method == "redfeat":
        path = args.weights / "official" / "redfeat" / "VIS_SAR.pth"
        info["weights_sha256"] = sha256(path)
        if info["weights_sha256"] != RED_SHA256:
            raise ValueError(f"Unexpected ReDFeat weights: {path}")
        single = ReDFeatAdapter(repo, path, device=args.device, **cfg["params"]).detect
    elif method == "superpoint_original":
        single = OriginalSuperPoint(repo, args.device).detect
    elif method in ("spsg", "minima_splg"):
        single = point_model(method, cfg, args.device)
    elif method == "comir":
        path = args.weights / cfg["weights"]
        info["weights_sha256"] = sha256(path)
        adapter = CoMIRAdapter(repo, path, device=args.device, **cfg["params"])

        def detect_pair(images):
            representations = adapter.comir(images["optical"], images["sar"])
            result = {}
            for modality, representation in zip(MODALITIES, representations):
                features = adapter._features(representation[adapter.channel])
                # mpicbg exposes no response. Scale ranks features here.
                result[modality] = [(float(f.location[0]), float(f.location[1]), float(f.scale)) for f in features]
            return result

        return detect_pair, info
    else:
        raise ValueError(method)

    def detect_pair(images):
        return {m: np.column_stack(single(images[m], m)) for m in MODALITIES}

    return detect_pair, info


def atomic_npz(path, points):
    arrays = {m: np.asarray(points[m], dtype=np.float32).reshape(-1, 3) for m in MODALITIES}
    for modality, array in arrays.items():
        if not np.isfinite(array).all() or ((array[:, :2] < -0.5) | (array[:, :2] >= SIZE - 0.5)).any():
            raise ValueError(f"Invalid coordinates or scores: {path}/{modality}")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_mat(path, images):
    from scipy.io import savemat

    temporary = path.with_suffix(".tmp")
    savemat(temporary, images, appendmat=False)
    temporary.replace(path)


def run(args):
    cfg = config(args.method)
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    portable_rift = args.method == "rift2" and args.rift2_backend == "portable"
    prepare_rift = args.method == "rift2" and not portable_rift
    if prepare_rift:
        process, info = save_mat, {"stage": "prepare_matlab_inputs"}
        extension = ".mat"
    else:
        if portable_rift:
            from .rift2_portable import detect

            def detect_pair(images):
                return {modality: detect(image) for modality, image in images.items()}

            info = {"backend": "SciPy FFT and floating-point FAST-9",
                    "upstream_commit": verified_commit(ROOT / cfg["repo"], cfg["repo"])}
        else:
            detect_pair, info = detector(args, cfg)

        def process(path, images):
            atomic_npz(path, detect_pair(images))

        extension = ".npz"

    directory = args.output / args.domain
    output_method = "rift2_portable" if portable_rift else "rift2_input" if prepare_rift else args.method
    method_dir = directory / output_method
    method_dir.mkdir(parents=True, exist_ok=True)
    rows = list(moon_pairs(args.dataset) if args.domain == "moon" else whu_pairs(args.dataset))
    counts = len(rows), sum(row["reference"] is not None for row in rows)
    if counts != TEST_COUNTS[args.domain]:
        raise ValueError(f"Unexpected image-pair and reference counts for {args.domain}: {counts}")
    if args.limit:
        rows = rows[:args.limit]
    manifest = directory / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != rows:
            raise ValueError(f"Manifest changed: {manifest}")
    else:
        method_manifest = method_dir / "manifest.json"
        write_json(method_manifest, rows)
        method_manifest.replace(manifest)
    def pending(group_rows, read):
        for row in group_rows:
            path = method_dir / (row["pair"].replace("/", "__") + extension)
            if path.exists():
                continue
            images = read(row)
            if any(image.shape != (SIZE, SIZE) or not np.isfinite(image).all() for image in images.values()):
                raise ValueError(f"Invalid mapped images for {row['pair']}")
            yield path, images

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for group, group_rows in groupby(rows, key=lambda row: row["group"]):
            group_rows = list(group_rows)
            with image_reader(args.domain, group_rows[0]["paths"]) as read:
                jobs = []
                for path, images in pending(group_rows, read):
                    if args.workers == 1:
                        process(path, images)
                    else:
                        jobs.append(pool.submit(process, path, images))
                        if len(jobs) >= 2 * args.workers:
                            jobs.pop(0).result()
                for job in jobs:
                    job.result()
            print(f"{args.domain} {args.method} {group}", flush=True)
    print(f"{args.domain} {args.method}: {len(rows)} pairs", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=DOMAINS)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--rift2-backend", choices=("matlab", "portable"), default="matlab")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel pairs for CoMIR or the portable RIFT2 detector")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    parallel_rift = args.method == "rift2" and args.rift2_backend == "portable" and args.device == "cpu"
    if args.workers > 1 and args.method != "comir" and not parallel_rift:
        parser.error("--workers > 1 requires CoMIR or the portable RIFT2 CPU detector")

    config_name = f"full_{args.domain}_{args.method}"
    if args.method == "rift2" and args.rift2_backend == "portable":
        config_name += "_portable"
    if args.limit:
        config_name += "_smoke"
    launch.begin(RUN, config_name, [sys.executable, "-m", __spec__.name, *sys.argv[1:]],
                 args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                 config=config(args.method))
    try:
        info = run(args)
    except BaseException:
        launch.end(RUN, config_name, "failed")
        raise
    launch.end(RUN, config_name, "ok", **info)


if __name__ == "__main__":
    main()
