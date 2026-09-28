"""在一个方法的环境里，对一个 split 跑匹配，存原始点对（RANSAC 之前）。

    python -m baselines.match configs/baselines/loftr.json --split val \
        --data $MOON_DATA --weights-root $MOON_WEIGHTS --out $MOON_RESULTS/baselines

产物（<out>/<method>/）：
    <split>.npz         key = pair（/ 换成 __），value = N×5 float32 (x0, y0, x1, y1, conf)，原 512 网格、整数 = 像素中心
    <split>.jsonl       每 pair 一行：n、sec、error
    <split>.meta.json   配置、权重 sha256、代码 commit、环境版本、机器、起止时间

之后在工作台环境里用 baselines.fit 估仿射、写进 runs/<id>/preds/。兼容 py3.8，只依赖 numpy（加上适配器自己的依赖）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import adapters, inputs
from .data import Data

REPO = Path(__file__).resolve().parents[1]
FLUSH_EVERY = 50


class PairTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise PairTimeout()


def sha256(path, bufsize=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def git_head(path) -> str | None:
    stamp = Path(path) / "COMMIT"   # 没有 .git 的部署副本（git archive 解包，见 #49）在这里记 commit
    if stamp.exists():
        return stamp.read_text().strip()
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        # --no-optional-locks：不刷新索引、不拿 index.lock。gpfs 上 status 很慢，被超时杀掉时会留下残锁
        dirty = subprocess.run(["git", "--no-optional-locks", "-C", str(path), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True, timeout=60)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except Exception:
        return None


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "numpy": np.__version__, "host": socket.gethostname(),
            "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for mod in ("torch", "cv2", "kornia"):
        if mod in sys.modules:
            info[mod] = getattr(sys.modules[mod], "__version__", "?")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def seed_all(seed: int):
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def key(pair: str) -> str:
    return pair.replace("/", "__")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--data", default=os.environ.get("MOON_DATA"))
    ap.add_argument("--weights-root", default=os.environ.get("MOON_WEIGHTS", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timeout", type=int, default=120, help="单个 pair 的超时秒数（0 不设）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 pair（调试用）")
    ap.add_argument("--resume", action="store_true", help="跳过 <split>.jsonl 里已有的 pair")
    ap.add_argument("--weights", help="覆盖配置里的权重路径（如微调产出的 ckpt）")
    ap.add_argument("--name", help="覆盖配置里的 method，决定输出目录名")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.weights:
        cfg["weights"] = str(Path(args.weights).resolve())
    if args.name:
        cfg["method"] = args.name
    method = cfg["method"]
    out = Path(args.out) / method
    out.mkdir(parents=True, exist_ok=True)
    npz_path, log_path, meta_path = (out / f"{args.split}.{ext}" for ext in ("npz", "jsonl", "meta.json"))

    repo = (REPO / cfg["repo"]) if cfg.get("repo") else None
    w = cfg.get("weights")
    if w and w.startswith("repo:"):          # 权重随方法仓库发布（在 submodule 里）
        weights = repo / w[len("repo:"):]
    else:
        weights = (Path(args.weights_root) / w) if w else None
    map_opt = inputs.get("optical", cfg["input"]["optical"])
    map_sar = inputs.get("sar", cfg["input"]["sar"])
    seed = int(cfg.get("seed", 0))

    data = Data(args.data)
    pairs = data.pairs(args.split)
    if args.limit:
        pairs = pairs[:args.limit]

    store, done = {}, set()
    if args.resume and log_path.exists():
        done = {json.loads(l)["pair"] for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        if npz_path.exists():
            with np.load(npz_path) as z:
                store = {k: z[k] for k in z.files}
    elif log_path.exists():
        log_path.unlink()

    t_start = time.time()
    meta = {
        "method": method, "split": args.split, "config": cfg, "config_path": str(args.config),
        "weights": str(weights) if weights else None,
        "weights_sha256": sha256(weights) if weights else None,
        "commit": git_head(REPO), "repo_commit": git_head(repo) if repo else None,
        "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Adapter = adapters.load(cfg["adapter"])
    seed_all(seed)
    t0 = time.time()
    matcher = Adapter(repo=repo, weights=weights, device=args.device, **cfg.get("params", {}))
    meta["load_sec"] = round(time.time() - t0, 2)
    meta["notes"] = getattr(matcher, "notes", "")

    use_alarm = args.timeout > 0 and hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)

    todo = [p for p in pairs if p not in done]
    print(f"[{method}/{args.split}] {len(todo)} pairs to run ({len(done)} done)", flush=True)
    n_err = 0
    with open(log_path, "a", encoding="utf-8") as log:
        for i, pair in enumerate(todo):
            rec = {"pair": pair}
            t = time.time()
            try:
                opt = map_opt(data.optical(args.split, pair))
                sar = map_sar(data.sar(args.split, pair))
                seed_all(seed)
                if use_alarm:
                    signal.alarm(args.timeout)
                try:
                    kp0, kp1, conf = matcher.match(opt, sar)
                finally:
                    if use_alarm:
                        signal.alarm(0)
                kp0 = np.asarray(kp0, np.float64).reshape(-1, 2)
                kp1 = np.asarray(kp1, np.float64).reshape(-1, 2)
                conf = np.ones(len(kp0)) if conf is None else np.asarray(conf, np.float64).reshape(-1)
                if not (len(kp0) == len(kp1) == len(conf)):
                    raise ValueError(f"长度不一致 {len(kp0)}/{len(kp1)}/{len(conf)}")
                store[key(pair)] = np.c_[kp0, kp1, conf].astype(np.float32)
                rec["n"] = len(kp0)
            except PairTimeout:
                rec["error"] = f"timeout>{args.timeout}s"
            except Exception as e:  # 记下来，继续下一个 pair
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["trace"] = traceback.format_exc(limit=3)
            rec["sec"] = round(time.time() - t, 3)
            n_err += "error" in rec
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            if (i + 1) % FLUSH_EVERY == 0 or i + 1 == len(todo):
                np.savez_compressed(npz_path, **store)
                print(f"  {i + 1}/{len(todo)}  errors={n_err}  {time.time() - t_start:.0f}s", flush=True)
    if not todo and not npz_path.exists():
        np.savez_compressed(npz_path, **store)

    meta.update(env=env_info(), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                total_sec=round(time.time() - t_start, 1), n_pairs=len(pairs), n_errors_this_session=n_err)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
