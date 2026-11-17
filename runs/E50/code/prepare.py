"""分片生成固定老师匹配，保留10000原始点对；汇总全部7907对的伪仿射与名单。"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from workbench import launch

R = ROOT / "runs/E50"
CONFIG = "configs/baselines/anymatch_roma__minmax.json"
SHARDS = 16


def wait_idle():
    """每个分段重新等空卡；本脚本本身不创建 CUDA 上下文。"""
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    while True:
        out = subprocess.check_output(["timeout", "15", "nvidia-smi", "-i", gpu,
              "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(out.strip()) < 100:
            return
        print(f"wait for idle gpu {gpu}: {out.strip()}MiB", flush=True)
        time.sleep(30)


def worker(index):
    infer = json.loads((ROOT / CONFIG).read_text())
    for k in range(index, SHARDS, 2):
        name = f"teacher_{k:02d}"
        raw = R / "raw" / name
        meta = raw / "train.meta.json"
        log = raw / "train.jsonl"
        saved = set()
        if log.exists():
            import numpy as np
            if (raw / "train.npz").exists():
                with np.load(raw / "train.npz") as z:
                    saved = {k.replace("__", "/") for k in z.files}
            old = log.read_text()
            records = [json.loads(line) for line in old.splitlines()]
            complete = [r for r in records if "error" not in r and r["pair"] in saved]
            if len(complete) != len(records):
                # 日志每对落盘、npz每50对落盘。恢复前保留原日志，重算未落盘或出错的对。
                (raw / f"train_attempt_{time.time_ns()}.jsonl").write_text(old)
                log.write_text("".join(json.dumps(r) + "\n" for r in complete))
            if meta.exists():
                info = json.loads(meta.read_text())
                if info.get("finished") and len(complete) == info["n_pairs"] and len(saved) == info["n_pairs"]:
                    continue
        wait_idle()
        cmd = [sys.executable, "-m", "baselines.match", CONFIG, "--split", "train", "--data",
               "/remote-home/xufang/YGC/dataset/Moon", "--weights-root", "/remote-home/xufang/YGC/weights",
               "--out", str(R / "raw"), "--name", name, "--shard", f"{k}/{SHARDS}", "--resume"]
        launch.begin(R, name, cmd, entry="runs/E50/code/prepare.py", config=infer,
                     args={"shard": f"{k}/{SHARDS}", "seed": 0, "sample_num": 10000, "split": "train"})
        try:
            subprocess.run(cmd, cwd=ROOT, check=True, timeout=3 * 3600 + 45 * 60)
        except BaseException:
            launch.end(R, name, "fail")
            raise
        launch.end(R, name, "ok")
        # 分片结束释放显存，并留出让卡窗口。
        time.sleep(30)


def fit():
    import hashlib
    import numpy as np
    from baselines.data import Data
    from moonlib.ransac import fit_affine
    from baselines.match import sha256
    from finetune.label import MIN_INLIERS, MIN_MATCHES

    pairs = Data("/remote-home/xufang/YGC/dataset/Moon").pairs("train", labelled_only=False)
    assert len(pairs) == 7907, len(pairs)
    rows, sessions, seen, errors = {}, [], set(), []
    t0 = time.monotonic()
    for k in range(SHARDS):
        raw = R / "raw" / f"teacher_{k:02d}"
        meta = json.loads((raw / "train.meta.json").read_text())
        assert meta.get("finished"), raw
        sessions.append(meta)
        for line in (raw / "train.jsonl").read_text().splitlines():
            rec = json.loads(line)
            assert rec["pair"] not in seen, rec["pair"]
            seen.add(rec["pair"])
            if "error" in rec:
                errors.append(rec)
        with np.load(raw / "train.npz") as z:
            for key in z.files:
                pair = key.replace("__", "/")
                M = z[key].astype(np.float32)
                assert M.shape == (10000, 5), (pair, M.shape)
                A, inliers, _ = fit_affine(M, 3.)
                n = int(inliers.sum()) if inliers is not None else 0
                keep = A is not None and len(M) >= MIN_MATCHES and n >= MIN_INLIERS
                rows[pair] = {"pair": pair, "A": None if A is None else A.tolist(), "n_match": len(M),
                              "n_inliers": n, "keep": bool(keep)}
    assert seen == set(pairs), (len(seen), len(pairs))
    assert not errors, errors[:3]
    assert set(rows) == set(pairs), (len(rows), len(pairs))
    rows = [rows[q] for q in pairs]
    extra = R / "extra"
    extra.mkdir(exist_ok=True)
    labels = extra / "labels.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    kept = [r["pair"] for r in rows if r["keep"]]
    (extra / "kept.json").write_text(json.dumps(kept, indent=1) + "\n")
    upstream = ROOT / "third_party/AnyMatch"
    versions = {"anymatch": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()}
    for name in ("train_roma_outdoor.py", "romatch/losses/robust_loss.py", "romatch/datasets/megadepth.py", "romatch/models/encoders.py"):
        versions[name] = sha256(upstream / "third_party/RoMa_AnyMatch" / name)
    attempts = [r for k in range(SHARDS) for r in json.loads((R / "launch" / f"teacher_{k:02d}.json").read_text())]
    process_sec = sum((datetime.strptime(r["end"], "%Y-%m-%d %H:%M:%S %z") -
                       datetime.strptime(r["start"], "%Y-%m-%d %H:%M:%S %z")).total_seconds()
                      for r in attempts if r["end"])
    summary = {"total": len(rows), "kept": len(kept), "labels_sha256": sha256(labels),
               "keep_sha256": hashlib.sha256("\n".join(kept).encode()).hexdigest(),
               "n_inliers_quantiles": np.quantile([r["n_inliers"] for r in rows], [0, .25, .5, .75, 1]).tolist(),
               "teacher_last_sessions_sec": sum(s["total_sec"] for s in sessions),
               "teacher_process_sec": process_sec, "teacher_unfinished_attempts": sum(not r["end"] for r in attempts),
               "teacher_attempts": attempts, "fit_sec": time.monotonic() - t0,
               "upstream": versions, "teacher_sessions": sessions}
    (extra / "labels_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "teacher_sessions"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, choices=(0, 1))
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    launch.require_clean(ROOT)
    if args.fit:
        cmd = [sys.executable, *sys.argv]
        launch.begin(R, "labels", cmd, entry="runs/E50/code/prepare.py", args={"threshold": 3, "min_matches": 100, "min_inliers": 20})
        try:
            fit()
        except BaseException:
            launch.end(R, "labels", "fail")
            raise
        launch.end(R, "labels", "ok")
    else:
        assert args.worker is not None
        worker(args.worker)
