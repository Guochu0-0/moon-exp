import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from baselines import fit
from finetune.config import expand
from moonlib.data_version import manifest_sha256, prediction_sha256, sha256
from workbench.__main__ import main as workbench


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_driver_routes_training_and_evaluation_and_selects_valid_checkpoint(tmp_path, monkeypatch, version):
    spec = importlib.util.spec_from_file_location(
        "versioned_driver", Path(__file__).resolve().parents[1] / "scripts/finetune/run.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    run = tmp_path / "runs/E1"
    run.mkdir(parents=True)
    (run / "exp.toml").write_text('id = "E1"\ntitle = "test"\n' +
                                  ('data_version = "v2"\n' if version == "v2" else ''))
    roots = {v: tmp_path / v for v in ("v1", "v2")}
    pairs = {"v1": ["ROI_035/patch_1"], "v2": ["ROI_001/patch_1", "ROI_001/patch_2"]}
    test_pairs = ["ROI_035/patch_1", "ROI_006/patch_1"]
    for v, root in roots.items():
        (root / "Train").mkdir(parents=True)
        for split, ids in [("Val", pairs[v]), ("Test", test_pairs if v == "v2" else test_pairs[1:])]:
            for pair in ids:
                roi, patch = pair.split("/")
                for kind in ("Optical", "SAR"):
                    path = root / split / roi / kind / f"{patch}.tif"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                if v == "v1" or split == "Test":
                    path = root / split / roi / "Label" / f"{patch}.txt"
                    path.parent.mkdir(exist_ok=True)
                    path.write_text("1 -1 1 -1\n")
    monkeypatch.setenv("MOON_DATA", str(roots["v1"]))
    monkeypatch.setenv("MOON_DATA_V2", str(roots["v2"]))
    cfg = expand({"model": {"name": "roma"},
                  "run": {"select": "val_inlier_mean"} if version == "v2" else {}})
    repo = tmp_path / "repo"
    infer = repo / cfg["model"]["config"]
    infer.parent.mkdir(parents=True)
    infer.write_text('{}')
    monkeypatch.setattr(driver, "REPO", repo)
    calls = []
    fail_test = False
    xy = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=np.float32)
    matching = np.c_[xy, xy, np.ones(4)]

    def run_child(cmd, logfile, env, gpu):
        args = list(map(str, cmd[3:]))
        module = cmd[2]
        with monkeypatch.context() as mp:
            for k, v in env.items():
                mp.setenv(k, str(v))
            if module == "finetune.train":
                if version == "v2":
                    assert Path(args[args.index("--data") + 1]) == roots[version]
                else:
                    assert Path(env.get("MOON_DATA", str(roots[version]))) == roots[version]
                for step in ((1, 2, 3, 4, 5) if version == "v2" else (1, 2, 3)):
                    (Path(logfile).parent / f"ckpt_{step}.pt").write_bytes(bytes([step]))
            elif module == "baselines.match":
                get = lambda key: args[args.index(key) + 1]
                split, name = get("--split"), get("--name")
                out = Path(get("--out")) / name
                out.mkdir(parents=True, exist_ok=True)
                ids = pairs[version] if split == "val" else test_pairs if version == "v2" else test_pairs[1:]
                calls.append((split, name))
                if split == "test" and fail_test:
                    raise subprocess.CalledProcessError(1, cmd)
                rows, store = [], {}
                for i, pair in enumerate(ids):
                    if version == "v2" and split == "val" and name == "step4" and i == 1:
                        continue
                    if version == "v2" and split == "val" and name == "step3" and i == 1:
                        rows.append({"pair": pair, "error": "inference failed"})
                        continue
                    mat = matching if (split == "test" or name in ("step3", "step5")
                                       or name == "step2" and i == 0) else matching[:0]
                    rows.append({"pair": pair, "n": len(mat)})
                    store[pair.replace("/", "__")] = mat
                if version == "v2" and split == "val" and name == "step5":
                    rows.append(rows[0])
                (out / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows)))
                np.savez_compressed(out / f"{split}.npz", **store)
                if version == "v2":
                    assert Path(get("--run")) == run
                weight = Path(get("--weights")).resolve()
                (out / f"{split}.meta.json").write_text(json.dumps({
                    "data_version": version, "split": split, "manifest_sha256": manifest_sha256(ids),
                    "weights": str(weight), "weights_sha256": sha256(weight), "config": {}, "finished": "yes",
                    "config_sha256": sha256(repo / args[0]),
                }))
            elif module == "baselines.fit":
                fit.main(args)
            elif module == "workbench":
                workbench((["--runs", str(run.parent)] if "--runs" not in args else []) + args)

    monkeypatch.setattr(driver, "run", run_child)
    job = driver.Job(run, "main", cfg, "0")
    job()
    peak = json.loads((job.sw / "peak.json").read_text())
    assert peak["step"] == 2
    assert [c for c in calls if c[0] == "test"] == [("test", "step2")]
    if version == "v2":
        assert peak["data_version"] == "v2"
        scores = {c["step"]: c for c in peak["candidates"]}
        assert scores[1]["summary"]["fit_failures"] == 2
        assert scores[2]["summary"]["inlier_mean"] == scores[2]["summary"]["inlier_median"] == 0.5
        assert scores[3]["summary"]["execution_errors"] == 1
        assert scores[3]["summary"]["inlier_mean"] is None
        assert scores[4]["coverage"]["missing"] == 1
        assert scores[4]["summary"]["execution_errors"] == 0 and scores[4]["summary"]["inlier_mean"] is None
        assert "重复 patch 对" in scores[5]["error"]
        assert peak["test"]["auc@10"] == 1
        meta = json.loads((run / "preds/main/test.meta.json").read_text())
        assert meta["data_version"] == "v2" and meta["step"] == 2

        calls.clear()
        job.run_evaluation()
        assert ("val", "step2") not in calls and ("test", "step2") not in calls

        cached_val = job.S / "preds/step2/val.jsonl"
        rows = cached_val.read_text(encoding="utf-8").splitlines()
        cached_val.write_text("\n".join([*rows, rows[0]]) + "\n", encoding="utf-8")
        cached_meta = cached_val.with_suffix(".meta.json")
        metadata = json.loads(cached_meta.read_text(encoding="utf-8"))
        metadata["prediction_sha256"] = prediction_sha256(cached_val)
        cached_meta.write_text(json.dumps(metadata), encoding="utf-8")
        calls.clear()
        job.run_evaluation()
        assert ("val", "step2") in calls and ("test", "step2") not in calls

        (job.ck / "ckpt_2.pt").write_bytes(b"updated")
        calls.clear()
        job.run_evaluation()
        assert ("val", "step2") in calls and ("test", "step2") in calls

        (job.S / "preds/step2/test_matches.npz").unlink()
        calls.clear()
        job.run_evaluation()
        assert ("val", "step2") not in calls and ("test", "step2") in calls

        (job.ck / ".evaluating").mkdir()
        with pytest.raises(driver.EvaluationError, match="已有评测进程"):
            job.run_evaluation()
        (job.ck / ".evaluating").rmdir()

        previous_peak = (job.sw / "peak.json").read_bytes()
        previous_meta = (run / "preds/main/test.meta.json").read_bytes()
        (job.ck / "ckpt_2.pt").write_bytes(b"broken")
        fail_test = True
        with pytest.raises(subprocess.CalledProcessError):
            job.run_evaluation()
        assert (job.sw / "peak.json").read_bytes() == previous_peak
        assert (run / "preds/main/test.meta.json").read_bytes() == previous_meta
        attempt = json.loads((job.sw / "peak_attempt.json").read_text())
        assert attempt["step"] == 2 and attempt["test"] is None and attempt["error"]
