"""A recorded weight must be the exact file the official constructor opens."""
import hashlib
import os
from pathlib import Path
import runpy

import pytest


def test_official_cache_is_pinned_and_hashed(tmp_path, monkeypatch):
    driver = runpy.run_path(str(Path(__file__).resolve().parents[1] / "runs/B0/code/supplement268.py"))
    files = {"superpoint": "official/superpoint/superpoint_v1.pth",
             "lightglue": "official/lightglue/superpoint_lightglue_v0-1_arxiv.pth"}
    for name, relative in files.items():
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((name + " official weights").encode())
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "unrelated_cache"))
    metadata = driver["prepare_weights"]("splg", {}, tmp_path)
    expected_cache = tmp_path / "supplement268_torch_home"
    assert Path(os.environ["TORCH_HOME"]) == expected_cache
    for name, entry in metadata.items():
        loaded = Path(entry["path"])
        assert loaded.parent == expected_cache / "hub/checkpoints"
        assert loaded.samefile(tmp_path / files[name])
        assert entry["sha256"] == hashlib.sha256(loaded.read_bytes()).hexdigest()
    # Refuse an existing stale cache rather than hashing a different official file beside it.
    Path(metadata["superpoint"]["path"]).unlink()
    Path(metadata["superpoint"]["path"]).write_bytes(b"another model")
    with pytest.raises(RuntimeError, match="cache differs"):
        driver["prepare_weights"]("splg", {}, tmp_path)
