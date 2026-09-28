"""sync：按实验把服务器 runs/<id>/ 里被 gitignore 的部分拉回本地。远端用本地目录冒充，ssh 换成本地 shell。"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from workbench import __main__ as cli
from workbench import sync


def _posix_shell():
    """本地冒充远端的 shell：需要 GNU find / tar / timeout。Windows 上用 Git 自带的 bash（PATH 里的 bash 可能是 WSL）。"""
    if os.name == "nt":
        r = subprocess.run(["git", "--exec-path"], capture_output=True, text=True)
        bash = Path(r.stdout.strip()).parents[2] / "bin" / "bash.exe" if r.returncode == 0 else None
        return [str(bash), "-c"] if bash and bash.exists() else None
    return ["bash", "-c"] if shutil.which("bash") else None


SHELL = _posix_shell()
pytestmark = pytest.mark.skipif(SHELL is None or shutil.which("tar") is None, reason="需要 bash 与 tar")

REMOTE_FILES = [
    "B0/exp.toml",
    "B0/preds/loftr/val.jsonl",
    "B0/preds/loftr/val.meta.json",
    "B0/preds/loftr/val_matches.npz",
    "B0/preds/loftr/test_matches.npz",
    "B0/preds/roma/val_matches.npz",
    "E1/preds/main/val_matches.npz",
    "E1/inter/main/certainty/meta.json",
    "E1/inter/main/certainty/val.npz",
    "E1/inter/main/warp/meta.json",
    "E1/inter/main/warp/val/ROI_002__patch_0.png",
    "E1/inter/main/overlay/val/ROI_002__patch_0.png",
    "E1/tb/main/scalars/events.out.tfevents.1.host",
    "E1/tb/main/media/events.out.tfevents.1.host",
    "E1/tb/up/lightning_logs/version_0/events.out.tfevents.2.host",
    "E1/tb/up/lightning_logs/version_0/hparams.yaml",
    "E1/tb/up/lightning_logs/version_0/checkpoints/last.ckpt",
    "E1/ckpt/main/last.ckpt",
    "E1/ckpt/main/best.pth",
]


@pytest.fixture
def remote(tmp_path):
    root = tmp_path / "remote"
    for rel in REMOTE_FILES:
        p = root / "runs" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rel.encode())
    return root


def pulled(local: Path) -> set[str]:
    return {p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()}


def run(remote, local, ids, **kw):
    return sync.sync(ids, local, remote_root=str(remote).replace("\\", "/"), shell=SHELL, **kw)


def test_default_pulls_matches_and_tb_scalars_only(remote, tmp_path):
    local = tmp_path / "local"
    run(remote, local, ["B0", "E1"])
    assert pulled(local) == {
        "B0/preds/loftr/val_matches.npz",
        "B0/preds/loftr/test_matches.npz",
        "B0/preds/roma/val_matches.npz",
        "E1/preds/main/val_matches.npz",
        "E1/tb/main/scalars/events.out.tfevents.1.host",
        "E1/tb/up/lightning_logs/version_0/events.out.tfevents.2.host",
        "E1/tb/up/lightning_logs/version_0/hparams.yaml",
    }
    assert (local / "B0/preds/loftr/val_matches.npz").read_bytes() == b"B0/preds/loftr/val_matches.npz"


def test_only_requested_experiments(remote, tmp_path):
    local = tmp_path / "local"
    run(remote, local, ["B0"])
    assert all(p.startswith("B0/") for p in pulled(local))


def test_extra_by_name(remote, tmp_path):
    local = tmp_path / "local"
    run(remote, local, ["E1"], extras=["certainty", "warp"])
    got = {p for p in pulled(local) if "/inter/" in p}
    assert got == {"E1/inter/main/certainty/val.npz", "E1/inter/main/warp/val/ROI_002__patch_0.png"}


def test_tb_all_still_excludes_checkpoints(remote, tmp_path):
    local = tmp_path / "local"
    run(remote, local, ["E1"], tb_all=True)
    got = pulled(local)
    assert "E1/tb/main/media/events.out.tfevents.1.host" in got
    assert not any("ckpt" in p for p in got)


def test_incremental_skips_unchanged(remote, tmp_path):
    local = tmp_path / "local"
    first = run(remote, local, ["B0"])
    assert len(first.fetched) == 3 and not first.skipped
    second = run(remote, local, ["B0"])
    assert not second.fetched and len(second.skipped) == 3

    p = remote / "runs/B0/preds/roma/val_matches.npz"
    p.write_bytes(b"changed and longer")
    third = run(remote, local, ["B0"])
    assert third.fetched == ["B0/preds/roma/val_matches.npz"]
    assert (local / "B0/preds/roma/val_matches.npz").read_bytes() == b"changed and longer"


def test_missing_experiment_is_reported(remote, tmp_path):
    local = tmp_path / "local"
    r = run(remote, local, ["B0", "E9"])
    assert r.missing == ["E9"] and len(r.fetched) == 3


def test_tb_version_rule_only_for_upstream_layout():
    assert sync.should_pull("E1/tb/up/lightning_logs/version_3/events.x")
    assert not sync.should_pull("E1/tb/main/media/version_0/events.x")
    assert sync.should_pull("E1/tb/main/media/version_0/events.x", tb_all=True)


def test_cli_passes_options_through(monkeypatch, tmp_path):
    seen = {}

    def fake(ids, runs, **kw):
        seen.update(ids=ids, runs=runs, **kw)
        return sync.Result([], [], [], 0)

    monkeypatch.setattr(sync, "sync", fake)
    cli.main(["--runs", str(tmp_path), "sync", "B0", "B0m", "--extra", "certainty", "--extra", "warp"])
    assert seen == {"ids": ["B0", "B0m"], "runs": tmp_path, "host": None, "remote_root": None,
                    "extras": ["certainty", "warp"], "tb_all": False}
    cli.main(["--runs", str(tmp_path), "sync", "E1", "--tb", "all", "--host", "h", "--remote-root", "/r"])
    assert seen["tb_all"] is True and seen["host"] == "h" and seen["remote_root"] == "/r"

    monkeypatch.setattr(sync, "sync", lambda *a, **kw: sync.Result([], [], ["E9"], 0))
    with pytest.raises(SystemExit) as e:
        cli.main(["--runs", str(tmp_path), "sync", "E9"])
    assert e.value.code == 1


def test_host_and_root_defaults(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(sync, "list_remote", lambda ids, root, shell: (seen.append((root, shell)), ({}, []))[1])
    monkeypatch.delenv(sync.HOST_ENV, raising=False)
    monkeypatch.delenv(sync.ROOT_ENV, raising=False)
    sync.sync(["B0"], tmp_path)
    monkeypatch.setenv(sync.HOST_ENV, "xufang160外网")
    monkeypatch.setenv(sync.ROOT_ENV, "/elsewhere")
    sync.sync(["B0"], tmp_path)
    sync.sync(["B0"], tmp_path, host="h", remote_root="/r")
    assert seen == [(sync.DEFAULT_ROOT, ["ssh", sync.DEFAULT_HOST]), ("/elsewhere", ["ssh", "xufang160外网"]),
                    ("/r", ["ssh", "h"])]
