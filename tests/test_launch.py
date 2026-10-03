"""启动检查与运行记录：什么算不干净、放行开关、launch/<method>.json 的追加与收尾。"""
import json
import subprocess

import pytest

from workbench import launch


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "finetune").mkdir(parents=True)
    (r / "finetune" / "train.py").write_text("x = 1\n")
    (r / ".gitignore").write_text("runs/*/ckpt/\n")
    git(r, "init", "-q")
    git(r, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return r


def test_clean_repo_passes(repo, monkeypatch):
    monkeypatch.delenv(launch.ALLOW_ENV, raising=False)
    st = launch.require_clean(repo)
    assert st["commit"] and not st["dirty"] and not st["allow_dirty"]


def test_run_products_do_not_count(repo):
    (repo / "runs" / "E1" / "preds" / "main").mkdir(parents=True)
    (repo / "runs" / "E1" / "preds" / "main" / "val.jsonl").write_text("{}\n")
    (repo / "runs" / "E1" / "ckpt").mkdir()
    (repo / "runs" / "E1" / "ckpt" / "a.pt").write_text("w")
    assert not launch.repo_state(repo)["dirty"]


@pytest.mark.parametrize("path", ["finetune/train.py", "finetune/new.py", "runs/E1/code/plot.py",
                                  "runs/E1/configs/main.toml"])
def test_uncommitted_code_refused(repo, monkeypatch, path):
    monkeypatch.delenv(launch.ALLOW_ENV, raising=False)
    f = repo / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("y = 2\n")
    with pytest.raises(SystemExit) as e:
        launch.require_clean(repo)
    assert e.value.code == 2


def test_not_a_repo_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(launch.ALLOW_ENV, raising=False)
    with pytest.raises(SystemExit):
        launch.require_clean(tmp_path)


def test_allow_dirty_is_recorded(repo, tmp_path, monkeypatch):
    monkeypatch.setenv(launch.ALLOW_ENV, "1")
    (repo / "finetune" / "train.py").write_text("x = 3\n")
    rec = launch.begin(tmp_path / "runs" / "E1", "main", ["train", "--lr", "1e-5"], repo=repo)
    assert rec["dirty"] and rec["allow_dirty"] and rec["changes"]


def test_begin_appends_and_end_closes_last(repo, tmp_path, monkeypatch):
    monkeypatch.delenv(launch.ALLOW_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    run = tmp_path / "runs" / "E1"
    launch.begin(run, "main", ["a"], repo=repo)
    launch.end(run, "main", "fail")
    launch.begin(run, "main", ["a", "--resume"], repo=repo)
    launch.end(run, "main", "ok")
    items = json.loads((run / "launch" / "main.json").read_text(encoding="utf-8"))
    assert [i["status"] for i in items] == ["fail", "ok"]
    assert items[1]["cmd"] == ["a", "--resume"] and items[1]["cuda_visible_devices"] == "3"
    assert all(i["end"] for i in items)


def test_begin_records_config(repo, tmp_path, monkeypatch):
    monkeypatch.delenv(launch.ALLOW_ENV, raising=False)
    run = tmp_path / "runs" / "E1"
    cfg = {"model": {"name": "loftr"}, "optim": {"lr": 1e-5}, "cexp": {"r_out": -0.25}}
    rec = launch.begin(run, "main", ["run.py", "runs/E1"], repo=repo, entry="scripts/finetune/run.py",
                       config_file="runs/E1/configs/main.toml", args=cfg, config={"adapter": "loftr"})
    items = json.loads((run / "launch" / "main.json").read_text(encoding="utf-8"))
    assert items[0]["args"] == cfg == rec["args"] and items[0]["entry"] == "scripts/finetune/run.py"
    assert items[0]["config"] == {"adapter": "loftr"}
