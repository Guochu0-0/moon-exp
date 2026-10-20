"""结票收尾脚本：从总入口 main() 测。ssh 换成本地 shell，临时目录冒充 GitHub（裸仓库）、gpfs 主 checkout 与 worktree、
A6000 主 checkout 与 worktree、本地仓库；gh 换成按裸仓库办事的替身。先例：test_sync.py。"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("close_ticket", REPO / "scripts" / "close_ticket.py")
close_ticket = importlib.util.module_from_spec(_spec)
sys.modules["close_ticket"] = close_ticket   # dataclass 要在 sys.modules 里找到所在模块
_spec.loader.exec_module(close_ticket)

SHELL = ["bash", "-c"] if os.name != "nt" and shutil.which("bash") else None
pytestmark = pytest.mark.skipif(SHELL is None or shutil.which("tar") is None, reason="需要 bash、tar 与 GNU stat（在服务器上跑）")

# gh 替身：PR 列表存在 JSON 里，headRefOid 现读裸仓库；merge 在临时克隆里做 --no-ff 合并再推回裸仓库
FAKE_GH = r'''
import json, os, subprocess, sys, tempfile
path = os.environ["FAKE_GH_STATE"]
st = json.load(open(path))
args = sys.argv[1:]
st["calls"].append(args)
def git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
def oid(head):
    return git("-C", st["bare"], "rev-parse", "refs/heads/" + head)
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
def save():
    json.dump(st, open(path, "w"))
if args[:2] == ["pr", "list"]:
    head, state = opt("--head"), opt("--state", "open")
    out = [dict(pr, headRefOid=oid(pr["headRefName"])) for pr in st["prs"]
           if (head is None or pr["headRefName"] == head) and (state == "all" or pr["state"] == state.upper())]
    save()
    print(json.dumps(out))
elif args[:2] == ["pr", "merge"]:
    pr = next(p for p in st["prs"] if p["number"] == int(args[2]))
    if opt("--match-head-commit") != oid(pr["headRefName"]):
        save()
        sys.exit("head moved")
    with tempfile.TemporaryDirectory() as d:
        git("clone", "-q", st["bare"], d)
        git("merge", "-q", "--no-ff", "-m", "Merge " + pr["headRefName"], "origin/" + pr["headRefName"], cwd=d)
        git("push", "-q", "origin", "main", cwd=d)
        pr["mergeCommit"] = {"oid": git("rev-parse", "HEAD", cwd=d)}
    pr["state"] = "MERGED"
    st["merged"].append(pr["number"])
    save()
else:
    save()
    sys.exit("fake gh: unsupported %r" % args)
'''


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def write(root: Path, files: dict):
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())


def tree(root: Path) -> dict:
    """root 下全部文件（不含 .git）→ 内容。"""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(root).parts}


class World:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.bare = tmp / "github.git"
        self.gpfs, self.gpfs_wt = tmp / "gpfs/moon-exp", tmp / "gpfs/wt"
        self.a6000, self.a6000_wt = tmp / "a6000/moon-exp", tmp / "a6000/wt"
        self.local = tmp / "local"
        git("init", "-q", "--bare", str(self.bare))
        git("-C", str(self.bare), "symbolic-ref", "HEAD", "refs/heads/main")
        seed = tmp / "seed"
        git("init", "-q", str(seed))
        git("-C", str(seed), "symbolic-ref", "HEAD", "refs/heads/main")
        write(seed, {
            "scripts/wt.sh": (REPO / "scripts/wt.sh").read_bytes(),
            ".gitignore": (REPO / ".gitignore").read_bytes(),
            "AGENTS.md": "规矩\n",
            "workbench/util.py": "x = 1\n",
            "moonlib/ransac.py": "y = 1\n",
            "runs/canvas.json": "{}\n",
            "runs/E1/exp.toml": "title = 'E1'\n",
            "runs/E1/notes.md": "# E1\n",
            "runs/E1/metrics.json": "{}\n",
        })
        git("-C", str(seed), "add", "-A")
        git("-C", str(seed), "commit", "-q", "-m", "init")
        git("-C", str(seed), "push", "-q", str(self.bare), "main")
        git("clone", "-q", str(self.bare), str(self.gpfs))
        git("clone", "-q", str(self.gpfs), str(self.a6000))   # A6000 的 remote 是 gpfs 主 checkout
        git("clone", "-q", str(self.bare), str(self.local))
        self.gpfs_wt.mkdir(parents=True)
        self.a6000_wt.mkdir(parents=True)
        self.state = tmp / "gh.json"
        self.state.write_text(json.dumps({"bare": str(self.bare), "prs": [], "calls": [], "merged": []}))
        (tmp / "gh.py").write_text(FAKE_GH)

    def ticket(self, branch, files, big=None, *, on="gpfs", start="origin/main", pr=True):
        """在 gpfs（或 A6000）上开 worktree、提交 files、写不进 git 的 big，推到 GitHub，开 PR。返回 worktree 路径。"""
        main, wts = (self.gpfs, self.gpfs_wt) if on == "gpfs" else (self.a6000, self.a6000_wt)
        d = wts / branch
        git("-C", str(main), "fetch", "-q", "origin")
        git("-C", str(main), "worktree", "add", "-q", "-b", branch, str(d), start)
        write(d, files)
        git("-C", str(d), "add", "-A")
        git("-C", str(d), "commit", "-q", "-m", branch)
        write(d, big or {})
        git("-C", str(d), "push", "-q", "origin", branch)
        if on == "a6000":   # A6000 推到 gpfs 主 checkout，再从 126 推到 GitHub
            git("-C", str(self.gpfs), "push", "-q", "origin", f"{branch}:{branch}")
        if pr:
            st = self.gh_state()
            n = 100 + len(st["prs"])
            st["prs"].append({"number": n, "state": "OPEN", "headRefName": branch, "baseRefName": "main",
                              "mergeCommit": None, "title": branch})
            self.state.write_text(json.dumps(st))
        return d

    def gh_state(self):
        return json.loads(self.state.read_text())

    def main_has(self, branch):
        tip = git("-C", str(self.bare), "rev-parse", f"refs/heads/{branch}")
        return subprocess.run(["git", "-C", str(self.bare), "merge-base", "--is-ancestor", tip, "main"]).returncode == 0

    def close(self, branch, *flags, test_cmd="true", relay="bash -c"):
        sites = close_ticket.Sites(
            local=self.local, gpfs_shell=SHELL, gpfs_main=str(self.gpfs), gpfs_wt=str(self.gpfs_wt),
            a6000_shell=SHELL, a6000_main=str(self.a6000), a6000_wt=str(self.a6000_wt),
            relay=relay, gh=[sys.executable, str(self.tmp / "gh.py")], test_cmd=test_cmd)
        return close_ticket.main([branch, *flags], sites=sites)


@pytest.fixture
def w(tmp_path, monkeypatch):
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "t")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "t@t")
    monkeypatch.setenv("FAKE_GH_STATE", str(tmp_path / "gh.json"))
    return World(tmp_path)


E2 = {"runs/E2/exp.toml": "title = 'E2'\n", "runs/E2/notes.md": "# E2\n"}
E2_BIG = {"runs/E2/preds/m/val_matches.npz": b"npz" * 10, "runs/E2/ckpt/m/last.ckpt": b"ckpt" * 100}


def test_happy_path_merges_moves_artifacts_removes_worktree_and_syncs(w, capsys):
    wt = w.ticket("t-e2", E2, E2_BIG)
    code = w.close("t-e2", test_cmd="false")   # 只改 runs/：不跑测试
    out = capsys.readouterr().out
    assert code == 0, out
    assert w.main_has("t-e2")
    assert not wt.exists()
    assert (w.gpfs / "runs/E2/ckpt/m/last.ckpt").read_bytes() == E2_BIG["runs/E2/ckpt/m/last.ckpt"]
    assert git("-C", str(w.local), "rev-parse", "HEAD") == git("-C", str(w.bare), "rev-parse", "main")
    assert (w.local / "runs/E2/notes.md").exists()
    assert (w.local / "runs/E2/preds/m/val_matches.npz").read_bytes() == E2_BIG["runs/E2/preds/m/val_matches.npz"]
    assert not (w.local / "runs/E2/ckpt").exists()   # sync 永不拉 ckpt
    assert "可在工作台查看：E2" in out


def test_rerun_after_done_is_harmless(w, capsys):
    w.ticket("t-e2", E2, E2_BIG)
    assert w.close("t-e2") == 0
    assert w.close("t-e2") == 0, capsys.readouterr().out
    assert w.gh_state()["merged"] == [100]


def test_gpfs_main_has_same_name_different_size_stops_and_deletes_nothing(w, capsys):
    wt = w.ticket("t-e2", E2, E2_BIG)
    write(w.gpfs, {"runs/E2/ckpt/m/last.ckpt": b"other"})
    before_wt, before_main = tree(wt), tree(w.gpfs)
    code = w.close("t-e2")
    out = capsys.readouterr().out
    assert code == 1, out
    assert "大小不同" in out
    assert tree(wt) == before_wt
    after = tree(w.gpfs)
    assert all(after.get(p) == v for p, v in before_main.items())   # pull 只会多出文件，不会删、不会改
    assert after["runs/E2/ckpt/m/last.ckpt"] == b"other"


def test_a6000_ticket_copies_back_via_relay_then_removes_a6000_copy(w, capsys):
    wt = w.ticket("t-e2", E2, E2_BIG, on="a6000")
    code = w.close("t-e2")
    out = capsys.readouterr().out
    assert code == 0, out
    assert w.main_has("t-e2")
    assert not wt.exists()
    for p, v in E2_BIG.items():
        assert (w.gpfs / p).read_bytes() == v
        assert not (w.a6000 / p).exists()
    assert (w.a6000 / "runs/E2/notes.md").exists()   # 进 git 的不动
    assert (w.local / "runs/E2/preds/m/val_matches.npz").exists()


def test_a6000_copy_mismatch_stops_and_keeps_a6000_copy(w, capsys):
    w.ticket("t-e2", E2, E2_BIG, on="a6000")
    bad = w.tmp / "bad_relay.sh"   # 收下后把一个文件截短，冒充传坏了
    bad.write_text(f'bash -c "$1"; truncate -s 1 {w.gpfs}/runs/E2/ckpt/m/last.ckpt\n')
    code = w.close("t-e2", relay=f"bash {bad}")
    out = capsys.readouterr().out
    assert code == 1, out
    assert "核对不一致" in out
    for p, v in E2_BIG.items():
        assert (w.a6000 / p).read_bytes() == v


def test_local_notes_canvas_and_new_experiment_committed_before_pull(w, capsys):
    w.ticket("t-e2", E2, E2_BIG)
    write(w.local, {"runs/E1/notes.md": "# E1\n用户改过\n", "runs/canvas.json": '{"x": 1}\n',
                    "runs/E9/notes.md": "# E9\n", "runs/E9/exp.toml": "title = 'E9'\n"})
    code = w.close("t-e2")
    out = capsys.readouterr().out
    assert code == 0, out
    assert git("-C", str(w.local), "status", "--porcelain") == ""
    assert git("-C", str(w.local), "rev-parse", "HEAD") == git("-C", str(w.bare), "rev-parse", "main")
    msg = git("-C", str(w.bare), "log", "-1", "--format=%s", "main")
    assert msg.startswith("用户编辑：") and "E1 Notes" in msg and "画布" in msg and "E9" in msg
    assert w.main_has("t-e2")
    assert (w.gpfs / "runs/E1/notes.md").read_text(encoding="utf-8") == "# E1\n用户改过\n"   # 也进了 gpfs


def test_local_other_uncommitted_change_stops_before_merge(w, capsys):
    w.ticket("t-e2", E2, E2_BIG)
    write(w.local, {"workbench/util.py": "x = 2\n"})
    code = w.close("t-e2")
    out = capsys.readouterr().out
    assert code == 1, out
    assert "workbench/util.py" in out
    assert not w.main_has("t-e2")
    assert (w.local / "workbench/util.py").read_text() == "x = 2\n"


def test_shared_code_failing_tests_do_not_merge(w, capsys):
    w.ticket("t-code", {"workbench/util.py": "x = 3\n", **E2})
    code = w.close("t-code", test_cmd="false")
    out = capsys.readouterr().out
    assert code == 1, out
    assert "测试失败" in out
    assert not w.main_has("t-code")
    assert w.gh_state()["merged"] == []


def test_shared_code_tests_run_in_worktree_then_merge(w, capsys):
    wt = w.ticket("t-code", {"workbench/util.py": "x = 3\n", **E2})
    code = w.close("t-code", test_cmd=f"test \"$PWD\" = {wt} && grep -q 3 workbench/util.py")
    assert code == 0, capsys.readouterr().out
    assert w.main_has("t-code")


@pytest.mark.parametrize("files", [
    {"AGENTS.md": "新规矩\n"},
    {"moonlib/ransac.py": "y = 2\n"},
    {"runs/E1/metrics.json": '{"auc": 1}\n'},
])
def test_needs_approval_stops_without_merge_until_confirmed(w, capsys, files):
    w.ticket("t-rule", files)
    code = w.close("t-rule")
    out = capsys.readouterr().out
    assert code == 2, out
    assert "--confirm" in out
    assert not w.main_has("t-rule")
    assert w.close("t-rule", "--confirm") == 0, capsys.readouterr().out
    assert w.main_has("t-rule")


def test_dependent_prs_merged_first_in_order(w, capsys):
    w.ticket("t-a", {"runs/E3/notes.md": "# E3\n"})
    w.ticket("t-b", {"runs/E4/notes.md": "# E4\n"}, start="t-a")
    w.ticket("t-c", {"runs/E5/notes.md": "# E5\n"}, start="t-b")
    code = w.close("t-c")
    out = capsys.readouterr().out
    assert code == 0, out
    assert w.gh_state()["merged"] == [100, 101, 102]
    assert all(w.main_has(b) for b in ("t-a", "t-b", "t-c"))
    assert not any((w.gpfs_wt / b).exists() for b in ("t-a", "t-b", "t-c"))
    assert "可在工作台查看：E3 E4 E5" in out
