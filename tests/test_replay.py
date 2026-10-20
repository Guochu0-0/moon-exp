"""旧配置重放：用当前的配置展开逻辑重新展开每个已跑训练的配置文件，与启动记录里存的完整配置逐项对比。

守的是「新行为只能通过新增参数开启，默认保持旧行为」：改了某个默认值、改了补算规则，旧实验的配置在今天展开出来
就和当年不一样，这里变红。纳入的是 `scripts/finetune/run.py` 写的启动记录（RECORDS.md「启动记录」）：
入口为训练驱动、记了 `config_file` 与 `args`；补记（backfilled）的不纳入。配置文件按记录里的 commit 取出，
这样事后改配置文件不算差异，只比较代码的展开逻辑。

历史上合法的差异逐条列在 ALLOWED 里，每条写明原因。新增参数（旧记录里没有的键）一律视为差异，
须逐条登记：新参数的默认值必须等于旧行为，登记时就要核对这一点。
"""
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENTRY = "scripts/finetune/run.py"

# (配置文件, 键路径) → 原因。键路径写 "表.键"；整张表缺失或多出时写 "表"。
ALLOWED = {
}


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def records():
    """所有纳入重放的启动记录：(记录文件, 序号, 记录)。"""
    out = []
    for f in sorted((REPO / "runs").glob("*/launch/*.json")):
        for i, r in enumerate(json.loads(f.read_text(encoding="utf-8"))):
            if r.get("entry") == ENTRY and r.get("config_file") and r.get("args") and not r.get("backfilled"):
                out.append((f.relative_to(REPO).as_posix(), i, r))
    return out


def _configs_at(commit: str, config_file: str, dst: Path) -> Path:
    """把 commit 时 config_file 所在目录的全部 .toml 写到 dst（base 只在同目录内继承）。"""
    d = config_file.rsplit("/", 1)[0]
    for p in _git("ls-tree", "--name-only", commit, "--", d + "/").split():
        if p.endswith(".toml"):
            (dst / p.rsplit("/", 1)[1]).write_text(_git("show", f"{commit}:{p}"), encoding="utf-8")
    return dst / config_file.rsplit("/", 1)[1]


def _flat(cfg: dict) -> dict:
    return {f"{t}.{k}": v for t, tab in cfg.items() for k, v in tab.items()}


def diff(old: dict, new: dict) -> dict:
    """逐项差异：键路径 → (记录里的值, 今天展开的值)；缺失写 "<缺>"。整张表多出或缺失按表报一条。"""
    out = {}
    for t in sorted(set(old) | set(new)):
        if t not in old or t not in new:
            out[t] = ("<缺>" if t not in old else "<有>", "<缺>" if t not in new else "<有>")
    a = _flat({t: v for t, v in old.items() if t in new})
    b = _flat({t: v for t, v in new.items() if t in old})
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<缺>"), b.get(k, "<缺>")
        if va != vb:
            out[k] = (va, vb)
    return out


RECORDS = records()


def test_replay_has_records():
    assert len(RECORDS) >= 80          # 2026-10-20 时有 89 条；变少说明筛选条件或记录格式坏了


@pytest.mark.parametrize("src,i,rec", RECORDS, ids=[f"{s}#{i}" for s, i, _ in RECORDS])
def test_replay_old_config(src, i, rec, tmp_path):
    from finetune import config as C

    cfg = C.load(_configs_at(rec["commit"], rec["config_file"], tmp_path))
    d = diff(rec["args"], cfg)
    bad = {k: v for k, v in d.items() if (rec["config_file"], k) not in ALLOWED}
    assert not bad, f"{rec['config_file']}（commit {rec['commit'][:9]}）今天展开与启动记录不同：{bad}"


def test_allowed_entries_still_needed():
    """白名单里每条都要对应一个今天仍存在的差异，修好了就删掉，不留空条目。"""
    from finetune import config as C
    import tempfile

    seen = set()
    for _, _, rec in RECORDS:
        with tempfile.TemporaryDirectory() as t:
            cfg = C.load(_configs_at(rec["commit"], rec["config_file"], Path(t)))
        seen |= {(rec["config_file"], k) for k in diff(rec["args"], cfg)}
    assert not set(ALLOWED) - seen, sorted(set(ALLOWED) - seen)


def test_diff_reports_changed_missing_and_extra():
    old = {"optim": {"lr": 1e-5, "wd": 0.0}, "neg": {}}
    new = {"optim": {"lr": 3e-5, "wd": 0.0, "ema": 0.0}, "aug": {"geo": True}}
    assert diff(old, new) == {"aug": ("<缺>", "<有>"), "neg": ("<有>", "<缺>"),
                              "optim.lr": (1e-5, 3e-5), "optim.ema": ("<缺>", 0.0)}
