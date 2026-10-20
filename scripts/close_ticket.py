"""结票收尾（#173「收尾脚本」）：把一张票从「PR 已开」带到「工作台上能看」。在用户电脑上运行，只做 git、ssh、gh 与文件传输。

    python scripts/close_ticket.py <分支> [--confirm] [--skip-a6000]

步骤、停下的情形与退出码的含义见 .claude/skills/server/closing.md。需用户点头的类别是下面的 EVAL_PATHS、RULE_PATHS
与 Closer.closed_record_reasons；合并前跑的全量测试是 TEST_CMD。

测试时向 main() 传 sites=Sites(...)：ssh 换成本地 shell，gh 换成替身，路径换成临时目录（同 workbench sync 的 shell 注入）。
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from workbench import sync as wb_sync  # noqa: E402

YGC = "/remote-home/xufang/YGC"   # A6000 上有同名软链指向它自己的 NFS（.claude/skills/server/a6000.md）

# 需用户点头的类别（#173「收尾脚本」第 1 步）。评测口径：算 metrics 的代码（workbench eval 经 evaluate → protocol、
# perturb、dataset 读检查点）与 moonlib；reference.py、inter.py 只管显示，不在内。
EVAL_PATHS = ("moonlib/", "workbench/evaluate.py", "workbench/protocol.py", "workbench/perturb.py", "workbench/dataset.py")
RULE_PATHS = ("AGENTS.md", "CLAUDE.md", "CONTEXT.md", "CODING_STANDARDS.md", "docs/agents/", "docs/adr/", ".claude/skills/")
RECORD = re.compile(r"runs/([^/]+)/(notes\.md|metrics\.json)")
NOTES = re.compile(r"runs/([^/]+)/notes\.md")
PR_MERGE = re.compile(r"Merge pull request #(\d+) from [^/\s]+/(\S+)")
CANVAS = "runs/canvas.json"
PR_FIELDS = "number,state,headRefName,headRefOid,baseRefName,mergeCommit,title,closingIssuesReferences"

# 合并前的全量测试，在 PR 的服务器 worktree 里跑。两段都过才合并：
# 工作台环境（py3.11，无 torch）跑大部分；训练环境（有 torch）跑 import torch 的训练测试与 py3.8 检查。
# 训练环境里要装 pytest（是否装、装在哪由用户定）；装好前可临时 PYTHONPATH 指向单独 pip --target 的目录。
WB_PYTHON = "/opt/envs/wb/bin/python"
TORCH_PYTHON = "/opt/envs/loftr/bin/python"
TORCH_TESTS = ("tests/test_finetune.py", "tests/test_replay.py", "tests/test_py38.py")
TEST_CMD = ("export CUDA_VISIBLE_DEVICES=; "
            f"timeout 3600 {WB_PYTHON} -m pytest -q -p no:cacheprovider tests "
            + " ".join(f"--ignore={t}" for t in TORCH_TESTS) + "; wb=$?; "
            f"timeout 3600 {TORCH_PYTHON} -m pytest -q -p no:cacheprovider {' '.join(TORCH_TESTS)}; torch=$?; "
            'echo "close-tests: wb=$wb torch=$torch"; [ $wb -eq 0 ] && [ $torch -eq 0 ]')


@dataclass
class Sites:
    local: Path = REPO
    gpfs_shell: list = field(default_factory=lambda: ["ssh", "xufang126内网"])   # 126 能连 GitHub
    gpfs_main: str = f"{YGC}/moon-exp"
    gpfs_wt: str = f"{YGC}/wt"
    a6000_shell: list = field(default_factory=lambda: ["ssh", "xufangA6000内网"])
    a6000_main: str = f"{YGC}/moon-exp"
    a6000_wt: str = f"{YGC}/wt"
    relay: str = "ssh -i /root/.ssh/ygc_154 -p 20128 root@10.254.1.150"   # 在 A6000 上执行，接收端是 gpfs（150）
    gh: list = field(default_factory=lambda: ["gh"])
    test_cmd: str = TEST_CMD


class Stop(Exception):
    def __init__(self, what: str, advice: str, code: int = 1):
        super().__init__(what)
        self.what, self.advice, self.code = what, advice, code


@dataclass
class Ticket:
    branch: str
    pr: dict
    tip: str
    base: str     # 与 main 的分叉点
    files: list   # [(状态, 路径)]，相对分叉点

    @property
    def merged(self) -> bool:
        return self.pr["state"] == "MERGED"

    @property
    def closes(self) -> set:
        return {i["number"] for i in self.pr.get("closingIssuesReferences") or []}

    @property
    def ids(self) -> list:
        return sorted({p.split("/")[1] for _, p in self.files if p.startswith("runs/") and p.count("/") >= 2})

    @property
    def shared(self) -> list:
        return [p for _, p in self.files if not p.startswith("runs/")]


class Closer:
    def __init__(self, s: Sites, confirm: bool, skip_a6000: bool):
        self.s, self.confirm, self.skip_a6000 = s, confirm, skip_a6000
        self.done: list[str] = []
        self.on_a6000: dict[str, bool] = {}   # 分支 → A6000 上有没有它的 worktree（预检时查）

    # ---- 底层调用 ----
    def log(self, msg: str):
        print(f"close: {msg}", flush=True)

    def step(self, msg: str):
        self.done.append(msg)
        self.log(msg)

    def _run(self, what: str, cmd: list, **kw) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(cmd, capture_output=True, **kw)
        except subprocess.TimeoutExpired:
            raise Stop(f"{what} 超时", "检查网络或远端状态后重跑")

    def git_rc(self, *args, timeout=300) -> tuple[int, str]:
        r = self._run(f"本地 git {' '.join(args)}", ["git", "-c", "core.quotepath=false", *args],
                      cwd=self.s.local, timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace")

    def git(self, *args, timeout=300) -> str:
        r = self._run(f"本地 git {' '.join(args)}", ["git", "-c", "core.quotepath=false", *args],
                      cwd=self.s.local, timeout=timeout)
        if r.returncode:
            raise Stop(f"本地 git {' '.join(args)} 失败：{r.stderr.decode('utf-8', 'replace').strip()}", "看上面的错误，处理后重跑")
        return r.stdout.decode("utf-8", "replace")

    def remote(self, where: str, script: str, *, input: str | None = None, timeout=600) -> tuple[int, str]:
        shell = self.s.gpfs_shell if where == "gpfs" else self.s.a6000_shell
        try:
            r = subprocess.run([*shell, script], input=None if input is None else input.encode("utf-8"),
                               capture_output=True, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            raise Stop(f"{where} 上的命令超时：{script}", "远端进程可能已半途退出，先用只读命令查看现场再重跑")
        out = (r.stdout + r.stderr).decode("utf-8", "replace").strip()
        if r.returncode == 255:   # ssh 连不上时的退出码；远端命令本身不用 255
            raise Stop(f"连不上 {where}（{' '.join(shell)}）：{out}",
                       "按 .claude/skills/server/machines.md「怎么连」检查工位机与 Tailscale 后重跑"
                       + ("；若本票确实没在 A6000 跑，可带 --skip-a6000 重跑" if where == "a6000" else ""))
        return r.returncode, out

    def must(self, where: str, script: str, what: str, advice: str, **kw) -> str:
        rc, out = self.remote(where, script, **kw)
        if rc:
            raise Stop(f"{what}（{where}，退出码 {rc}）：\n{out}", advice)
        return out

    def gh(self, *args) -> str:
        r = self._run(f"gh {' '.join(args)}", [*self.s.gh, *args], cwd=self.s.local, timeout=300)
        if r.returncode:
            raise Stop(f"gh {' '.join(args)} 失败：{(r.stdout + r.stderr).decode('utf-8', 'replace').strip()}",
                       "看上面的错误，处理后重跑")
        return r.stdout.decode("utf-8")

    def gh_json(self, *args):
        out = self.gh(*args)
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            raise Stop(f"gh {' '.join(args)} 的输出不是 JSON：{out[:500]}", "检查 gh 是否登录、版本是否支持这些字段后重跑")

    # ---- 本地仓库 ----
    def local_changes(self) -> list[tuple[str, str]]:
        raw = self.git("status", "--porcelain=v1", "-z", "--untracked-files=all").split("\0")
        out, i = [], 0
        while i < len(raw):
            e = raw[i]
            i += 1
            if not e:
                continue
            st, path = e[:2], e[3:]
            if st[0] in "RC":
                i += 1   # -z 下原路径另占一项
            out.append((st, path))
        return out

    def classify_local(self, changes) -> tuple[list[str], list[str]]:
        """返回（提交说明的各部分，不该替用户提交的路径）。"""
        parts, other = [], []
        for st, path in changes:
            if "D" in st:
                other.append(path)
            elif st == "??" and path.startswith("runs/") and path.count("/") >= 2 \
                    and not self.git("ls-files", "--", "runs/" + path.split("/")[1]).strip():
                parts.append(f"{path.split('/')[1]} 新实验目录")
            elif path == CANVAS:
                parts.append("画布")
            elif m := NOTES.fullmatch(path):
                parts.append(f"{m[1]} Notes")
            else:
                other.append(path)
        return list(dict.fromkeys(parts)), other

    def check_local(self):
        br = self.git("rev-parse", "--abbrev-ref", "HEAD").strip()
        if br != "main":
            raise Stop(f"本地仓库 {self.s.local} 不在 main 上（在 {br}）", "切回 main 后重跑")
        _, other = self.classify_local(self.local_changes())
        if other:
            raise Stop("本地有 Notes、画布、未跟踪实验目录以外的改动，不替用户提交：\n  " + "\n  ".join(other),
                       "请用户决定这些改动（提交、挪走或丢弃）后重跑")

    def check_local_pushed(self):
        n = self.git("rev-list", "--count", "origin/main..HEAD").strip()
        if n != "0":
            raise Stop(f"本地 main 有 {n} 个未推送的提交", "请用户确认这些提交（推送或撤回）后重跑")

    # ---- GitHub ----
    def find_pr(self, branch: str) -> dict:
        prs = self.gh_json("pr", "list", "--head", branch, "--state", "all", "--json", PR_FIELDS)
        for state in ("OPEN", "MERGED"):
            for pr in prs:
                if pr["state"] == state:
                    return pr
        raise Stop(f"分支 {branch} 没有开着或已合并的 PR", "先开 PR 再重跑")

    def ticket(self, branch: str, pr: dict) -> Ticket:
        tip = pr["headRefOid"]
        if pr["state"] == "OPEN":
            if pr["baseRefName"] != "main":
                raise Stop(f"PR #{pr['number']}（{branch}）的目标分支是 {pr['baseRefName']}，不是 main",
                           "把 PR 的目标改成 main（gh pr edit --base main）后重跑")
            rc, remote_tip = self.git_rc("rev-parse", "-q", "--verify", f"origin/{branch}")
            remote_tip = remote_tip.strip()
            if rc:
                raise Stop(f"本地看不到 origin/{branch}", "确认分支已推送到 GitHub 后重跑")
            if remote_tip != tip:
                raise Stop(f"PR #{pr['number']} 的 head {tip[:8]} 与本地看到的 origin/{branch} {remote_tip[:8]} 不一致",
                           "稍后重跑（可能刚有推送）")
            main = "origin/main"
        else:
            main = pr["mergeCommit"]["oid"] + "^1"
        base = self.git("merge-base", main, tip).strip()
        files = []
        for line in self.git("diff", "--name-status", "--no-renames", base, tip).splitlines():
            st, path = line.split("\t", 1)
            files.append((st, path))
        return Ticket(branch, pr, tip, base, files)

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.git_rc("merge-base", "--is-ancestor", a, b)[0] == 0

    def dependencies(self, target: Ticket) -> list[Ticket]:
        prs = self.gh_json("pr", "list", "--state", "open", "--limit", "200", "--json", PR_FIELDS)
        deps = []
        for pr in prs:
            oid = pr["headRefOid"]
            if pr["headRefName"] == target.branch or self.git_rc("cat-file", "-e", oid)[0]:
                continue
            if self.is_ancestor(oid, target.tip) and not self.is_ancestor(oid, "origin/main"):
                deps.append(pr)
        # 先合并的在前：按「自己包含了几个其他依赖」升序，祖先排在后代前面
        anc = {p["number"]: sum(self.is_ancestor(q["headRefOid"], p["headRefOid"]) for q in deps if q is not p)
               for p in deps}
        return [self.ticket(p["headRefName"], p) for p in sorted(deps, key=lambda p: anc[p["number"]])]

    def closed_record_reasons(self, t: Ticket) -> list[str]:
        """改了已结票实验的 notes / metrics：分叉点上已有这份记录，且把这个实验带进 main 的 PR 里有不属于本票的。
        「属于本票」= 与本 PR 同一分支名，或 Closes 的 issue 落在本 PR 的 Closes 里。实验的 PR 从 main 的
        first-parent 历史里的「Merge pull request #N from …/分支」读；一个也读不到（如直接提交进 main）时按需点头算。"""
        def own(n: int, br: str) -> bool:
            if br == t.branch:
                return True
            if not t.closes:
                return False
            refs = self.gh_json("pr", "view", str(n), "--json", "closingIssuesReferences").get("closingIssuesReferences")
            theirs = {i["number"] for i in refs or []}
            return bool(theirs) and theirs <= t.closes

        owners: dict[str, tuple] = {}   # 实验 id → （带进 main 的 PR 号，其中不属于本票的）
        out = []
        for st, p in t.files:
            m = RECORD.fullmatch(p)
            if not m or st == "A":
                continue
            rid = m[1]
            if rid not in owners:
                log = self.git("log", "--first-parent", "--format=%s", t.base, "--", f"runs/{rid}/")
                prs = [(int(g[1]), g[2]) for g in map(PR_MERGE.match, log.splitlines()) if g]
                owners[rid] = (prs, [n for n, br in prs if not own(n, br)])
            prs, foreign = owners[rid]
            if not prs or foreign:
                why = f"由 PR {'、'.join(f'#{n}' for n in foreign)} 建立" if foreign else "不是经 PR 进的 main"
                out.append(f"改了已结票实验的记录：{p}（{rid} {why}）")
        return out

    def approval_reasons(self, t: Ticket) -> list[str]:
        out = []
        for _, p in t.files:
            if p.startswith(EVAL_PATHS):
                out.append(f"改了评测口径：{p}")
            elif p in RULE_PATHS or p.startswith(tuple(r for r in RULE_PATHS if r.endswith("/"))):
                out.append(f"改了规矩文件：{p}")
        return out + self.closed_record_reasons(t)

    # ---- 服务器 ----
    def wt_exists(self, where: str, branch: str) -> bool:
        root = self.s.gpfs_wt if where == "gpfs" else self.s.a6000_wt
        rc, _ = self.remote(where, f"test -d {shlex.quote(root + '/' + branch)}", timeout=60)
        return rc == 0

    def probe_a6000(self, tickets):
        """合并前只读地查 A6000：各分支在那里有没有 worktree。连不上就在合并前停下。"""
        if self.skip_a6000:
            self.log("--skip-a6000：不查 A6000")
            return
        self.on_a6000 = {t.branch: self.wt_exists("a6000", t.branch) for t in tickets}
        on = [b for b, yes in self.on_a6000.items() if yes]
        self.log(f"A6000 上有 worktree 的分支：{' '.join(on) or '无'}")

    def run_tests(self, target: Ticket):
        for where in ("gpfs", "a6000"):
            if where == "a6000" and (self.skip_a6000 or not self.on_a6000.get(target.branch)):
                continue
            if where == "gpfs" and not self.wt_exists(where, target.branch):
                continue
            d = (self.s.gpfs_wt if where == "gpfs" else self.s.a6000_wt) + "/" + target.branch
            head = self.must(where, f"git -C {shlex.quote(d)} rev-parse HEAD", "读 worktree 的 HEAD 失败", "检查 worktree").strip()
            if head != target.tip:
                raise Stop(f"{where} 上的 worktree {d} 停在 {head[:8]}，不是 PR 的 head {target.tip[:8]}",
                           "在 worktree 里推送或 pull 到与 PR 一致后重跑")
            self.log(f"在 {where} 的 {d} 跑测试：{self.s.test_cmd}")
            rc, out = self.remote(where, f"cd {shlex.quote(d)} && {self.s.test_cmd}", timeout=7300)
            if rc:
                raise Stop(f"测试失败（退出码 {rc}），未合并：\n{out[-4000:]}", "修好后推送再重跑")
            self.step(f"测试通过（{where}：{d}）")
            return
        raise Stop(f"改了共享代码，但 gpfs{'' if self.skip_a6000 else ' 与 A6000'} 上都没有 {target.branch} 的 worktree，没法跑测试",
                   f"scripts/wt.sh new {target.branch} 后重跑")

    def pull_main(self, where: str):
        main = self.s.gpfs_main if where == "gpfs" else self.s.a6000_main
        self.must(where, f"timeout 300 git -C {shlex.quote(main)} pull -q --ff-only",
                  f"{main} pull --ff-only 失败", "主 checkout 只做 pull --ff-only；查看它为什么不能快进，处理后重跑")
        self.step(f"{where} 主 checkout 已 pull")

    def wt_close(self, where: str, branch: str):
        main, wt = (self.s.gpfs_main, self.s.gpfs_wt) if where == "gpfs" else (self.s.a6000_main, self.s.a6000_wt)
        if not self.wt_exists(where, branch):
            self.log(f"{where} 上没有 {branch} 的 worktree，跳过 close")
            return False
        cmd = f"MAIN={shlex.quote(main)} WT={shlex.quote(wt)} bash {shlex.quote(main + '/scripts/wt.sh')} close {shlex.quote(branch)}"
        for dry in (" --dry-run", ""):
            out = self.must(where, cmd + dry, f"wt.sh close{dry} 停下", "按 wt.sh 的提示处理现场（它没删任何东西）后重跑", timeout=1800)
            self.log(out)
        self.step(f"{where} 上的 worktree {wt}/{branch} 已关")
        return True

    def untracked(self, where: str, root: str, ids, *, ignored: bool) -> dict[str, int]:
        """root 下本票实验 runs/<id>/ 里未跟踪的文件 → 大小；ignored 选被 gitignore 的或不被 gitignore 的。
        .claims 是各机器自己的占位，不算。"""
        paths = " ".join(shlex.quote(f"runs/{i}") for i in ids)
        out = self.must(where, f"cd {shlex.quote(root)} && git ls-files -z --others{' --ignored' if ignored else ''}"
                               f" --exclude-standard -- {paths} | xargs -0 -r stat --printf '%s\\t%n\\n'",
                        f"列 {where} 上 {root} 的未跟踪文件失败", f"检查 {where} 上的 {root}")
        files = {}
        for line in out.splitlines():
            size, _, p = line.partition("\t")
            if p and "/.claims/" not in p:
                files[p] = int(size)
        return files

    def check_untracked(self, tickets):
        """合并前查各分支的 gpfs 与 A6000 worktree：本票实验里未跟踪、又不被 gitignore 的文件多半是忘了 git add，
        wt.sh close 会连 worktree 一起删掉它们。有就停下。"""
        loose = []
        for t in tickets:
            if not t.ids:
                continue
            for where, root in (("gpfs", self.s.gpfs_wt), ("a6000", self.s.a6000_wt)):
                if where == "a6000" and not self.on_a6000.get(t.branch):
                    continue
                if where == "gpfs" and not self.wt_exists("gpfs", t.branch):
                    continue
                d = f"{root}/{t.branch}"
                loose += [f"{where}：{d}/{p}" for p in sorted(self.untracked(where, d, t.ids, ignored=False))]
        if loose:
            raise Stop("worktree 里有本票实验未提交、也不被 gitignore 的文件（忘了 git add？），什么都没合并、没删：\n  "
                       + "\n  ".join(loose),
                       "在 worktree 里把它们提交到分支并推送（不要的删掉）后重跑")

    def a6000_untracked(self, ids) -> dict[str, tuple[int, bool]]:
        """A6000 主 checkout 里本票实验未跟踪的文件 → （大小，是否被 gitignore）。"""
        files = {}
        for ignored in (True, False):
            for p, size in self.untracked("a6000", self.s.a6000_main, ids, ignored=ignored).items():
                files[p] = (size, ignored)
        return files

    def gpfs_sizes(self, paths) -> dict[str, int]:
        out = self.must("gpfs", f"cd {shlex.quote(self.s.gpfs_main)} && while IFS= read -r f; do "
                                f"[ -f \"$f\" ] && printf '%s\\t%s\\n' \"$(stat -c %s -- \"$f\")\" \"$f\"; done; true",
                        "核对 gpfs 上的文件失败", "检查 gpfs 主 checkout", input="".join(p + "\n" for p in paths))
        return {p: int(size) for size, _, p in (line.partition("\t") for line in out.splitlines()) if p}

    def a6000(self, t: Ticket):
        if self.skip_a6000:
            return
        had_wt = self.on_a6000.get(t.branch, False)
        if had_wt:
            self.pull_main("a6000")
            self.wt_close("a6000", t.branch)
        found = self.a6000_untracked(t.ids) if t.ids else {}
        loose = sorted(p for p, (_, ignored) in found.items() if not ignored)
        if loose:
            raise Stop("A6000 主 checkout 里有本票实验未提交、也不被 gitignore 的文件（该进 git 的记录？），什么都没拷、没删：\n  "
                       + "\n  ".join(loose),
                       "请用户决定：该进 git 的经 worktree 提交、开 PR；该忽略的补进 .gitignore；不要的删掉。处理后重跑")
        files = {p: size for p, (size, _) in found.items()}
        if not files:
            if had_wt:
                self.step("A6000 上没有要拷回的产物")
            return
        total = sum(files.values())
        self.log(f"A6000 → gpfs：{len(files)} 个文件，{total / 2**20:.1f} MiB")
        have = self.gpfs_sizes(files)
        clash = [p for p in files if p in have and have[p] != files[p]]
        if clash:
            raise Stop("gpfs 主 checkout 已有同名但大小不同的文件，没有拷，A6000 副本保留：\n  "
                       + "\n  ".join(f"{p}（A6000 {files[p]}，gpfs {have[p]}）" for p in clash),
                       "请用户判断留哪一份（可能是上次传了一半），处理后重跑")
        todo = [p for p in files if p not in have]
        if todo:
            recv = shlex.quote(f"tar xf - -C {shlex.quote(self.s.gpfs_main)}")
            self.must("a6000", f"cd {shlex.quote(self.s.a6000_main)} && timeout 7200 tar cf - -T - | {self.s.relay} {recv}",
                      "经 150 拷回 gpfs 失败，A6000 副本保留", "重跑（已拷完的会跳过；传了一半的会被当成大小不同而停下）",
                      input="".join(p + "\n" for p in todo), timeout=7200)
        got = self.gpfs_sizes(files)
        bad = [p for p in files if got.get(p) != files[p]]
        if bad:
            raise Stop(f"拷回后核对不一致（{len(bad)} / {len(files)} 个文件），A6000 副本保留：\n  "
                       + "\n  ".join(f"{p}（A6000 {files[p]}，gpfs {got.get(p, '缺')}）" for p in bad),
                       "查看 gpfs 上这些文件，处理后重跑")
        self.step(f"已拷回 gpfs 并核对：{len(files)} 个文件大小一致")
        self.must("a6000", f"cd {shlex.quote(self.s.a6000_main)} && while IFS= read -r f; do rm -f -- \"$f\"; done",
                  "删 A6000 副本失败（gpfs 上已有完整一份）", "手动删 A6000 上的这些文件", input="".join(p + "\n" for p in files))
        self.step("已删 A6000 上的副本")

    # ---- 本地收尾 ----
    def undo_auto_commit(self):
        """撤回刚替用户做的提交，改动回到工作区（未暂存），本地 main 回到原处，重跑不会被「未推送的提交」挡住。"""
        self.git("reset", "-q", "HEAD~1")

    def finish_local(self):
        self.check_local()
        parts, _ = self.classify_local(self.local_changes())
        self.git("fetch", "-q", "origin")
        self.check_local_pushed()
        if not parts:
            self.git("pull", "-q", "--ff-only")
            self.step("本地已 pull")
            return
        msg = "用户编辑：" + "、".join(parts)
        self.git("add", "-A", "--", "runs")
        self.git("commit", "-q", "-m", msg)
        self.step(f"已替用户提交：{msg}")
        if self.git_rc("rebase", "-q", "origin/main")[0]:
            self.git_rc("rebase", "--abort")
            self.undo_auto_commit()
            raise Stop("用户的编辑与 main 上的改动冲突：rebase 已撤回，替用户做的提交也已撤回，改动原样留在本地工作区（未提交）",
                       "请用户对照 origin/main 手动合并这些改动（如 git stash、git pull --ff-only、git stash pop 后解决冲突），再重跑")
        if self.git_rc("push", "-q", "origin", "main")[0]:
            self.undo_auto_commit()
            raise Stop("推送失败（不强推）：替用户做的提交已撤回，改动留在本地工作区（未提交）",
                       "稍后重跑，或查看远端是否又有新提交、是否拒收")
        self.step("已推送，本地与 origin/main 一致")
        self.pull_main("gpfs")

    def sync(self, ids):
        if not ids:
            return
        try:
            r = wb_sync.sync(ids, self.s.local / "runs", remote_root=self.s.gpfs_main, shell=self.s.gpfs_shell)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            raise Stop(f"workbench sync 失败：{e}", f"稍后在本地重跑 python -m workbench sync {' '.join(ids)}")
        self.step(f"已 sync：拉回 {len(r.fetched)} 个文件（{r.fetched_bytes / 2**20:.1f} MiB），跳过 {len(r.skipped)} 个")

    # ---- 总流程 ----
    def run(self, branch: str):
        self.check_local()
        self.git("fetch", "-q", "--prune", "origin")
        self.check_local_pushed()
        target = self.ticket(branch, self.find_pr(branch))
        tickets = ([] if target.merged else self.dependencies(target)) + [target]
        for t in tickets:
            self.log(f"PR #{t.pr['number']} {t.branch}：{'已合并' if t.merged else '待合并'}，实验 {' '.join(t.ids) or '无'}，"
                     f"共享代码 {len(t.shared)} 个文件")
        reasons = [f"#{t.pr['number']} {r}" for t in tickets if not t.merged for r in self.approval_reasons(t)]
        if reasons:
            if not self.confirm:
                raise Stop("这些改动需要用户点头才能合并：\n  " + "\n  ".join(reasons),
                           f"转问用户；同意后 python scripts/close_ticket.py {branch} --confirm", code=2)
            self.log("已带 --confirm：" + "；".join(reasons))
        self.probe_a6000(tickets)
        self.check_untracked(tickets)
        if any(t.shared for t in tickets if not t.merged):
            self.run_tests(target)   # target 包含全部依赖的提交
        for t in tickets:
            if not t.merged:
                self.gh("pr", "merge", str(t.pr["number"]), "--merge", "--match-head-commit", t.tip)
                self.step(f"已合并 PR #{t.pr['number']}（{t.branch}）")
            self.pull_main("gpfs")
            self.wt_close("gpfs", t.branch)
            self.a6000(t)
        self.finish_local()
        ids = sorted({i for t in tickets for i in t.ids})
        self.sync(ids)
        self.log("完成。可在工作台查看：" + (" ".join(ids) if ids else "（本票没有动实验目录）"))


def main(argv=None, *, sites: Sites | None = None) -> int:
    ap = argparse.ArgumentParser(description="结票收尾：合并 PR、收服务器 worktree、拷回 A6000 产物、提交本地编辑、同步大文件")
    ap.add_argument("branch")
    ap.add_argument("--confirm", action="store_true", help="用户已同意合并需点头类别的改动")
    ap.add_argument("--skip-a6000", action="store_true", help="不查 A6000（连不上且本票确实没在 A6000 跑时用）")
    a = ap.parse_args(argv)
    c = Closer(sites or Sites(), a.confirm, a.skip_a6000)
    try:
        c.run(a.branch)
        return 0
    except Stop as e:
        print(f"close: 停下：{e.what}", flush=True)
        print("close: 已完成：" + ("；".join(c.done) if c.done else "无"), flush=True)
        print(f"close: 建议：{e.advice}", flush=True)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
