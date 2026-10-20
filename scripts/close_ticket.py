"""结票收尾（#173「收尾脚本」）：把一张票从「PR 已开」带到「工作台上能看」。在用户电脑上运行，只做 git、ssh、gh 与文件传输。

    python scripts/close_ticket.py <分支> [--confirm] [--skip-a6000]

步骤：
1. 判断分支相对 main 改了什么。需用户点头的类别（评测口径、已结票实验的 metrics / Notes、规矩文件）停下并列出原因，
   用户同意后带 --confirm 重跑。
2. 分支包含其他未合并 PR 的提交时，这些 PR 按先后顺序一起处理。
3. 动了共享代码（runs/ 以外）时，在服务器 worktree 里跑全量测试；失败不合并。
4. gh 合并 PR（--match-head-commit，合并的正是测过的那个 commit）。
5. gpfs 主 checkout pull --ff-only；scripts/wt.sh close（先 --dry-run）。
6. A6000 上有本票的 worktree 或产物时：A6000 主 checkout pull、wt.sh close，把不进 git 的产物经 150 内网拷回 gpfs 主 checkout，
   逐文件核对大小，全部一致才删 A6000 上的这些文件。gpfs 已有同名不同大小的文件时不拷、停下。
7. 本地仓库：改动只有 runs/*/notes.md、runs/canvas.json 和未跟踪的实验目录时替用户提交、rebase 到 origin/main、推送，
   gpfs 主 checkout 再 pull 一次；有别的改动或本地有未推送的提交时停下。没改动时 pull --ff-only。
8. workbench sync 本票动过的实验；输出可在工作台查看的实验 id。

任何一步停下：不删文件、不强推，输出已完成的步骤、现状和建议，非零退出（1 停下，2 等用户点头）。重跑是安全的：
已合并的 PR 跳过合并，已删的 worktree 跳过 close。A6000 连不上而本票确实没在 A6000 跑时，带 --skip-a6000 重跑。

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
YGC = "/remote-home/xufang/YGC"   # A6000 上有同名软链指向它自己的 NFS（docs/agents/servers.md「A6000」）

# 需用户点头的类别（#173「收尾脚本」第 1 步）
EVAL_PATHS = ("moonlib/", "workbench/evaluate.py", "workbench/protocol.py")
RULE_PATHS = ("AGENTS.md", "CLAUDE.md", "CONTEXT.md", "CODING_STANDARDS.md", "docs/agents/", "docs/adr/", ".claude/skills/")
CLOSED_RECORD = re.compile(r"runs/[^/]+/(notes\.md|metrics\.json)")
NOTES = re.compile(r"runs/([^/]+)/notes\.md")
CANVAS = "runs/canvas.json"


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
    test_cmd: str = "CUDA_VISIBLE_DEVICES= timeout 3600 /opt/envs/wb/bin/python -m pytest -q tests"


class Stop(Exception):
    def __init__(self, what: str, advice: str, code: int = 1):
        super().__init__(what)
        self.what, self.advice, self.code = what, advice, code


@dataclass
class Ticket:
    branch: str
    pr: dict
    tip: str
    files: list   # [(状态, 路径)]，相对分叉点

    @property
    def merged(self) -> bool:
        return self.pr["state"] == "MERGED"

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

    # ---- 底层调用 ----
    def log(self, msg: str):
        print(f"close: {msg}", flush=True)

    def step(self, msg: str):
        self.done.append(msg)
        self.log(msg)

    def git(self, *args, check=True, timeout=300) -> str:
        r = subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=self.s.local, capture_output=True, timeout=timeout)
        out = r.stdout.decode("utf-8", "replace")
        if check and r.returncode:
            raise Stop(f"本地 git {' '.join(args)} 失败：{r.stderr.decode('utf-8', 'replace').strip()}", "看上面的错误，处理后重跑")
        return out if check else (r.returncode, out)

    def remote(self, where: str, script: str, *, input: str | None = None, timeout=600) -> tuple[int, str]:
        shell = self.s.gpfs_shell if where == "gpfs" else self.s.a6000_shell
        try:
            r = subprocess.run([*shell, script], input=None if input is None else input.encode("utf-8"),
                               capture_output=True, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            raise Stop(f"{where} 上的命令超时：{script}", "远端进程可能已半途退出，先用只读命令查看现场再重跑")
        out = (r.stdout + r.stderr).decode("utf-8", "replace").strip()
        if r.returncode == 255 and shell[0] == "ssh":
            raise Stop(f"连不上 {where}（{shell[1]}）：{out}",
                       "按 docs/agents/servers.md「怎么连」检查工位机与 Tailscale 后重跑"
                       + ("；若本票确实没在 A6000 跑，可带 --skip-a6000 重跑" if where == "a6000" else ""))
        return r.returncode, out

    def must(self, where: str, script: str, what: str, advice: str, **kw) -> str:
        rc, out = self.remote(where, script, **kw)
        if rc:
            raise Stop(f"{what}（{where}，退出码 {rc}）：\n{out}", advice)
        return out

    def gh(self, *args):
        r = subprocess.run([*self.s.gh, *args], cwd=self.s.local, capture_output=True, timeout=300)
        if r.returncode:
            raise Stop(f"gh {' '.join(args)} 失败：{(r.stdout + r.stderr).decode('utf-8', 'replace').strip()}",
                       "看上面的错误，处理后重跑")
        return r.stdout.decode("utf-8")

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
        prs = json.loads(self.gh("pr", "list", "--head", branch, "--state", "all", "--json",
                                 "number,state,headRefName,headRefOid,baseRefName,mergeCommit,title"))
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
            rc, remote_tip = self.git("rev-parse", "-q", "--verify", f"origin/{branch}", check=False)
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
        return Ticket(branch, pr, tip, files)

    def dependencies(self, target: Ticket) -> list[Ticket]:
        prs = json.loads(self.gh("pr", "list", "--state", "open", "--limit", "200", "--json",
                                 "number,state,headRefName,headRefOid,baseRefName,mergeCommit,title"))
        deps = []
        for pr in prs:
            oid = pr["headRefOid"]
            if pr["headRefName"] == target.branch or self.git("cat-file", "-e", oid, check=False)[0]:
                continue
            if self.git("merge-base", "--is-ancestor", oid, target.tip, check=False)[0] == 0 \
                    and self.git("merge-base", "--is-ancestor", oid, "origin/main", check=False)[0] != 0:
                deps.append(pr)
        # 先合并的在前：被越多其他依赖包含的越靠前
        anc = {p["number"]: sum(self.git("merge-base", "--is-ancestor", q["headRefOid"], p["headRefOid"], check=False)[0] == 0
                                for q in deps if q is not p) for p in deps}
        return [self.ticket(p["headRefName"], p) for p in sorted(deps, key=lambda p: anc[p["number"]])]

    @staticmethod
    def approval_reasons(t: Ticket) -> list[str]:
        out = []
        for st, p in t.files:
            if p.startswith(EVAL_PATHS):
                out.append(f"改了评测口径：{p}")
            elif p in RULE_PATHS or p.startswith(tuple(r for r in RULE_PATHS if r.endswith("/"))):
                out.append(f"改了规矩文件：{p}")
            elif CLOSED_RECORD.fullmatch(p) and st != "A":
                out.append(f"改了已结票实验的记录：{p}")
        return out

    # ---- 服务器 ----
    def wt_exists(self, where: str, branch: str) -> bool:
        root = self.s.gpfs_wt if where == "gpfs" else self.s.a6000_wt
        rc, _ = self.remote(where, f"test -d {shlex.quote(root + '/' + branch)}", timeout=60)
        return rc == 0

    def run_tests(self, target: Ticket):
        for where in ("gpfs", "a6000"):
            if where == "a6000" and self.skip_a6000:
                continue
            if not self.wt_exists(where, target.branch):
                continue
            d = (self.s.gpfs_wt if where == "gpfs" else self.s.a6000_wt) + "/" + target.branch
            head = self.must(where, f"git -C {shlex.quote(d)} rev-parse HEAD", "读 worktree 的 HEAD 失败", "检查 worktree").strip()
            if head != target.tip:
                raise Stop(f"{where} 上的 worktree {d} 停在 {head[:8]}，不是 PR 的 head {target.tip[:8]}",
                           "在 worktree 里推送或 pull 到与 PR 一致后重跑")
            self.log(f"在 {where} 的 {d} 跑测试：{self.s.test_cmd}")
            rc, out = self.remote(where, f"cd {shlex.quote(d)} && {self.s.test_cmd}", timeout=3700)
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

    def ignored_files(self, ids) -> dict[str, int]:
        """A6000 主 checkout 里本票实验不进 git 的文件 → 大小（.claims 是 A6000 自己的占位，不拷）。"""
        paths = " ".join(shlex.quote(f"runs/{i}") for i in ids)
        out = self.must("a6000", f"cd {shlex.quote(self.s.a6000_main)} && git ls-files -z --others --ignored --exclude-standard"
                                 f" -- {paths} | xargs -0 -r stat --printf '%s\\t%n\\n'",
                        "列 A6000 上的产物失败", "检查 A6000 主 checkout")
        files = {}
        for line in out.splitlines():
            size, _, p = line.partition("\t")
            if p and "/.claims/" not in p:
                files[p] = int(size)
        return files

    def gpfs_sizes(self, paths) -> dict[str, int]:
        out = self.must("gpfs", f"cd {shlex.quote(self.s.gpfs_main)} && while IFS= read -r f; do "
                                f"[ -f \"$f\" ] && printf '%s\\t%s\\n' \"$(stat -c %s -- \"$f\")\" \"$f\"; done; true",
                        "核对 gpfs 上的文件失败", "检查 gpfs 主 checkout", input="".join(p + "\n" for p in paths))
        return {p: int(size) for size, _, p in (line.partition("\t") for line in out.splitlines()) if p}

    def a6000(self, t: Ticket):
        if self.skip_a6000:
            self.log("--skip-a6000：不查 A6000")
            return
        had_wt = self.wt_exists("a6000", t.branch)
        if had_wt:
            self.pull_main("a6000")
            self.wt_close("a6000", t.branch)
        files = self.ignored_files(t.ids) if t.ids else {}
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
        rc, _ = self.git("rebase", "-q", "origin/main", check=False)
        if rc:
            self.git("rebase", "--abort", check=False)
            raise Stop("用户的编辑与 main 上的改动冲突，rebase 已撤回；提交留在本地 main，未推送",
                       "请用户看冲突（git rebase origin/main）后手动处理")
        rc, out = self.git("push", "-q", "origin", "main", check=False)
        if rc:
            raise Stop("推送失败（不强推），提交留在本地 main", "稍后重跑，或查看远端是否又有新提交")
        self.step("已推送，本地与 origin/main 一致")
        self.pull_main("gpfs")

    def sync(self, ids):
        if not ids:
            return
        sys.path.insert(0, str(REPO))
        from workbench import sync
        try:
            r = sync.sync(ids, self.s.local / "runs", remote_root=self.s.gpfs_main, shell=self.s.gpfs_shell)
        except RuntimeError as e:
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
