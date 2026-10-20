# 收尾脚本

`/server` 技能的参考文件。收尾把一张票从「PR 已开」带到「工作台上能看」，由 `scripts/close_ticket.py`（#175）一条命令完成。

## 用法

在用户电脑的本地仓库（main）里运行，用本机跑工作台的那个 python：

    python scripts/close_ticket.py <分支> [--confirm] [--skip-a6000]

- `--confirm`：用户已同意合并「需点头」类别的改动。第一次不带；脚本以退出码 2 停下并列出原因，转问用户，同意后带它重跑。
- `--skip-a6000`：不查 A6000。只在 A6000 连不上、且本票确实没在 A6000 跑时用。

脚本只做 git、ssh、gh 与文件传输，不在本机计算。

## 做什么（按顺序）

1. 检查本地：在 main 上、没有未推送的提交、未提交改动只有 `runs/*/notes.md`、`runs/canvas.json`、未跟踪的实验目录。否则停下（1）。
2. 找分支的 PR（开着的或已合并的）。分支包含其他开着的 PR 的提交时，这些 PR 按先后顺序一起处理。
3. 需点头的类别（任一待合并 PR 改了它们就停下，退出码 2）：
   - 评测口径：`moonlib/`、`workbench/evaluate.py`、`workbench/protocol.py`
   - 已结票实验的记录：已存在于 main 的 `runs/<id>/notes.md`、`runs/<id>/metrics.json` 被改或删
   - 规矩文件：`AGENTS.md`、`CLAUDE.md`、`CONTEXT.md`、`CODING_STANDARDS.md`、`docs/agents/`、`docs/adr/`、`.claude/skills/`
4. 改了 `runs/` 以外的文件时，在目标分支的 gpfs worktree（没有则 A6000 worktree）里跑全量测试：
   `CUDA_VISIBLE_DEVICES= timeout 3600 /opt/envs/wb/bin/python -m pytest -q tests`。worktree 的 HEAD 必须等于 PR head。失败则停下（1），不合并。
5. 每个 PR：`gh pr merge --merge --match-head-commit <测过的 commit>` → gpfs 主 checkout `pull --ff-only` →
   gpfs 上有 worktree 就 `wt.sh close --dry-run` 再 `wt.sh close` → A6000 上有该分支的 worktree 或本票实验的产物时：
   A6000 主 checkout pull、`wt.sh close`、把不进 git 的产物（`.claims` 除外）经 150 拷回 gpfs、逐文件核对大小、一致才删 A6000 上的这些文件。
6. 本地：有允许的改动就提交（信息形如「用户编辑：E13 Notes、画布、E11 新实验目录」）、rebase 到 origin/main、推送，gpfs 主 checkout 再 pull；
   没有就 `pull --ff-only`。
7. `workbench sync` 本票（含依赖 PR）动过的实验，最后一行输出 `close: 完成。可在工作台查看：<id …>`。

`wt.sh close <分支>` 先核对本分支动过的 `runs/<id>/` 与主 checkout 一致，再把不进 git 的文件（ckpt、tb、点对、sweep 原始数据）挪到主 checkout，然后删 worktree，分支保留。有冲突或剩余文件时它会停下，不删任何东西。

## 停下与退出码

- 0：走完。
- 1：停下。输出三行：`close: 停下：…`（原因和现场）、`close: 已完成：…`（已做的步骤）、`close: 建议：…`。
  不删文件、不强推。常见原因：本地有别的改动或未推送提交；没有 PR / PR 目标不是 main；测试失败；
  `wt.sh close` 停下（同名不同大小、剩余产物、未提交代码，原样转述它的输出）；gpfs 已有与 A6000 同名不同大小的文件（不拷）；
  拷回核对不一致（A6000 副本保留）；rebase 冲突（已撤回，提交留在本地）；推送失败；连不上服务器（A6000 连不上时提示 `--skip-a6000`）。
- 2：需要用户点头，什么都没动。

重跑是安全的：已合并的 PR 跳过合并，已关的 worktree 跳过 close，已拷过且大小一致的文件跳过。
