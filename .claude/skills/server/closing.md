# 收尾脚本

`/server` 技能的参考文件。收尾把一张票从「PR 已开」带到「工作台上能看」，由 `scripts/close_ticket.py`（#175）一条命令完成。

## 用法

在用户电脑的本地仓库（main）里运行，用本机跑工作台的那个 python：

    python scripts/close_ticket.py <分支> [--confirm] [--skip-a6000]

- `--confirm`：用户已同意合并「需点头」类别的改动。第一次不带；脚本以退出码 2 停下并列出原因，转问用户，同意后带它重跑。
- `--skip-a6000`：不查 A6000。只在 A6000 连不上、且本票确实没在 A6000 跑时用。

脚本只做 git、ssh、gh 与文件传输，不在本机计算。

## 做什么（按顺序）

1. 检查本地：在 main 上、没有未推送的提交、未提交改动只有 `runs/*/notes.md`、`runs/canvas.json`、未跟踪的实验目录。否则停下。
2. 找分支的 PR（开着的或已合并的）。分支包含其他开着的 PR 的提交时，这些 PR 按先后顺序一起处理。
3. 任一待合并 PR 改了需点头的类别就停下（退出码 2）。类别以脚本为准：
   - 评测口径与规矩文件：`EVAL_PATHS`、`RULE_PATHS` 两个常量列出的路径。
   - 已结票实验的 `notes.md` / `metrics.json`：分叉点上已有这份记录，且把这个实验带进 main 的 PR 里有不属于本票的（与本 PR 分支名不同、Closes 的 issue 也不在本 PR 的 Closes 里），或这个实验不是经 PR 进的 main。同一张票的第二个 PR 补改自己的记录不算。
4. 只读地查 A6000 上有没有这些分支的 worktree；连不上就在合并前停下。
5. 改了 `runs/` 以外的文件时，在目标分支的 gpfs worktree（没有则 A6000 worktree）里跑全量测试（脚本常量 `TEST_CMD`：工作台环境与训练环境两段，都过才算过）。worktree 的 HEAD 必须等于 PR head。失败则停下，不合并。
6. 每个 PR：`gh pr merge --merge --match-head-commit <测过的 commit>` → gpfs 主 checkout `pull --ff-only` →
   gpfs 上有 worktree 就 `wt.sh close --dry-run` 再 `wt.sh close` → A6000 上有该分支的 worktree 或本票实验的产物时：
   A6000 主 checkout pull、`wt.sh close`、把不进 git 的产物（`.claims` 除外）经 150 拷回 gpfs、逐文件核对大小、一致才删 A6000 上的这些文件。
   A6000 主 checkout 里本票实验有未跟踪、又不被 gitignore 的文件时，什么都不拷、停下列出它们。
7. 本地：有允许的改动就提交（信息形如「用户编辑：E13 Notes、画布、E11 新实验目录」）、rebase 到 origin/main、推送，gpfs 主 checkout 再 pull；
   没有就 `pull --ff-only`。
8. `workbench sync` 本票（含依赖 PR）动过的实验，最后一行输出 `close: 完成。可在工作台查看：<id …>`。

`wt.sh close <分支>` 先核对本分支动过的 `runs/<id>/` 与主 checkout 一致，再把不进 git 的文件（ckpt、tb、点对、sweep 原始数据）挪到主 checkout，然后删 worktree，分支保留。有冲突或剩余文件时它会停下，不删任何东西。

## 停下与退出码

- **0：走完。** 最后一行列出可在工作台查看的实验 id。
- **2：需要用户点头，什么都没动。** 把列出的原因转问用户；用户同意后带 `--confirm` 重跑。
- **1：停下。** 输出三行：`close: 停下：…`（原因和现场）、`close: 已完成：…`（已做的步骤）、`close: 建议：…`。
  把这三行连同你的判断报告给用户，现场原样留给用户，等用户决定；处理完后重跑。
  脚本自己不删文件、不强推。常见原因：本地有别的改动或未推送提交；没有 PR / PR 目标不是 main；worktree 不在 PR head；测试失败；
  `wt.sh close` 停下（同名不同大小、剩余产物、未提交代码，原样转述它的输出）；gpfs 已有与 A6000 同名不同大小的文件（不拷）；
  A6000 上有未提交又不被忽略的记录；拷回核对不一致（A6000 副本保留）；
  rebase 冲突或推送失败（替用户做的提交已撤回，改动留在本地工作区）；git / gh / ssh 超时或输出异常；连不上服务器（A6000 连不上时提示 `--skip-a6000`）。

重跑是安全的：已合并的 PR 跳过合并，已关的 worktree 跳过 close，已拷过且大小一致的文件跳过。
