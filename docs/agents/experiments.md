# 跑实验：代码、分支与服务器

讲在服务器上跑实验时代码放哪、怎么启动、结束后怎么收尾。结果写成什么样见 `workbench/RECORDS.md`，服务器本身的规矩见 `docs/agents/servers.md`。来由见「实验代码与结果的管理规范」（#75）。

## 三条原则

1. **GitHub 是代码的唯一来源。** 用来跑的代码必须已经提交，可以是未合并分支上的提交。驱动和训练入口在启动时检查这一点，不干净就拒绝启动。
2. **一张票对应一个分支、一个 worktree、一个 PR。** 结票即合并进 main。被放弃的实验，记录也照样进 main。
3. **结果只有一个权威存储：gpfs 主 checkout 的 `runs/`。** 工作台的 `sync` 只从这里拉。

## 位置

| 位置 | 是什么 | 规矩 |
|---|---|---|
| `YGC/moon-exp` | 主 checkout | 永远停在 main，只做 `git pull --ff-only`。不在这里改代码、不在这里跑实验。 |
| `YGC/wt/<分支>` | 本票的 worktree | 在这里改代码、提交、跑实验。结果先写在这里的 `runs/<id>/`。 |
| A6000 的 `YGC/moon-exp`、`YGC/wt/` | A6000 自己的一套 | 同上；跑完把结果拷回 gpfs，见 `servers.md`「A6000」。 |

`YGC/results/finetune/`（只剩离线伪标签 `labels_*`）和导出副本是旧做法，停止写入；旧产物已迁进 `runs/`，其余归档在 `YGC/_archive/2026-10/`（「历史代码与结果整理」，#77）。

## 流程

1. **开 worktree**（在 154 或 126 上，它们能连 GitHub）：
   `/remote-home/xufang/YGC/moon-exp/scripts/wt.sh new <分支>`：已有远端分支就跟踪它，否则从 `origin/main` 新建。子模块从主 checkout 借对象初始化。路径默认值可用环境变量 `MAIN`、`WT` 覆盖。
2. **建实验**：`python -m workbench new <id> --parent <父> --title "…"`（见 RECORDS.md）。
3. **写配置（和需要的代码），提交并推送。**
   - 每个方法一份 `runs/<id>/configs/<方法>.toml`：选模型、列出用到的训练成分、只写与默认不同的参数；同一实验内可用 `base` 继承另一个方法的配置，文件名以 `_` 开头的只当 base。格式与全部参数见 `finetune/README.md`。
   - 现有训练成分不够用时改 `finetune/`：新行为只能通过新增参数开启，默认保持旧行为；修 bug 例外，但在 commit 和受影响实验的 notes 里写明。
   - 可复用的代码（被多个实验调用、或被 import 的）放在 `finetune/`、`baselines/`、`moonlib/`、`scripts/` 等。`scripts/` 只放可复用工具。
   - 只服务这一个实验的放 `runs/<id>/code/`：诊断和画图脚本、一次性命令。以后被第二个实验用到时，再提升到 `scripts/` 或包里。
   - 跨实验的分析，归「Notes 里写这份结论的实验」；没有合适的实验，就为这次分析单独建一个。
4. **启动**（先查空卡，见 `servers.md`）：
   ```bash
   cd $WT/<分支>
   GPU=<空卡> nohup /opt/envs/loftr/bin/python scripts/finetune/run.py runs/<id> \
       > runs/<id>/ckpt/queue_<host>_g<GPU>.log 2>&1 < /dev/null &
   ```
   - 驱动直接接收实验目录，按文件名顺序领取 `configs/` 里的方法。每张卡起一个进程，靠 `runs/<id>/.claims/<方法>` 占位。中途新增的方法也会被领走，但要先提交配置，否则会被拒绝启动。开跑前驱动先把全部配置展开一遍，有错就不开跑。
   - 每个方法的启动记录写进 `runs/<id>/launch/<方法>.json`，进 git：commit、主机、GPU、命令、起止时间、结果，以及入口（`entry`）、配置文件（`config_file`）和**展开所有默认值后的完整配置**（`config`）。
   - 一次性的手工命令同样要先提交再跑。临时调试可以设 `MOON_ALLOW_DIRTY=1` 放行，launch 记录里会留下 dirty 标记；不能用它产出要进论文的结果。
5. **写 Notes，提交小文件，开 PR。** 大文件由 `.gitignore` 排除，留在 worktree 里。
6. **结票**：合并 PR → 在主 checkout 上 `git pull --ff-only` → `scripts/wt.sh close <分支> --dry-run` 看一眼 → `scripts/wt.sh close <分支>`。
   close 先核对本分支动过的 `runs/<id>/` 与主 checkout 一致，再把不进 git 的文件（ckpt、tb、点对、sweep 原始数据）挪到主 checkout，然后删 worktree，分支保留。有冲突或剩余文件时它会停下，不删任何东西。

## 不要做的事

- 不要用 `git archive` 导出副本跑实验，也不要在服务器上就地改未提交的代码再跑。
- 不要在主 checkout 上切分支或改文件，它的 `third_party/` 可能正被旧实验引用。
- 不要把产物写到 `runs/<id>/` 以外（`YGC/results/`、`YGC/tmp/`）。
- 不要动别的 worktree 里正在跑的东西。判断有没有在跑：`ps aux | grep '[r]un.py'`，以及各实验的 `.claims/`。
