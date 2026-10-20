---
name: server
description: 在实验室共享 GPU 服务器上跑代码：开 worktree、提交、启动、收尾结票，以及机器、连法、用卡规矩、A6000、MATLAB。碰服务器（ssh、训练、评测、跑测试、只用 CPU 的分析）、要跑任何计算、或结票时使用。
---

# 在服务器上跑代码

讲代码放哪、怎么启动、结束后怎么收尾。结果写成什么样见 `/records` 技能。来由见「实验代码与结果的管理规范」（#75）。

**一切计算都在实验室服务器上跑**，包括只用 CPU、数据正好在本机的分析：用户电脑带不动（2026-10-19 E10 在本机开 12 个 CPU 进程，电脑卡到不能用）。本机只改代码、做 git、跑单进程、几对样本的小探针。

## 参考文件

服务器的事实放在本目录的参考文件里，到了对应的情形就读：

- [machines.md](machines.md)：机器一览、怎么连、存储/环境/数据放在哪、大文件怎么进服务器、MATLAB。第一次连某台机器、找环境或数据路径、连不上时读。
- [rules.md](rules.md)：用卡规矩与操作注意。**任何远程操作之前读**；把服务器探查派给 sub agent 时，把这两节原样写进它的提示。
- [a6000.md](a6000.md)：A6000 的 GPU 编号、存储、网络、代码同步与结果拷回。要在 A6000 上做任何事时读。
- [closing.md](closing.md)：收尾脚本的用法、步骤、停下的情形与退出码。收尾时读。

## 三条原则

1. **GitHub 是代码的唯一来源。** 用来跑的代码必须已经提交，可以是未合并分支上的提交。驱动和训练入口在启动时检查这一点，不干净就拒绝启动。
2. **一张票对应一个分支、一个 worktree、一个 PR。** 结票即合并进 main。被放弃的实验，记录也照样进 main。
3. **结果只有一个权威存储：gpfs 主 checkout 的 `runs/`。** 工作台的 `sync` 只从这里拉。

## 位置

| 位置 | 是什么 | 规矩 |
|---|---|---|
| `YGC/moon-exp` | 主 checkout | 永远停在 main，只做 `git pull --ff-only`。改代码、跑实验都在 worktree 里做；主 checkout 的 `third_party/` 可能正被旧实验引用，不在这里切分支或改文件。 |
| `YGC/wt/<分支>` | 本票的 worktree | 在这里改代码、提交、跑实验。结果先写在这里的 `runs/<id>/`。 |
| A6000 的 `YGC/moon-exp`、`YGC/wt/` | A6000 自己的一套 | 同上；跑完把结果拷回 gpfs，见 a6000.md。 |

`YGC/results/finetune/`（只剩离线伪标签 `labels_*`）和导出副本是旧做法，停止写入；旧产物已迁进 `runs/`，其余归档在 `YGC/_archive/2026-10/`（「历史代码与结果整理」，#77）。产物只写进 `runs/<id>/`。

## 流程

1. **开 worktree**（在 154 或 126 上，它们能连 GitHub）：
   `/remote-home/xufang/YGC/moon-exp/scripts/wt.sh new <分支>`：已有远端分支就跟踪它，否则从 `origin/main` 新建。子模块从主 checkout 借对象初始化。路径默认值可用环境变量 `MAIN`、`WT` 覆盖。
2. **建实验**：`python -m workbench new <id> --parent <父> --title "…"`（见 `/records`）。
3. **写配置（和需要的代码），提交并推送。**
   - 每个方法一份 `runs/<id>/configs/<方法>.toml`：选模型、列出用到的训练成分、只写与默认不同的参数；同一实验内可用 `base` 继承另一个方法的配置，文件名以 `_` 开头的只当 base。格式与全部参数见 `finetune/README.md`。
   - 现有训练成分不够用时改 `finetune/`：新行为只能通过新增参数开启，默认保持旧行为；修 bug 例外，但在 commit 和受影响实验的 notes 里写明。
   - 可复用的代码（被多个实验调用、或被 import 的）放在 `finetune/`、`baselines/`、`moonlib/`、`scripts/` 等。`scripts/` 只放可复用工具。
   - 只服务这一个实验的放 `runs/<id>/code/`：诊断和画图脚本、一次性命令。以后被第二个实验用到时，再提升到 `scripts/` 或包里。
   - 跨实验的分析，归「Notes 里写这份结论的实验」；没有合适的实验，就为这次分析单独建一个。
4. **启动**（先按 rules.md「用卡规矩」查空卡）：
   ```bash
   cd $WT/<分支>
   GPU=<空卡> nohup /opt/envs/loftr/bin/python scripts/finetune/run.py runs/<id> \
       > runs/<id>/ckpt/queue_<host>_g<GPU>.log 2>&1 < /dev/null &
   ```
   - 驱动直接接收实验目录，按文件名顺序领取 `configs/` 里的方法。每张卡起一个进程，靠 `runs/<id>/.claims/<方法>` 占位。中途新增的方法也会被领走，但要先提交配置，否则会被拒绝启动。开跑前驱动先把全部配置展开一遍，有错就不开跑。
   - 每个方法的启动记录写进 `runs/<id>/launch/<方法>.json`，进 git：commit、主机、GPU、命令、起止时间、结果，以及入口（`entry`）、配置文件（`config_file`）、**展开所有默认值后的完整配置**（`args`）和推理配置（`config`）。字段见 `/records` 的 formats.md「启动记录」。
   - 一次性的手工命令同样要先提交再跑。临时调试可以设 `MOON_ALLOW_DIRTY=1` 放行，launch 记录里会留下 dirty 标记；不能用它产出要进论文的结果。
   - 进度用只读命令去查（rules.md「操作注意」）。别的 worktree 里正在跑的东西保持原样；判断有没有在跑：`ps aux | grep '[r]un.py'`，以及各实验的 `.claims/`。
5. **写 Notes，提交小文件，开 PR。** 大文件由 `.gitignore` 排除，留在 worktree 里。PR 正文含 `Closes #<票>`，合并即关票。
6. **收尾**：在用户电脑的本地仓库（main）里运行 `python scripts/close_ticket.py <分支>`，读 closing.md 处理它的结果。
   - 退出码 0：最后一行列出可在工作台查看的实验 id，收尾完成。
   - 退出码 2：PR 改了需用户点头的类别，脚本什么都没动。把它列出的原因转问用户；用户同意后带 `--confirm` 重跑。
   - 退出码 1：脚本停下了。把它输出的三行（停下的原因、已完成的步骤、建议）连同你的判断报告给用户，现场原样留给用户（不删文件、不强推），等用户决定怎么处理；处理完后重跑，重跑是安全的。

代码只从已提交的 git worktree 跑：`git archive` 导出副本和服务器上就地改的未提交代码都不能用来跑实验。
