# 服务器使用要点

更新于 2026-10-02。本文是服务器事实与用法的唯一来源：机器、怎么连、用卡规矩、操作注意、存储与环境、大文件、A6000、MATLAB。
**代码怎么上服务器、结果放哪**不在本文，见 `docs/agents/experiments.md`（来由见[实验代码与结果的管理规范](https://github.com/Guochu0-0/moon-exp/issues/75)）。

每台机器都是课题组统一管理的 Docker 容器，用户是 root。容器重启后挂载可能变化，凡是涉及路径的事，每台机器单独核实（`df`、`hostname`），不要凭以前的印象。

## 机器一览

| 主机 | GPU | 存储 | 现状 |
|---|---|---|---|
| 154 | TITAN RTX | gpfs | 唯一有下载隧道端口和 MATLAB 的机器。2026-09-30 内网端口拒绝连接，2026-10-02 ssh 握手被断开，待查 |
| 126 | 4 × V100-SXM2 32 GB | gpfs | 可用。GPU0 常被别人占（约 17 GB） |
| 160 | — | gpfs | 很少用 |
| 150 | 8 × RTX 3090 24 GB | gpfs（2026-09-30 起重新挂上） | 可用，用卡有专门规矩（见下）。**容器时钟慢约 13 小时** |
| A6000 | 6 × RTX A6000 48 GB + 2 × RTX 5880 Ada 46 GB | 独立 NFS | 可用，别人用得多，见「A6000」一节 |
| 155 | — | — | 不使用 |

- 154 / 160 / 126 / 150 挂同一份 gpfs，用户目录是 `/remote-home/xufang/YGC/`。四台共用 `results/finetune/_claims`（旧队列的占位；新驱动 `scripts/finetune/run.py` 的占位在实验目录的 `runs/<id>/.claims/` 里，同样跨这四台）。
- A6000 不挂 gpfs，`_claims` 不与上面共享。

## 怎么连

主机清单在用户本机的 `~/.ssh/config`，分两组：

- `xufang<编号>外网`：经外网网关 146.56.220.99。**经常断**（`kex_exchange_identification: Connection closed`）。
- `xufang<编号>内网`：经用户工位机跳转（`ProxyJump lab工位机`），**首选**。

| 内网主机名 | 地址 |
|---|---|
| `xufang126内网` | 172.16.2.126:20102 |
| `xufang160内网` | 10.254.1.160:20198 |
| `xufang150内网` | 10.254.1.150:20128 |
| `xufangA6000内网` | 10.254.29.178:15038 |
| 154 | 10.254.1.154:20020（2026-09-30 拒绝连接，暂无条目） |

- 工位机 `lab工位机`：WD-018，Tailscale 地址 100.91.240.17，用户 guochu，key `D:/ssh_key/id_rsa`。内网组的前提是工位机开着、Tailscale 在线、不睡眠；连不上时先请用户查这三样。
- 内网组即使跳转成功，单次连接也偶尔被断。见「操作注意」里的重试规矩。

## 用卡规矩（和其他同学共用）

原则：尽量用满空闲算力，同时给其他同学留余地。用户 2026-10-02 定。

1. **只用空闲卡**。用之前先查；有别人进程的卡一律不碰，也不和别人同卡。启动后发现同卡，撤下自己的任务。
2. **自己占用有上限**。上限只限制我们自己同时占多少张，不是给机器保留的空卡数：空卡少于上限时，空几张就可以用几张。

   | 主机 | 总卡数 | 白天上限 | 夜间上限 | 依据 |
   |---|---|---|---|---|
   | 150 | 8 | 3 | 6 | 同学用得最多、也最常整机空着，所以白天紧、夜里放宽 |
   | 126 | 4 | 2 | 3 | GPU0 常年有人占，实际可用的多半是 1–3 号 |
   | A6000 | 8 | 2 | 4 | 别人用得最多，常常只有一两张空卡 |
   | 154 / 160 | — | 2 | 卡数的一半 | 目前基本不用，恢复后先按此 |

   - 夜间 = 北京时间 23:00–08:00，以本机时间为准，不看服务器时钟（150 慢约 13 小时）。
   - 用户当天另有指示时以指示为准。
3. **及时让卡**。单个任务（训练 + 评测）尽量控制在 4 小时以内，长的拆段或分段续训，让卡每隔几小时空出来一次。夜间多占的部分只放 08:00 前能跑完的任务，到点没完的停在最近的 ckpt，白天按白天上限接着跑。
4. **到上限就排队，不加卡**。多出来的任务留在队列里，等自己的卡空出来再跑。
5. 注意内存和 IO，别把机器拖卡。

计划把第 2、4 条做进 `queue.sh`：每领一个新任务前，按当前时段的上限和自己已占的卡数决定是否等待（随[管理规范落地：文档与工具](https://github.com/Guochu0-0/moon-exp/issues/76)实现）。实现之前，起队列前手动按上表核对。
- 停掉自己的任务之前，先确认它在哪个阶段（训练 / 评测）、产物是否已写出，再决定删不删目录。

## 操作注意

- 远程命令都加 `timeout`。本机断开 ssh 并不会停掉远端进程；反过来，本机 `timeout` 杀掉 ssh，远端进程可能已半途退出并留下半成品文件。
- **只对只读检查和拷文件做重试。** 启动任务、清理占位这类非幂等操作，写成 `mkdir` 锁保护的脚本（放在 `YGC/` 下，不放 `/root`），执行一次，再用单独的只读命令核对。2026-09-30 曾因重试循环把同一任务在一张卡上起了两份。
- 长任务用 `nohup … < /dev/null &` 在远端起，写日志；结果用只读命令去查，不要让本机 ssh 挂着等。
- `pkill -f` / `pgrep -f` 会匹配到自己这条 ssh 命令（命令行里含同样的字符串）。用 `[x]yz` 写法，或把命令放进脚本再执行；按 PID 杀之前先 `ps -o cmd= -p <pid>` 确认。
- 不要在同一条命令里既用管道往 ssh 送数据（如 `git archive … | ssh … tar -x`），又用 heredoc 给 `bash -s` 送脚本：两者抢 stdin，tar 会读到脚本、什么都不执行。分成两条命令。
- 本机是 Windows 的 Git Bash：参数里的 `/remote-home/...` 传给 Windows 程序（如本机 python）时会被改写成 `C:/Program Files/Git/remote-home/...`。生成要上服务器的文件时用 heredoc 直接写，或设 `MSYS_NO_PATHCONV=1`，写完 `grep "Program Files"` 核对。
- **时钟**：150 容器时钟慢约 13 小时，日志、`_claims` 里的时间都受影响。跨机器比较时间先在各自机器上 `date -u` 换算，不要直接拿各机器日志里的本地时间比先后。
- 共享存储上不要跑深层 `find` / `du`，尤其不要扫 `YGC/dataset`。
- 服务器探查交给 sub agent，并把「用卡规矩」「操作注意」两节原样写进它的提示。

## 存储、环境、数据放在哪

- 本仓库克隆在 `YGC/moon-exp`，是**主 checkout**：永远停在 main，只做 `git pull --ff-only`；每张票的代码在 `YGC/wt/<分支>` 的 worktree 里改和跑（`scripts/wt.sh`）。git 网络操作在能连 GitHub 的机器上做（154、126），用仓库级 deploy key `YGC/.ssh/moon-exp_deploy`，已配成 `core.sshCommand`。
- baseline 代码以 git submodule 形式放在 `third_party/`，commit 钉死。
- 软件和 conda 环境装在**各容器本地**，不装在 gpfs 上（gpfs 传输慢）。gpfs 只放代码、数据、权重和结果。
  - `/opt/envs/loftr`（py3.10，torch 2.1.2+cu118，训练与推理）、`/opt/envs/wb`（py3.11，工作台与评测）。
  - 150 的环境从 `YGC/tmp/envpack/envs.tar` 解出，与 154 一致。
- 数据在 `YGC/dataset/Moon`（Train 7907 / Val 825 / Test 1130 对）。Val 的 ROI_060 已删除，Val 为 6 个 ROI。
- 权重在 `YGC/weights/`（`anymatch/`、`minima/`、`matchanything/` 等）。RoMa 系要的 DINOv2 缓存在 `YGC/weights/torch_home`（设 `TORCH_HOME` 指向它，gpfs 上各机共用）。
- 训练产物写进实验目录 `runs/<id>/`，结票后归到主 checkout 的 `runs/`。`YGC/results/finetune/<name>/` 是 10/02 以前的旧产物，停止写入；其中的离线伪标签 `labels_*.jsonl` 照常读取。
- `YGC/moon-exp-*`（ft26、ft50、r2、roma、cf70 等）是 9/28 以后用 `git archive` 导出的实验副本，没有 `.git`。按管理规范禁止再用这种副本跑实验，现有副本冻结，待整理后归档。
- 旧的 `projects/optical-sar-matching/` 不再使用。

显存参考（bs 1，PyTorch 峰值分配；nvidia-smi 看到的再多约 1.5 GB）：LoFTR 类 RIPE 带负样本对在 24 GB 卡上会 OOM；RoMa 伪标签约 6.3 GB、解冻 VGG 约 7.0 GB，RoMa 类 RIPE 带负样本对约 11.3 GB（各 run 的 `log.jsonl` 第一行 `mem_gb`）。

## 大文件进服务器

实验室要求大文件走学院内网。

- 不从宿舍电脑往服务器传大文件。数据集绝对不行，权重一般也不行。
- 尽量少占服务器自己的外网带宽。

**已建好的通道**：工位机（Windows，Clash Verge）上有计划任务 `ygc-tunnel`（`%USERPROFILE%\ygc-tunnel\ygc-tunnel.ps1`，断线自动重连）。它维持一条经内网连到 154（`root@10.254.1.154 -p 20020`）的反向 ssh 隧道。154 连不上时这条通道也用不了。

以下两个端口**只在 154 上**可用：

| 端口 | 作用 |
|---|---|
| `127.0.0.1:11080` | SOCKS5，出口是工位机直连 |
| `127.0.0.1:17890` | 工位机的 Clash，能访问 Google Drive、huggingface.co |

用法：
- 在 154 上用 `curl -x socks5h://127.0.0.1:11080 ...` 下载，Google Drive 和 HF 用 `-x http://127.0.0.1:17890`。
  - Google Drive 大文件：`https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t`。
  - 文件夹里各文件的 ID 可以从 `https://drive.google.com/embeddedfolderview?id=<文件夹 ID>` 抓取。
- 文件存到 `YGC/weights/`，gpfs 上各机同样可见。
- 隧道目标写在工位机的 `ygc-tunnel.ps1` 里。要换机器，改其中的地址和端口，再执行 `Stop-ScheduledTask ygc-tunnel; Start-ScheduledTask ygc-tunnel`。
- 两个端口不在时，说明工位机关机或已登出，请用户检查。
- 停用隧道：在工位机上执行 `Unregister-ScheduledTask ygc-tunnel`。

服务器能直连的站点：GitHub（154、126、A6000）、pypi、hf-mirror、dl.fbaipublicfiles.com。

## A6000（2026-09-28 起）

- 8 卡：0 号、7 号是 RTX 5880 Ada（46 GB），1–6 号是 RTX A6000（48 GB）。别人用得多，常常只有一两张空卡。
- **GPU 编号**：CUDA 默认顺序和 nvidia-smi 不一致，`CUDA_VISIBLE_DEVICES=7` 会拿到别的卡。必须同时设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`（`scripts/finetune/run.py` 已设）。
- **`/dev/shm` 只有 64 MB**：DataLoader 多进程会报 `Bus error`，`--workers` 不超过 2。
- 这个容器别人也在用：`/root` 下的文件、`/root/anaconda3` 的环境、`/workspace/xufang/moon` 都不是我们的，不碰。辅助脚本也不放 `/root`。
- 存储：`/workspace/xufang` 是组里的 NFS（与 gpfs 是两套），我们的目录是 `/workspace/xufang/YGC`（dataset / weights / results / 代码）。容器里建了软链 `/remote-home/xufang/YGC → /workspace/xufang/YGC`，脚本里的路径不用改。
- 根分区（overlay）很满，只剩约 80 GB，只放环境。
- 环境：`/opt/envs/loftr` 与 `/opt/envs/wb`，包版本与 154 一致，从 pypi / download.pytorch.org 装。
- 网络：GitHub、pypi、hf-mirror、download.pytorch.org 能直连；能经内网到 154（`10.254.1.154:20020`）。
- **经内网从 154 拉文件**：A6000 上有 key `/root/.ssh/ygc_154`（公钥已加到 154 的 `authorized_keys`，注释 `ygc-a6000-to-154`）。
  用法：`ssh -i /root/.ssh/ygc_154 -p 20020 root@10.254.1.154 "tar cf - -C <目录> <内容>" | tar xf - -C <目标>`（154 没有 rsync）。
- 代码：`YGC/moon-exp` 从 154 的仓库克隆（remote = `ssh://root@10.254.1.154:20020/remote-home/xufang/YGC/moon-exp`，`GIT_SSH_COMMAND` 带上面的 key）；LoFTR 子模块直接从 GitHub 拉。A6000 能直连 GitHub，也可以直接 fetch。
- 速度：LoFTR 伪标签训练每步约 0.16 s（154 的 TITAN RTX 约 0.3 s）。
- A6000 上的 `YGC/moon-exp` 是它自己的主 checkout，worktree 同样建在 `YGC/wt/`。占位跨不过存储，给 A6000 单独的任务清单。跑完先在 A6000 上 `scripts/wt.sh close`，再把 `runs/<id>/` 整份（含 ckpt）拷回 gpfs 主 checkout，核对文件数和大小后才能删 A6000 上的副本。

## MATLAB（目前只在 154）

- R2024b，装在 `/opt/matlab/R2024b`，含 Image Processing、Computer Vision、Statistics、Signal、Parallel 工具箱。
- 学校 Total Headcount 的个人 license 只支持在线授权，不能用 license 文件激活。
- 授权方式：
  1. matlab-proxy 常驻在 154 上（`/root/start-mwi.sh`，端口 8888，token 在 `/root/.mwi_token`）。
  2. 用户经 `ssh -N -L 8888:localhost:8888 xufang154外网` 打开网页，选 **Online License Manager** 登录一次。**不要选 Existing License。**
- 之后无界面运行：`/root/matlab-batch.sh "<cmd>"`，它借用 proxy 会话的授权。
- proxy 会话断了，需要用户重新登录。
