# 服务器使用要点

截至 2026-09-26 的积累。主机清单在用户本机的 `~/.ssh/config`（`xufang<编号>外网`）。

## 机器

- 154 / 160 / 126 挂同一份 gpfs 存储。用户目录是 `/remote-home/xufang/YGC/`。
- **150 现在没有 YGC**：容器在 2026-09-26 00:18 重启，之后 `/remote-home/xufang` 变成本地 ext4 盘，没挂上 gpfs。之前是挂着的。用之前先 `df` 核实。
  2026-09-28 复查：ssh 能连，但仍未挂 gpfs。`/remote-home` 是本地 sdb1（15 T，已用 92%），没有 YGC 和 `/opt/envs`，8 张 3090 全部被占。目前不用。
- **A6000 是独立文件系统**，不挂 gpfs。2026-09-28 起纳入使用，见下面「A6000」一节。155 不使用。
- 容器重启后挂载可能变化。凡是涉及路径的事，每台机器单独核实（`df`、`hostname`），不要凭以前的印象。
- 每台都是课题组统一管理的 Docker 容器，用户是 root。

## 规矩（和其他同学共用）

- 不碰 `YGC/` 以外别人的文件和数据。
- 在容器里装软件、装环境都可以，只影响自己的容器。
- 每次用之前先查 GPU。非空闲的卡是别人在跑实验，一律不碰。
- 注意内存和 IO，别把机器拖卡。
- 共享存储上不要跑深层 `find` / `du`，尤其不要扫 `YGC/dataset`。
- 远程命令都加 `timeout`。本机断开 ssh 并不会停掉远端进程。
- `pkill -f` / `pgrep -f` 容易匹配到自己这条 ssh 命令。用 `[x]yz` 写法，或者把命令放进脚本再执行。
- 服务器探查交给 sub agent，并把本节规矩原样写进它的提示。
- Claude Code 会话处在 git worktree 里时，隔离检查会拦截大多数远程 `ssh … bash/nice` 命令。服务器操作要在主工作区里做。

## 代码、环境、数据放在哪

- 本仓库克隆在 `YGC/moon-exp`。
  - git 网络操作只在 154 上做（154、126 能连 GitHub）。
  - 用仓库级 deploy key `YGC/.ssh/moon-exp_deploy`，已配成 `core.sshCommand`。
- baseline 代码以 git submodule 形式放进 `third_party/`，commit 钉死。
- 旧的 `projects/optical-sar-matching/` 不再使用。
- 软件和 conda 环境装在**各容器本地**（如 `/opt`），不装在 gpfs 上，因为 gpfs 传输慢。gpfs 只放代码、数据、权重和结果。
- 数据在 `YGC/dataset/Moon`。Val 的 ROI_060 已删除，现在 Val 为 6 个 ROI、825 对。

## 大文件进服务器

实验室要求大文件走学院内网。

- 不从宿舍电脑往服务器传大文件。数据集绝对不行，权重一般也不行。
- 尽量少占服务器自己的外网带宽。

**已建好的通道**：工位机（Windows，Clash Verge）上有计划任务 `ygc-tunnel`（`%USERPROFILE%\ygc-tunnel\ygc-tunnel.ps1`，断线自动重连）。它维持一条经内网连到 154（`root@10.254.1.154 -p 20020`）的反向 ssh 隧道。2026-09-26 从 150 改到 154，因为 150 没了 YGC，下载的文件落不到共享存储上。

以下两个端口**只在 154 上**可用：

| 端口 | 作用 |
|---|---|
| `127.0.0.1:11080` | SOCKS5，出口是工位机直连 |
| `127.0.0.1:17890` | 工位机的 Clash，能访问 Google Drive、huggingface.co |

用法：
- 在 154 上用 `curl -x socks5h://127.0.0.1:11080 ...` 下载，Google Drive 和 HF 用 `-x http://127.0.0.1:17890`。
  - Google Drive 大文件：`https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t`。
  - 文件夹里各文件的 ID 可以从 `https://drive.google.com/embeddedfolderview?id=<文件夹 ID>` 抓取。
- 文件存到 `YGC/weights/`，160、126 同样可见。
- 隧道目标写在工位机的 `ygc-tunnel.ps1` 里。要换机器，改其中的地址和端口，再执行 `Stop-ScheduledTask ygc-tunnel; Start-ScheduledTask ygc-tunnel`。154 的内网地址是 `10.254.1.154:20020`。
- 两个端口不在时，说明工位机关机或已登出，请用户检查。
- 停用隧道：在工位机上执行 `Unregister-ScheduledTask ygc-tunnel`。

服务器能直连的站点：GitHub（仅 154、126）、pypi、hf-mirror、dl.fbaipublicfiles.com。

## A6000（2026-09-28 起）

- 8 卡：0 号、7 号是 RTX 5880 Ada（46 GB），1–6 号是 RTX A6000（48 GB）。别人用得多，常常只有一两张空卡。
- **GPU 编号**：CUDA 默认顺序和 nvidia-smi 不一致，`CUDA_VISIBLE_DEVICES=7` 会拿到别的卡。必须同时设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`（`scripts/finetune/scenes.sh` 已设）。
- 这个容器别人也在用：`/root` 下的文件、`/root/anaconda3` 的环境、`/workspace/xufang/moon` 都不是我们的，不碰。
- 存储：`/workspace/xufang` 是组里的 NFS（与 gpfs 是两套），我们的目录是 `/workspace/xufang/YGC`（dataset / weights / results / 代码）。容器里建了软链 `/remote-home/xufang/YGC → /workspace/xufang/YGC`，脚本里的路径不用改。
- 根分区（overlay）很满，只剩约 80 GB，只放环境。
- 环境：`/opt/envs/loftr`（py3.10，torch 2.1.2+cu118）与 `/opt/envs/wb`（py3.11），包版本与 154 一致，从 pypi / download.pytorch.org 装。
- 网络：GitHub、pypi、hf-mirror、download.pytorch.org 能直连；能经内网到 154（`10.254.1.154:20020`）。
- **经内网从 154 拉文件**：A6000 上有 key `/root/.ssh/ygc_154`（公钥已加到 154 的 `authorized_keys`，注释 `ygc-a6000-to-154`）。
  用法：`ssh -i /root/.ssh/ygc_154 -p 20020 root@10.254.1.154 "tar cf - -C <目录> <内容>" | tar xf - -C <目标>`（154 没有 rsync）。
- 已就位（2026-09-28）：数据 `YGC/dataset/Moon`（38 GB，Train 7907 / Val 825 / Test 1130 对）、`YGC/weights/anymatch/LoFTR_AnyMatch.ckpt`、`YGC/results/finetune/labels_b0.jsonl`。
- 代码：`YGC/moon-exp` 从 154 的仓库克隆（remote = `ssh://root@10.254.1.154:20020/remote-home/xufang/YGC/moon-exp`，`GIT_SSH_COMMAND` 带上面的 key），新提交先在 154 上从 GitHub fetch 再拉过来；LoFTR 子模块直接从 GitHub 拉。
- 速度：S1 式训练每步约 0.16 s（154 的 TITAN RTX 约 0.3 s）。
- 与 154/126 不共享存储，`queue.sh` 的 `_claims` 占位跨不过去：给 A6000 单独的任务清单，结果再拷回 gpfs。

## MATLAB（目前只在 154）

- R2024b，装在 `/opt/matlab/R2024b`，含 Image Processing、Computer Vision、Statistics、Signal、Parallel 工具箱。
- 学校 Total Headcount 的个人 license 只支持在线授权，不能用 license 文件激活。
- 授权方式：
  1. matlab-proxy 常驻在 154 上（`/root/start-mwi.sh`，端口 8888，token 在 `/root/.mwi_token`）。
  2. 用户经 `ssh -N -L 8888:localhost:8888 xufang154外网` 打开网页，选 **Online License Manager** 登录一次。**不要选 Existing License。**
- 之后无界面运行：`/root/matlab-batch.sh "<cmd>"`，它借用 proxy 会话的授权。
- proxy 会话断了，需要用户重新登录。
