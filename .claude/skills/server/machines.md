# 机器、连法、存储与环境

`/server` 技能的参考文件。更新于 2026-10-02。用卡规矩与操作注意见 [rules.md](rules.md)，A6000 见 [a6000.md](a6000.md)。

每台机器都是课题组统一管理的 Docker 容器，用户是 root。容器重启后挂载可能变化，凡是涉及路径的事，每台机器单独核实（`df`、`hostname`），不要凭以前的印象。

## 机器一览

| 主机 | GPU | 存储 | 现状 |
|---|---|---|---|
| 154 | TITAN RTX | gpfs | 唯一有下载隧道端口和 MATLAB 的机器。2026-09-30 内网端口拒绝连接，2026-10-02 ssh 握手被断开，待查 |
| 126 | 4 × V100-SXM2 32 GB | gpfs | 可用。GPU0 常被别人占（约 17 GB） |
| 160 | 4 × V100-SXM2 32 GB | gpfs | 可用，2026-10-12 起装了环境。GPU 0、2 当时有别人的任务。**容器时钟是 UTC**（比北京时间慢 8 小时）；RoMa 逐存档评测比 126 慢近一倍 |
| 150 | 8 × RTX 3090 24 GB | gpfs（2026-09-30 起重新挂上） | 可用，用卡有专门规矩（见 rules.md）。**容器时钟慢约 13 小时** |
| A6000 | 6 × RTX A6000 48 GB + 2 × RTX 5880 Ada 46 GB | 独立 NFS | 可用，别人用得多，见 a6000.md |
| 155 | — | — | 不使用 |

- 154 / 160 / 126 / 150 挂同一份 gpfs，用户目录是 `/remote-home/xufang/YGC/`。驱动 `scripts/finetune/run.py` 的占位在实验目录的 `runs/<id>/.claims/` 里，跨这四台共享（旧队列的 `results/finetune/_claims` 已归档）。
- A6000 不挂 gpfs，占位不与上面共享。

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
- 内网组即使跳转成功，单次连接也偶尔被断。见 rules.md「操作注意」里的重试规矩。

## 存储、环境、数据放在哪

- 本仓库克隆在 `YGC/moon-exp`，是**主 checkout**：永远停在 main，只做 `git pull --ff-only`；每张票的代码在 `YGC/wt/<分支>` 的 worktree 里改和跑（`scripts/wt.sh`）。git 网络操作在能连 GitHub 的机器上做（154、126），用仓库级 deploy key `YGC/.ssh/moon-exp_deploy`，已配成 `core.sshCommand`。
- baseline 代码以 git submodule 形式放在 `third_party/`，commit 钉死。
- 软件和 conda 环境装在**各容器本地**，不装在 gpfs 上（gpfs 传输慢）。gpfs 只放代码、数据、权重和结果。
  - `/opt/envs/loftr`（py3.10，torch 2.1.2+cu118，训练与推理）、`/opt/envs/wb`（py3.11，工作台与评测）。
  - 150 的环境从 `YGC/envpack/envs.tar`（2026-10-02 从 `tmp/` 挪出） 解出，与 154 一致；160 的环境 2026-10-12 同样解出（`tar xf … -C /opt`，约 20 秒）。
- 数据在 `YGC/dataset/Moon`（Train 7907 / Val 825 / Test 1130 对）。Val 的 ROI_060 已删除，Val 为 6 个 ROI。
- 权重在 `YGC/weights/`（`anymatch/`、`minima/`、`matchanything/` 等）。RoMa 系要的 DINOv2 缓存在 `YGC/weights/torch_home`（设 `TORCH_HOME` 指向它，gpfs 上各机共用）。
- 训练产物写进实验目录 `runs/<id>/`，结票后归到主 checkout 的 `runs/`。`YGC/results/finetune/` 停止写入，只剩离线伪标签 `labels_*.jsonl`（任务清单用绝对路径引用，照常读取）；10/02 以前的训练产物已迁进各实验的 `runs/<id>/`（#77）。
- `YGC/results/baselines`、`baselines_ablation` 是 B0、B0m 的原始匹配输出，`runs/B0/code/`、`runs/B0m/code/` 里的脚本读它们，留在原位。
- **归档**：`YGC/_archive/2026-10/` 放 10/02 整理时挪走的东西：旧 worktree 与 `git archive` 副本（`moon-exp-*`）、`tmp/`、`results/finetune` 的其余部分（R 系、S1 变体、日志、锁、一次性脚本原件）、`results/{coarse70,diag_reward}`、`projects/optical-sar-matching/`，以及各方法修剪下来的 ckpt（`ckpt/`：峰值与最后一个之外的，删除前问用户）。迁移日志在 `YGC/_archive/*77*.log`。

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

服务器能直连的站点：GitHub（154、126；A6000 只通 22 端口），pypi、hf-mirror、dl.fbaipublicfiles.com。

## MATLAB（目前只在 154）

- R2024b，装在 `/opt/matlab/R2024b`，含 Image Processing、Computer Vision、Statistics、Signal、Parallel 工具箱。
- 学校 Total Headcount 的个人 license 只支持在线授权，不能用 license 文件激活。
- 授权方式：
  1. matlab-proxy 常驻在 154 上（`/root/start-mwi.sh`，端口 8888，token 在 `/root/.mwi_token`）。
  2. 用户经 `ssh -N -L 8888:localhost:8888 xufang154外网` 打开网页，选 **Online License Manager** 登录一次。**不要选 Existing License。**
- 之后无界面运行：`/root/matlab-batch.sh "<cmd>"`，它借用 proxy 会话的授权。
- proxy 会话断了，需要用户重新登录。
