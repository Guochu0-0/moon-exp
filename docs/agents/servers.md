# 服务器使用要点

截至 2026-09-26 的积累。主机清单在用户本机的 `~/.ssh/config`（`xufang<编号>外网`）。

## 机器

- 150 / 154 / 160 / 126 挂同一份 gpfs 存储。用户目录是 `/remote-home/xufang/YGC/`。
- **A6000 是独立文件系统，没有 YGC**。155 不使用。
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

**已建好的通道**：工位机（Windows，Clash Verge）上有计划任务 `ygc-tunnel`（`%USERPROFILE%\ygc-tunnel\ygc-tunnel.ps1`，断线自动重连）。它维持一条经内网连到 150（`root@10.254.1.150 -p 20128`）的反向 ssh 隧道。

以下两个端口**只在 150 上**可用：

| 端口 | 作用 |
|---|---|
| `127.0.0.1:11080` | SOCKS5，出口是工位机直连 |
| `127.0.0.1:17890` | 工位机的 Clash，能访问 Google Drive、huggingface.co |

用法：
- 在 150 上用 `curl -x socks5h://127.0.0.1:11080 ...` 下载，Google Drive 和 HF 用 `-x http://127.0.0.1:17890`。
- 文件存到 `YGC/weights/`，154、160、126 同样可见。
- 两个端口不在时，说明工位机关机或已登出，请用户检查。
- 停用隧道：在工位机上执行 `Unregister-ScheduledTask ygc-tunnel`。

服务器能直连的站点：GitHub（仅 154、126）、pypi、hf-mirror、dl.fbaipublicfiles.com。

## MATLAB（目前只在 154）

- R2024b，装在 `/opt/matlab/R2024b`，含 Image Processing、Computer Vision、Statistics、Signal、Parallel 工具箱。
- 学校 Total Headcount 的个人 license 只支持在线授权，不能用 license 文件激活。
- 授权方式：
  1. matlab-proxy 常驻在 154 上（`/root/start-mwi.sh`，端口 8888，token 在 `/root/.mwi_token`）。
  2. 用户经 `ssh -N -L 8888:localhost:8888 xufang154外网` 打开网页，选 **Online License Manager** 登录一次。**不要选 Existing License。**
- 之后无界面运行：`/root/matlab-batch.sh "<cmd>"`，它借用 proxy 会话的授权。
- proxy 会话断了，需要用户重新登录。
