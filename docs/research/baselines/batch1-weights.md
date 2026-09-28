# Batch 1 权重清单：data-engine 多模态微调权重 + 通用/跨模态官方权重

> 调研日期：2026-09-25。本文只管**权重**：来源、文件、大小、版本、官方推理代码入口、服务器能否直接拿到。接口细节由另一份文档负责。
> 输入情形字母 A–K 沿用 `docs/research/baselines/README.md` §2。
> 服务器能访问：GitHub（含 release）、PyPI、hf-mirror.com。**不能访问**：huggingface.co 直连、Google Drive（GD）。GD 上的文件要先在 PC 下载，再 scp 到服务器。
> 标注：「读到」= 直接读了 README、加载代码、API 返回，或用 HTTP Range 读了压缩包目录 / ckpt 内的 pickle；「推断」= 我的判断。
> 文件大小都是字节数，来自 GitHub API `size`、HF API `siblings[].size`、GD 的 `Content-Range` 或 zip 中央目录（均为读到）。

---

## 1. Batch-1 全部 run（共 14 个）

可达性说明：**GH** = 服务器直接下载（GitHub）；**PC→scp** = 只在 Google Drive，必须先下到 PC；**hf-mirror** = 服务器走 hf-mirror.com。
**所有 RoMa 系 run 还需要额外下载 DINOv2 ViT-L/14**：`https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth`，1,217,586,395 B。这个文件**不在**任何一个 RoMa 微调 ckpt 里（见 §3），服务器能否访问 fbaipublicfiles **未核实**。

| # | run 名 | 基础匹配器 | 权重来源（文件） | URL | 官方推理代码路径 | 输入情形 | 服务器可达性 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 1 | `ma_eloftr` | ELoFTR（RoPE 版） | GD `weights.zip` → `weights/matchanything_eloftr.ckpt`（64,366,723 B） | https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d | HF Space `LittleFrog/MatchAnything@6a7bcb5`：`imcui/hloc/matchers/matchanything.py`（`PL_LoFTR(config, pretrained_ckpt=…)`，第 52-54 行）；评测脚本 `imcui/third_party/MatchAnything/scripts/evaluate/eval_visible_sar.sh` | A + D（长边 832，NPE=[832,832,r,r]） | **PC→scp**（zip 共 482,746,196 B，里面同时有 RoMa 版） | ckpt 结构 `{'state_dict': {'matcher.*'}}`，共 718 个 key，含 `rope_pos_enc`。备选：HF `zju-community/matchanything_eloftr@7bd52a4` 的 `model.safetensors`（64,263,488 B），走 transformers，属于**情形 K**，processor 固定缩放到 832×832、灰度、÷255（`preprocessor_config.json`，读到）；可以经 hf-mirror 下载，但它是 transformers 移植版（stevenbucaille 贡献），不是官方代码路径 |
| 2 | `ma_roma` | RoMa（DINOv2-L） | 同上 zip → `weights/matchanything_roma.ckpt`（445,649,931 B） | 同上 | 同上，`model_name='matchanything_roma'`；模型构造见 `third_party/ROMA/roma/matchanything_roma_model.py` | B′ + D（832，拉伸） | **PC→scp**，另需 DINOv2 | ckpt 结构 `{'state_dict': {'matcher.model.*'}}`，约 898 个 key，**不含 DINOv2**。`encoders.py:88-89` 在 `dinov2_weights is None` 时从 fbaipublicfiles 下载。第三方镜像 `vismatch/matchanything-roma@5f0b730`（445,502,400 B）可以走 hf-mirror，但它是 vismatch 格式，官方 `norm_img=False` 约定是否保留**未核实** |
| 3 | `minima_splg` | SuperPoint + LightGlue | GH release `LSXI7/storage` tag `MINIMA` → `minima_lightglue.pth`（47,484,144 B，2024-12-17） | https://github.com/LSXI7/storage/releases/download/MINIMA/minima_lightglue.pth | `LSXI7/MINIMA@796e772`：`load_model.py:64-89`（`load_sp_lg`，`self.matcher.load_state_dict(state_dict, strict=False)`） | A + D（640） | **GH** | 只有 LightGlue 部分被微调，ckpt 是裸 state_dict（`self_attn.*`、`cross_attn.*` 等 209 个 key）。SuperPoint 仍用原版 `superpoint_v1.pth`，由 `third_party/LightGlue/lightglue/superpoint.py:144` 从 GitHub `cvg/LightGlue` v0.1_arxiv 下载（5,206,086 B），服务器能直接拿到 |
| 4 | `minima_loftr` | LoFTR | 同 release → `minima_loftr.ckpt`（46,350,788 B，2024-12-17） | https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt | `load_model.py:37-56`（文件名不是 `outdoor_ds.ckpt` 时置 `temp_bug_fix=True`，第 49-51 行；`strict=True`） | A + D（640） | **GH** | 结构 `{'state_dict': {'matcher.*'}}`，369 个 key；`thr` 默认 0.2（`load_model.py:182`） |
| 5 | `minima_roma` | RoMa | 同 release → `minima_roma.pth`（445,638,463 B，2024-12-17） | https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth | `load_model.py:7-26`（`roma_outdoor(device, weights=state_dict)`，代码在 `third_party/RoMa_minima`） | B + D（640 → 560/864） | **GH**，另需 DINOv2 | 裸 state_dict（`encoder.cnn.*`、`decoder.*`），**不含 DINOv2**；`RoMa_minima/romatch/models/model_zoo/__init__.py:13` 从 fbaipublicfiles 下载 |
| 6 | `minima_eloftr` | ELoFTR（RoPE 版） | 同 release → `minima_eloftr.ckpt`（64,369,865 B，**2025-02-18** 才上传） | https://github.com/LSXI7/storage/releases/download/MINIMA/minima_eloftr.ckpt | **MINIMA 仓库里没有加载器**：`load_model.py:169` 的 choices 只有 xoftr/sp_lg/loftr/roma，`third_party/` 里也没有 EfficientLoFTR。推断应使用官方 `zju3dv/EfficientLoFTR@07e9c14` 的 `LoFTR(full_default_cfg)`（README 第 45-52 行），但需要去掉 key 前缀 `matcher.` | A + E(32)（按 ELoFTR 官方约定；MINIMA 论文的测试尺寸是长边 640） | **GH** | ckpt 结构与 `matchanything_eloftr.ckpt` 完全同构（都是 718 个 key，前缀相同，都有 `loftr_coarse.layers.*.rope_pos_enc`），推断两者都基于 RoPE 版 ELoFTR。**没有官方推理代码**，属于我们自己接的「半官方」路径 |
| 7 | `minima_xoftr` | XoFTR | 同 release → `minima_xoftr.ckpt`（44,484,144 B，**2025-02-18**） | https://github.com/LSXI7/storage/releases/download/MINIMA/minima_xoftr.ckpt | `load_model.py:143-153`（`load_xoftr`，`DataIOWrapper(matcher, config, ckpt=ckpt)`，代码在 `third_party/XoFTR`） | A + D（640） | **GH** | **陷阱**：`load_model.py:177` 中 xoftr 的 `--ckpt` 默认值是 `./weights/weights_xoftr_640.ckpt`，即**原版 XoFTR**，不是 MINIMA 权重。必须显式传 `--ckpt weights/minima_xoftr.ckpt`。另外还有 README.md §4 已记录的 `load_model()` 参数签名 bug（`load_model.py:143,160`） |
| 8 | `anymatch_loftr` | LoFTR（从 outdoor_ds 初始化，推断） | GD `Weights_AnyMatch.zip` → `LoFTR_AnyMatch.ckpt`（138,979,993 B） | https://drive.google.com/file/d/1JCz52pM7On_YWCjBWr_VuTiXt1uOXPD_ | `MnYangs/AnyMatch@259ad34`：`demo/demo_anymatch.py:155-157`（`--method loftr --ckpt ../weights/LoFTR_AnyMatch.ckpt --thr 0.1`），代码在 `third_party/LoFTR_AnyMatch`。**但 `load_model.py` 缺失**（见 §4） | A + D（640，`src/config/default.py:189`；`data_io_loftr.py` 灰度 ÷255） | **PC→scp**（zip 共 1,889,874,277 B） | 这是完整的 Lightning 训练 ckpt：含 `state_dict`（key 为 `matcher.*`）、`optimizer_states`、`lr_schedulers`、`callbacks`，所以约是裸权重的 3 倍。只取 `state_dict` 并去掉 `matcher.` 前缀 |
| 9 | `anymatch_edm` | EDM（ICCV 2025） | 同 zip → `EDM_AnyMatch.ckpt`（41,417,503 B） | 同上 | `demo_anymatch.py:158-160`（`--method edm`），代码在 `third_party/EDM_AnyMatch`（EDM 官方代码的分叉） | A + D（640；`data_io_edm.py:43-77` 灰度 ÷255，边长取 df 的倍数） | **PC→scp** | 同一个 zip 里还有 `EDM_original.ckpt`（大小与 AnyMatch 版相同），可以用作原版 EDM 对照（不在本批次内） |
| 10 | `anymatch_roma` | RoMa | 同 zip → `RoMa_AnyMatch.pth`（1,336,376,603 B） | 同上 | `demo_anymatch.py:161-165`（`--method roma --ckpt2 large --ckpt ../weights/RoMa_AnyMatch.pth`），代码在 `third_party/RoMa_AnyMatch` | B + D | **PC→scp**，另需 DINOv2 | 这是 RoMa 训练 ckpt：`{'model': {encoder.*, decoder.*}, 'optimizer', 'lr_scheduler', …}`，内部路径名为 `train_roma_megadepth_l2m_200000_latest/`，**不含 DINOv2**。要取 `['model']` 再传给 `roma_outdoor(weights=…)`（推断） |
| 11 | `xoftr` | XoFTR | GD 文件夹 → `weights_xoftr_640.ckpt`（44,486,099 B） | https://drive.google.com/file/d/1oRkEGsLpPIxlulc6a7c2q5H1XTNbfkVj（文件夹 https://drive.google.com/drive/folders/1RAI243OHuyZ4Weo1NiTy280bCE_82s4q） | `OnderT/XoFTR`（上游 `e0fbea4`；本地克隆是 `e9635d8`，比上游旧，缺 `np.float` 修复）：`test_relative_pose.py xoftr --ckpt weights/weights_xoftr_640.ckpt`（README 第 84 行） | A + D（640） | **PC→scp**；或走 hf-mirror 的第三方镜像 `vismatch/xoftr@d8ee7d8`（`xoftr_640.safetensors`，44,419,304 B） | **选 640，不选 840**：README 和测试配置默认都是 640（`src/config/default.py:189-190`、README 第 84 行），MINIMA 对照用的也是 640（`load_model.py:177`），而且 640 更接近我们的 512 输入。840 版文件大小相同（44,486,099 B），可以留作消融 |
| 12 | `loftr` | LoFTR | GD 文件夹 → `outdoor_ds.ckpt`（46,341,978 B） | https://drive.google.com/file/d/1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY（文件夹 https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp） | `zju3dv/LoFTR@df7ca80`：`LoFTR(config=default_cfg)`，`default_cfg` 来自 `src/loftr/utils/cvpr_ds_config.py`（README 第 84-89 行），等价于 `configs/loftr/outdoor/buggy_pos_enc/loftr_ds.py` | A + E(8)（demo 默认 640×480，我们保持 512） | **PC→scp** | **推荐 outdoor_ds，理由有三**：(1) outdoor 版在 MegaDepth（室外、大视角）上训练，比 ScanNet 室内更接近遥感；(2) ds（dual-softmax）不需要 SuperGlue 的最优传输代码（README 第 68 行：OT 版需要另装受许可限制的 SuperGlue 代码）；(3) 它就是 MINIMA-LoFTR 和 AnyMatch-LoFTR 的母体，可以直接比较。**必须 `TEMP_BUG_FIX=False`**（`cvpr_ds_config.py:28`；`src/config/default.py:23` 默认值是 True，不能直接用）。同目录还有 `indoor_ds_new.ckpt`、`indoor_ot.ckpt`、`outdoor_ot.ckpt`，不选 |
| 13 | `sp_sg` | SuperPoint + SuperGlue | 仓库内置 `models/weights/superglue_outdoor.pth`（48,233,807 B）+ `superpoint_v1.pth`（5,206,086 B） | https://github.com/magicleap/SuperGluePretrainedNetwork/tree/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/weights | `magicleap/SuperGluePretrainedNetwork@ddcf11f`：`match_pairs.py` → `models/matching.py`；权重路径在 `models/superglue.py:224-225`、`models/superpoint.py:136` | A（`models/utils.py:263-264` 灰度读入，第 260 行 ÷255）+ D（默认 `--resize 640 480`，`match_pairs.py:85`）。512² 建议用 `--resize -1` 保持原尺寸（推断） | **GH**（`git clone` 即可，权重已在仓库里） | **推荐 outdoor**（MegaDepth 训练；README 第 217-220 行推荐的 outdoor 设置是 `--superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float`）。脚本默认是 `indoor`（`match_pairs.py:94`），**必须显式改**。许可：仅限学术 / 非商业（LICENSE 第 1-3 行） |
| 14 | `rift2` | RIFT2（手工方法，无权重） | 无 | https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation/tree/0e980ce4124d2abd62727dcf040198db0a18b269 | `demo_RIFT2.m`（MATLAB） | H | **GH**；服务器上是否有 MATLAB **未核实** | 仓库自带 `sar-optical/pair1,2` 样例；没有 LICENSE 文件 |

---

## 2. 逐篇 data-engine 论文：论文里的权重 vs 实际发布

### 2.1 MatchAnything（He et al., TPAMI 2026；arXiv 2501.07556）
- 用户给的作者是 Jiang et al.，但论文首作者其实是 **Xingyi He**（`zju3dv/MatchAnything` README 第 5-11 行，读到）。
- 论文用这套框架训练了两个模型：ELoFTR 和 RoMa（arXiv HTML：「the ROMA [20] model trained with our framework … and the ELoFTR [78] model trained with our framework」，读到）。论文里没有看到其它架构的发布版本。

| 论文模型 | 发布了吗 | 文件 | 位置 |
|---|---|---|---|
| MA-ELoFTR | 是 | `matchanything_eloftr.ckpt` 64,366,723 B | GD zip（官方唯一一份原始 ckpt）；HF `zju-community/matchanything_eloftr@7bd52a4`（官方组织下的 transformers 移植，2025-08-21） |
| MA-RoMa | 是 | `matchanything_roma.ckpt` 445,649,931 B | **只有 GD zip**；zju-community 下**没有** RoMa 版（HF API 只列出 `efficientloftr` 和 `matchanything_eloftr`，读到） |

- 官方 zip 的来源：Space 内 README 第 37 行给出 GD 链接；Space 自己启动时也用 `gdown 12L3g9-…` 拉同一个 zip（`imcui/ui/app_class.py:25-26`，读到）。**GitHub 仓库只有 README 和 LICENSE，没有权重。**
- zip 的内容（通过 Range 读中央目录）：`weights/matchanything_roma.ckpt`（CRC32 31393b4a）、`weights/matchanything_eloftr.ckpt`（CRC32 fde5eff9）。GD 没有提供版本号，用 CRC32 作指纹。
- 第三方镜像：`vismatch/matchanything-eloftr@9ac6936`（64,251,040 B）、`vismatch/matchanything-roma@5f0b730`（445,502,400 B）；还有 `image-matching-models/matchanything-*` 和 `stevenbucaille/matchanything_eloftr`。
- **许可证变了**：`zju3dv/MatchAnything` 在 2026-09-15 的 `8cd8c11` 改为「Project Registration License v1.0」。学术研究、评测、发表**不需要登记**；只有机构的「项目使用」需要先登记（LICENSE 第 7-11、15 行，读到）。Space 和 HF 模型卡仍标 apache-2.0。

### 2.2 MINIMA（Ren et al., CVPR 2025；arXiv 2412.19412 v2）
- 论文里有 5 个模型：MINIMA_LG、MINIMA_LoFTR、MINIMA_RoMa，以及「further fine-tune ELoFTR and XoFTR … obtaining MINIMA_ELoFTR and MINIMA_XoFTR」（arXiv HTML，读到）。**5 个全部发布了**。
- 发布渠道：GitHub release `LSXI7/storage` tag `MINIMA`（2024-12-17 创建；eloftr/xoftr 两个文件 2025-02-18 上传，GitHub API 读到），以及 GD 文件夹 `16kZfehtXeIu6fJjUoYDzYb-3FkZBpkNd/weights`（同名 5 个文件，读到）。README 第 121-135 行。
- `weights/download.sh` 只下载 lg、loftr、roma 三个文件，**不含 eloftr 和 xoftr**（读到）。
- 官方推理路径：只有 `load_model.py` 里的 sp_lg / loftr / roma / xoftr；**ELoFTR 没有加载器**。HF Space `lsxi77777/MINIMA@1bddf24` 的 `ui/config.yaml` 也只暴露 3 个模型（splg、loftr、roma）。
- 第三方镜像：`vismatch/minima-{roma,loftr,xoftr}` 只有 yaml，没有权重文件，运行时回到 GitHub release 下载（推断）。

### 2.3 AnyMatch（身份确认）
- 论文：**Meng Yang, Zizhuo Li, Linfeng Tang, Fan Fan, Jiayi Ma, "AnyMatch: Supercharging Universal Multi-Modal Image Matching with Large-Scale Single-View Images", ECCV 2026**，arXiv 2606.31077（v1 2026-06-30，v2 2026-07-01），武汉大学马佳义组（和 MINIMA 同一个大组）。代码 `github.com/MnYangs/AnyMatch`（2026-07-02 创建，HEAD `259ad34`，2026-07-28）。
- 同名项目的歧义：GitHub 上叫 anymatch 的仓库还有 `micromatch/anymatch`（JS 字符串匹配）、`Jantory/anymatch`、`mityasmirnov/AnyMatch` 等，都与图像匹配无关。和「data engine + 发布现有匹配器的微调权重」对得上的只有这一篇。它的仓库 README 里 arXiv 徽章写成了 `2605.04730`，与实际编号 2606.31077 不一致，属于笔误（读到）。
- 论文微调了三个模型：LoFTR、EDM、RoMa（「We directly adopt the official pre-trained models of EDM, LoFTR, and RoMa as initialization weights」，读到）。**三个都发布了**，放在一个 GD zip 里，许可为「仅限学术研究」（`weights/readme.md` 第 11、56 行）。
- zip 内容（Range 读中央目录，读到）：

| 文件 | 字节 | CRC32 | 说明 |
|---|---|---|---|
| `LoFTR_AnyMatch.ckpt` | 138,979,993 | 5f92247f | 微调版（含优化器状态） |
| `LoFTR_original.ckpt` | 46,341,978 | 0fe4636b | 与 LoFTR `outdoor_ds.ckpt` **字节数完全一致**，推断就是它 |
| `EDM_AnyMatch.ckpt` | 41,417,503 | 49db285e | 微调版 |
| `EDM_original.ckpt` | 41,417,503 | 7d6c6de4 | 原版 EDM |
| `RoMa_original.pth` | 445,647,516 | 93d80eba | 与 `Parskatt/storage` 的 `roma_outdoor.pth` **字节数完全一致** |
| `RoMa_AnyMatch.pth` | 1,336,376,603 | 26b83cfb | 微调版（含优化器状态） |

- 论文中还有未发布的模型：Tab. 4 / Tab. 5 消融里的 EDM 变体（只用 RGB-IR 训练、从头训练、在 homo-syn 上训练、不同 η 阈值）**没有发布**。对我们来说不需要。

---

## 3. 与加载相关的 ckpt 结构（读 pickle 字符串得到，没有用 torch 真正加载）

| 文件 | 顶层结构 | state_dict key 前缀 | 含 DINOv2？ |
|---|---|---|---|
| matchanything_eloftr.ckpt | `state_dict` | `matcher.` | — |
| matchanything_roma.ckpt | `state_dict` | `matcher.model.` | 否 |
| minima_lightglue.pth | 裸 state_dict | 无前缀（LightGlue 模块名） | — |
| minima_loftr / minima_xoftr / minima_eloftr.ckpt | `state_dict` | `matcher.` | — |
| minima_roma.pth | 裸 state_dict | `encoder.` / `decoder.` | 否 |
| LoFTR_AnyMatch.ckpt | Lightning 完整 ckpt（`state_dict`、`optimizer_states`、`lr_schedulers`、`callbacks`、`epoch`、`global_step`） | `matcher.` | — |
| RoMa_AnyMatch.pth | `model`、`optimizer`、`lr_scheduler`、… | `encoder.` / `decoder.` | 否 |

---

## 4. 阻碍与意外

1. **AnyMatch 的 `load_model.py` 不存在**：`demo/demo_anymatch.py:17` 和 `test_relative_pose_infrared.py:25` 都 `from load_model import load_model`，但仓库 HEAD `259ad34` 里没有这个文件。`raw.githubusercontent` 上 `load_model.py`、`demo/load_model.py`、`src/utils/load_model.py` 都返回 404，git tree 也没有（读到）。评测框架明显是从 MINIMA 分叉来的（同样的 `src/utils/data_io_*.py`，README 第 224 行甚至原样保留了「initialized from the MINIMA models」），所以可以仿照 MINIMA 的 `load_model.py` 自己写 loader（推断）。EDM 的加载需要看 `third_party/EDM_AnyMatch/test.py` / `src/lightning`。**这 3 个 AnyMatch run 实际上都需要我们自己补 loader。**
2. **Google Drive 独占（必须 PC 下载后 scp）**：MatchAnything 两个权重（zip 482.7 MB）、AnyMatch 三个权重（zip 1.89 GB）、XoFTR 640、LoFTR outdoor_ds。本机 PC 已验证能访问 GD（能拿到 Content-Range）。捷径：AnyMatch zip 里已经带了 `LoFTR_original.ckpt`（字节数与 outdoor_ds 相同），下载这一个 zip 就同时拿到 run 8、9、10、12 需要的权重。
3. **DINOv2（1.22 GB）**：MA-RoMa、MINIMA-RoMa、AnyMatch-RoMa 三个 run 都会在构造模型时从 `dl.fbaipublicfiles.com` 下载。服务器能否访问**未核实**。如果不能，就在 PC 下载后 scp，再通过 `dinov2_weights=` 参数传入，或放进 torch hub 缓存（推断）。
4. **hf-mirror 的 LFS 文件会 302 跳到 `us.aws.cdn.hf.co`**（本机实测 `vismatch/matchanything-roma` 的 safetensors）。服务器能否跟随这个跳转**未核实**；小的 json 文件直接由 hf-mirror 返回。
5. **MINIMA-ELoFTR 没有官方推理代码**（MINIMA 仓库和 Space 都没有），只能用 EfficientLoFTR 官方代码加载，属于「半官方」路径。
6. **MINIMA-XoFTR 默认 ckpt 其实是原版 XoFTR**（`load_model.py:177`），不显式传 `--ckpt` 的话结果会与 `xoftr` run 重复。
7. **LoFTR outdoor_ds 必须用 `TEMP_BUG_FIX=False`**。MINIMA 会根据文件名自动处理（`load_model.py:49-51`）；自己写 wrapper 时如果用 `src/config/default.py`（默认 True）就会出错。
8. **MatchAnything GitHub 的 LICENSE 改成了 PRL v1.0**（2026-09-15），研究用途不受影响。
9. 用户提示里 MatchAnything 的首作者写成「Jiang et al.」，实际是 **He et al.**。

---

## 5. 来源（2026-09-25 读取）

**GitHub 仓库 / release**
- https://github.com/zju3dv/MatchAnything @ `8cd8c1129a6d22dabea9405a869e4fad6ff8b630`（README、LICENSE）
- https://huggingface.co/spaces/LittleFrog/MatchAnything @ `6a7bcb589ec8da3a9e861e799122beaa5eba2193`（本地克隆自 hf-mirror；`imcui/ui/app_class.py`、`imcui/hloc/matchers/matchanything.py`、`imcui/hloc/match_dense.py:47-82`、`imcui/third_party/MatchAnything/README.md`、`third_party/ROMA/roma/models/encoders.py`）
- https://github.com/LSXI7/MINIMA @ `796e7721174f9f829b79b3702bf8c2ae9a3d447a`（README 第 121-135 行、`weights/download.sh`、`load_model.py`）
- https://github.com/LSXI7/storage/releases/tag/MINIMA（通过 GitHub API 取资产大小和日期）
- https://huggingface.co/spaces/lsxi77777/MINIMA @ `1bddf24a4d744b20988de0a38471f938124ab83f`（`ui/config.yaml`）
- https://github.com/MnYangs/AnyMatch @ `259ad343ffa9ca0703965f56880b3b5d2ecbc9e1`（README、`weights/readme.md`、`weights/download.sh`、`demo/demo_anymatch.py`、`test_relative_pose_infrared.py`、`src/utils/data_io_*.py`、`src/config/default.py`、`third_party/EDM_AnyMatch/README.md`；git tree 通过 API 读取）
- https://github.com/zju3dv/LoFTR @ `df7ca80f917334b94cfbe32cc2901e09a80e70a8`（README 第 55-89 行、`src/loftr/utils/cvpr_ds_config.py:28`、`src/config/default.py:23`、`configs/loftr/outdoor/buggy_pos_enc/loftr_ds.py`）
- https://github.com/zju3dv/EfficientLoFTR @ `07e9c1401e71e9c556b1fda9d86d060af36ba176`（README 第 33-52、88-109 行，`src/config/default.py:28-29`）
- https://github.com/OnderT/XoFTR：本地克隆 `e9635d8baf95b5731bb1a142eff9a479a99e1e3b`；上游 `e0fbea431b30be9742effbf5577c90aa8eb938f9`（README 第 24、84 行，`src/config/default.py:189-190`）
- https://github.com/magicleap/SuperGluePretrainedNetwork @ `ddcf11f42e7e0732a0c4607648f9448ea8d73590`（README 第 24、217-220 行，`match_pairs.py:85,94`，`models/utils.py:260-264`，`models/superglue.py:224-225`，`models/superpoint.py:136`，LICENSE）
- https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation @ `0e980ce4124d2abd62727dcf040198db0a18b269`
- https://github.com/Parskatt/storage/releases/tag/roma（`roma_outdoor.pth` 445,647,516 B）
- https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth（5,206,086 B）

**Google Drive（通过 `drive.usercontent.google.com` 的 Content-Range 取大小，用 Range 读 zip 中央目录，用 `embeddedfolderview` 列文件夹）**
- MatchAnything `weights.zip`：https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d（482,746,196 B）
- AnyMatch `Weights_AnyMatch.zip`：https://drive.google.com/file/d/1JCz52pM7On_YWCjBWr_VuTiXt1uOXPD_（1,889,874,277 B）
- MINIMA 权重文件夹：https://drive.google.com/drive/folders/16kZfehtXeIu6fJjUoYDzYb-3FkZBpkNd
- LoFTR 权重文件夹：https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp（indoor_ds_new 46,355,053 B；indoor_ot 46,342,243 B；outdoor_ds 46,341,978 B；outdoor_ot 46,342,243 B）
- XoFTR 权重文件夹：https://drive.google.com/drive/folders/1RAI243OHuyZ4Weo1NiTy280bCE_82s4q（640 / 840 各 44,486,099 B）

**HuggingFace（API `api/models/<id>?blobs=true`）**
- zju-community/matchanything_eloftr @ `7bd52a4d5e2ca0f7c4edfaa518a25fb1cd6eea47`（README、`preprocessor_config.json`）
- vismatch/matchanything-roma @ `5f0b730663731dacffcc636265ef81220712d10e`；vismatch/matchanything-eloftr @ `9ac693698883b489c180b7ad8dd87e60281ba847`；vismatch/xoftr @ `d8ee7d89be3c9e5c157db3886db1c0f0e038b321`；vismatch/minima-roma @ `d350ee7`、minima-loftr @ `8c56e47`、minima-xoftr @ `b68b913`（后三个只有 yaml）

**论文**
- MatchAnything：https://arxiv.org/abs/2501.07556（HTML）
- MINIMA：https://arxiv.org/abs/2412.19412（v2，HTML）
- AnyMatch：https://arxiv.org/abs/2606.31077（v2 HTML）；Springer ECCV 2026 章节 https://link.springer.com/chapter/10.1007/978-3-032-37718-0_12
- DINOv2 权重：https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth（1,217,586,395 B，Last-Modified 2023-04-13）
