# 月面光学–SAR 仿射配准：zero-shot 基线方法调研

> 调研日期：2026-09-25
> 任务背景：CE-2 光学（7.5 m，uint8 灰度）↔ Mini-RF SAR（4 波段 Stokes float32；本项目统一的 SAR 量 = S1 = b1+b2 总功率，转 dB 并下限截断在 −25 dB），512×512 patch 对。
> 基线一律 **zero-shot**（官方预训练权重，不在我们的数据上微调），匹配结果统一交给同一个仿射 RANSAC。
>
> 标注约定：**「读到」** = 我直接读了官方代码或 README 的对应行（`文件:行号`，行号对应 [sources.md](sources.md) 里记下的 commit）；**「推断」** = 我的判断，已显式标出。
> 分方法的明细见 [deep-methods.md](deep-methods.md)（深度方法）和 [traditional-methods.md](traditional-methods.md)（传统/手工方法）。

---

## 0. 第一批（batch 1，用户已定，2026-09-25）

用户定的第一批：三篇数据引擎论文（MatchAnything、MINIMA、AnyMatch）放出的全部多模态微调权重，再加 XoFTR、LoFTR、SuperPoint+SuperGlue、RIFT2。共 14 个 run。下面第 1 节的「推荐 12 个」是更早的候选，仅作参考。

- [batch1-weights.md](batch1-weights.md)：14 个 run 各自的权重来源、下载地址、代码路径和阻碍。
- [interfaces-a.md](interfaces-a.md)：代码级接口，覆盖 LoFTR、XoFTR、SuperPoint+SuperGlue、RIFT2。
- [interfaces-b.md](interfaces-b.md)：代码级接口，覆盖 MatchAnything（ELoFTR、RoMa）和 MINIMA（5 个匹配器）。
- [modality-normalization.md](modality-normalization.md)：跨模态方法和光-SAR 数据集怎样归一化不同传感器的输入（主流做法：SAR 先逐图拉伸成普通灰度图，再走匹配器原生 loader）。

**更正**：XoFTR 在 forward 里对每张图各自做 z-score（`src/xoftr/xoftr.py:39-47`），应归入第 2 节的情形 C，不是情形 A。

---

## 1. 结论先行：推荐基线清单（12 个）

| # | 方法 | 类别 | 一句话理由 |
|---|---|---|---|
| 1 | **MatchAnything-RoMa** (TPAMI 2026) | 深度·跨模态·通用 | 官方有 visible–SAR 评测脚本且用的就是 `ransac_affine`；论文 SR@10px = 93.3%，是目前读到的 visible–SAR 最强 zero-shot 数字。 |
| 2 | **MatchAnything-ELoFTR** (TPAMI 2026) | 深度·跨模态·通用 | 同一套预训练、半稠密架构，SR@10px = 72.5%；官方权重在 HF `zju-community/matchanything_eloftr`（Apache-2.0），经 hf-mirror 可下载。 |
| 3 | **MINIMA-RoMa** (CVPR 2025) | 深度·跨模态·通用 | 第三方 SAR–光学 zero-shot 基准（Corley et al. 2026）里，在 SRIF 光学–SAR 子集上是唯一 0% 失败的方法；权重放在 GitHub release，可直接下载。 |
| 4 | **XoFTR** (CVPRW 2024) | 深度·跨模态(可见–热红外) | SpaceNet9 SAR–光学 zero-shot 平均误差 3.0 px，并列第一，而且比 RoMa 快约 10 倍。灰度 /255 的输入约定最简单。 |
| 5 | **ReDFeat (VIS_SAR 权重)** (TIP 2023) | 深度·**光学–SAR 专门训练** | 唯一一个权重就在 git 仓库里、并且是用光学–SAR（OSdataset，GF-3 512×512）训练的稀疏特征；在 MIFNet 论文的光学–SAR 表里 SRR = 41.4%，高于 MIFNet。作为「SAR 监督」参照。 |
| 6 | **FHReg (GUSO)** (ISPRS 2026) | 深度·**光学–SAR 专门训练** | 最新的 SAR–光学专用配准网络，官方声称能 zero-shot 迁移到 OSdataset/MSAW。**条件入选**：权重只在 Google Drive（需要在 PC 下载后传到服务器）。 |
| 7 | **RIFT2** (2023) | 传统·跨模态 | 相位一致性系方法里的事实标准基线，MATLAB 源码完整、没有 p-code。 |
| 8 | **HIMO** (TPAMI 2026)（或其前身 HOMO-Feature, ICCV 2025） | 传统·跨模态 | 2025–2026 最新的纯手工「跨任意模态」方法，MATLAB 核心是 `.m` 源码。 |
| 9 | **OS-SIFT** (TGRS 2018, 代码 v1.5 2023) | 传统·**光学–SAR 专用、SAR-aware** | 唯一对 SAR 做比值梯度（ROEWA）的方法，用来检验我们 SAR 输入语义（线性 vs dB）是否正确。代码是源码。 |
| 10 | **SRIF** (ISPRS 2023) | 传统·跨模态 | 与 RIFT 同作者的尺度+旋转不变升级版；同一个仓库还打包了 LNIFT、3MRS、MS-HLMO、CoFSM 的 demo。**注意：只有 Windows exe**。 |
| 11 | **RoMa v2** (arXiv 2025-11) | 深度·**非跨模态**通用锚点 | 当前稠密匹配 SOTA，权重在 GitHub release。 |
| 12 | **SuperPoint+LightGlue**（可换成 LoMa-B, ECCV 2026） | 深度·**非跨模态**稀疏锚点 | 最常用的稀疏基线。LoMa 是同一团队 2026 年更强的替代品。 |

可选/备选（有价值，但有阻碍或重复）：RoMa v1（MA-RoMa、MINIMA-RoMa 的母体，适合做消融）、ELoFTR（MA-ELoFTR 的母体）、MIFNet（TIP 2025，有 opt-sar 模式，但依赖 SD-2.1 且在其论文里不如 ReDFeat）、VMGGA（ISPRS 2026，光学–SAR 专用权重只在百度网盘）、HAPCG / MS-HLMOv2 / LNIFT / sRIFD（传统方法，见明细）。

---

## 2. 输入处理情形分类（taxonomy）

对**每个方法的官方推理路径**逐一归类。这一节是接 wrapper 时最需要看的：**每个方法必须按它自己的约定喂数据**，不能用一个统一的 loader 喂所有方法。

### 情形 A：8-bit 灰度 → 官方 loader 读入 → ÷255 → float[0,1]，之后不再归一化
- **成员**：LoFTR、ELoFTR、XoFTR、MINIMA-LoFTR / MINIMA-XoFTR、MatchAnything-ELoFTR、CasP（`data_mode=="gray"`）、VMGGA。
- 依据（读到）：ELoFTR `README.md:57-66`（`cv2.IMREAD_GRAYSCALE`、`/255.`）；XoFTR `src/utils/data_io.py:46-47,76`；MatchAnything `src/utils/dataset.py:255-257`，评测时 `read_gray=True, normalize_img=False`（`tools/evaluate_datasets.py:137`）；VMGGA `demo_vmgga.py:318-327`；CasP `demo.py:137-141`。
- 对我们的含义：光学 uint8 直接用；SAR 侧需要先变成 8-bit 灰度图（dB 截断后线性映射到 0–255）。这些方法对全局线性缩放**不**不变（没有内部归一化），所以 dB→uint8 用什么范围映射会影响结果（推断）。

### 情形 B：RGB float[0,1] + 在匹配器内部做 ImageNet mean/std
- **成员**：RoMa（`romatch/utils/utils.py:164-172`）、RoMa v2（`normalizers.py:4-9`，在 `features.py` 里对输入调用）、DKM（`get_tuple_transform_ops(normalize=True)`，`dkm/models/dkm.py:679-681`）、MINIMA-RoMa（通过其 RoMa 分叉）、LoMa（DaD 读图 `convert("RGB")`、`/255`，`detector/dad.py:167-180`）。
- 注意：**RoMa 要求 3 通道**（`matcher.py:548` 断言 `C == 3`；路径输入时 `convert("RGB")`）；**DKM 用路径输入时不会 `convert("RGB")`**（`dkm.py:663`），灰度 PNG 会以 1 通道张量进入，直接撞上 3 通道的 ImageNet Normalize（推断：会报错）→ 必须先把 PIL 图转成 RGB。RoMa v1 **张量输入不会自动做归一化**（`matcher.py:832-841`，直接用），所以喂张量就必须自己先做 ImageNet 归一化；RoMa v2 的张量输入只在 dtype 为 uint8 时才 ÷255（`romav2.py:294-295`）。

### 情形 B′：RGB float[0,1]，**不做** ImageNet 归一化（主干虽然是 DINOv2）
- **成员**：MatchAnything-RoMa。评测配置 `NORMALIZE_IMG=False`（`src/config/default.py:7`），`self_inference_time_match(norm_img=False)` 从原图 `convert("RGB")`、`/255.` 后不做 Normalize（`third_party/ROMA/roma/models/matcher.py:649-677`），resize 采用拉伸（`roma_model.py:2-3` 中 `RESIZE_BY_STRETCH=True`）。
- 风险：如果用 vismatch 或 RoMa 原版流水线去加载 MA-RoMa 的权重，很容易多做一次 ImageNet 归一化，从而偏离官方约定（推断）。

### 情形 C：模型内部做逐图标准化（InstanceNorm / z-score），因此对全局线性灰度变换基本不变
- **成员**：MIFNet 里的 XFeat 分支（`×255` 后在 `extract_raw_map` 里做通道均值 + `self.norm`，`xfeat_engine.py:173-174,307`）、RIPE（`use_instance_norm=True`，`backbones/vgg.py:11`；灰度会被复制成 3 通道，`backbone_base.py:45-53`）、ReDFeat（脚本里逐通道 z-score，`match.py:141,161`）、FHReg（stem 是 patchify 卷积 + LayerNorm2d，`wavelet_backbone.py:247-250`，推断为近似不变）。
- 对我们的含义：dB 值怎么映射到 0–255 影响不大，但 dB 这种**非线性**变换本身仍然会改变输入分布。

### 情形 D：官方路径把图缩放到固定尺寸或固定长边（常常是**拉伸**，而且对 512 输入是**放大**）
| 方法 | 官方尺寸 | 保持长宽比？ | 读到 |
|---|---|---|---|
| RoMa v1 | coarse 560×560 → upsample 864×864 | 否（直接 resize 成方形） | `model_zoo/roma_models.py:35-36`；README「Resolution」 |
| RoMa v2 `precise`（默认） | 800×800 / 1280×1280 | 否 | `romav2.py:79,152-158` |
| DKM v3 outdoor | 540×720 → 864×1152 | 否（512² 会被拉成 4:3） | `model_zoo/__init__.py:15-25` |
| MatchAnything | 长边 832（脚本）/ 840（论文） | 是 | `eval_visible_sar.sh:14,17` |
| XoFTR / MINIMA | 长边 640，df=8 | 是 | `src/config/default.py:189-192` |
| LightGlue 提取器 | 长边 1024 | 是 | `superpoint.py:115-117`、`disk.py:17-18`、`aliked.py:631-632` |
| MIFNet (XFeat+SD) | 768×768 | 否 | `configs/xfeat.yaml:3-4,12` |
| CasP | 长边 1152（步长 32） | 是 | `app.py:195-197`、`demo.py:59` |
| LoMa (DaD) | 长边 1024 | 是 | `detector/dad.py:36,173-178` |
| RIPE | 每边夹到 [512, 768] | — | `ripe/utils/utils.py:57-77` |

### 情形 E：只要求尺寸是 k 的整数倍（不改变尺度）
- ELoFTR：÷32（`README.md:60-62`）；LoFTR / XoFTR：÷8（df=8）；VMGGA：右下补边到 16 的倍数，而不是缩放（`demo_vmgga.py:323-324`）；RoMa 张量输入：必须是 14 的倍数（`matcher.py:549-550`）；XFeat：内部 resize 到 32 的倍数（`xfeat_engine.py:155-160`）。512 本身满足所有这些约束。

### 情形 F：输入**不对称**（哪张是光学、哪张是 SAR 会影响结果）
- OS-SIFT：图 1 走 `build_scale_opt`，图 2 走 `build_scale_sar`，两者梯度算子不同（`os_sift.m:8-11,34-35`）。
- ReDFeat：`forward1` 对应可见光、`forward2` 对应 SAR/其他模态（`match.py:75-78,144,165`）。
- FHReg：batch 的键名就是 `image_opt` / `image_sar`（`demo.py:63`）。
- VMGGA：image0 = 参考（光学），image1 = 待配准；权重按模态分开发布（README Model Zoo）。

### 情形 G：SAR-aware，假设乘性斑点噪声作用在**线性**强度/幅度上，且把 0 当作无效值
- OS-SIFT：SAR 侧梯度 = `log(M14./M23)`，即 ROEWA 比值（`build_scale_sar.m:31-37`）。比值对全局乘性缩放不变，但前提是输入为非负的线性量。另外 v1.4 以后把 `image==0.001`（原值为 0）的像素视为无效区并 mask 掉（`build_scale_sar.m:7`；README v1.4 说明）。
- SAR-SIFT（Dellinger 2015）：同为 ROEWA 比值梯度。没有官方代码，只有第三方实现（见明细）。
- 对我们的含义（推断）：**这两个方法的 SAR 侧应该喂线性 S1**（或其平方根幅度），缩放到 [0,1]，而不是 dB。如果喂 dB→uint8，比值算子等于在对数域上再取比值，物理意义不对；而且 −25 dB 截断后映射成 0 的像素会被当成无效区。官方 demo 还会先做一次 `imadjust`，即 1% 两端饱和拉伸（`os_sift.m:14-15`）。

### 情形 H：MATLAB `imread` → `im2uint8`/`uint8`/`im2double` → `rgb2gray` → 相位一致性等（坐标从 1 开始）
- RIFT / RIFT2：`im2uint8(imread)`，灰度图复制成 3 通道，`FeatureDetection` 里再转回灰度，然后算 `phasecong3`（`demo_RIFT2.m:6-15`，`FeatureDetection.m:3-8`）。
- sRIFD：沿用 RIFT2 框架。HAPCG：`rgb2gray` + `im2double`（`HAPCG_nonelinear_space.m:43,48`）。CoFSM / MS-HLMOv2 / HOMO / HIMO：读图和预处理在 **p-code**（`Readimage.p`、`Preproscessing.p`、`Deal_Extreme.p`）里，**无法核实**它们具体做了什么归一化。
- 相位一致性本身对对比度不敏感。输出坐标是 MATLAB 的 1-based 像素坐标，进入统一 RANSAC 前要 −1（推断，基于 MATLAB 图像坐标约定）。

### 情形 I：Windows 二进制，通过文件交互
- SRIF.exe、LNIFT.exe、3MRSMatcher.exe：读两张 PNG，写出 `matches.txt`（`demo_SRIF.m:20-24,29`；`demo_LNIFT.m:24`；`demo_3MRS.m:24`）。CFOG / HOPC / FED-HOPC 依赖 `.mexw64`（只能在 Windows 上跑），并且需要**初始控制点**来确定搜索区域（`matchFramework.m:5,122-136`）。这类方法属于面匹配/模板匹配，每个模板只搜平移。
- 输出坐标的约定（0-based 还是 1-based）二进制里看不到，需要先用已知平移的合成对标定（推断）。

### 情形 J：不做 ÷255 的 float 0–255 输入 / 原始 uint8 张量
- FHReg `demo.py` 用 rasterio 读成 float32 0–255，没有 ÷255（`demo.py:92-106,202-203`）；但它的 zero-shot 评测脚本用 `TF.to_tensor(uint8)`，得到的是 [0,1]（`zero_shot_msaw_os.py:116-117`）。**两条官方路径不一致**，建议以 zero-shot 脚本为准（推断）。
- MapGlue（HF Space）把 uint8 RGB 张量直接交给 TorchScript 模型（`app.py:132-149`），但模型文件需要 HF_TOKEN 才能下载，**事实上拿不到**。

### 情形 K：HF transformers 的 `AutoImageProcessor` 路径
- ELoFTR 和 MatchAnything-ELoFTR 都有 transformers 版本。ELoFTR README 明确写了默认 processor 会把图 resize 到 **480×640**（`README.md:170`）。如果走这条路，必须改 processor 配置，否则 512² 的图会被压成 4:3（推断）。

### 输出坐标约定（统一 RANSAC 前必须对齐）
- **RoMa / RoMa v2 / DKM**：warp 在 [-1,1] 网格上（`align_corners=False`），`to_pixel_coordinates` 用的是 `W/2*(x+1)`（`romatch/models/matcher.py:730`）。也就是说，第 i 个像素中心对应 **i+0.5**，与「像素中心在整数」的约定差 0.5 px（推断）。
- **LightGlue**：`(kp+0.5)/scale-0.5`，还原到原图坐标，像素中心在整数（`lightglue/utils.py:146`）。
- **LoFTR 系**（XoFTR / MINIMA wrapper）：先在 resize 后的网格里输出，再乘 `scale = w/w_new` 回到原图（`data_io.py:67,87-88`）。**ELoFTR README 的基础用法不会帮你乘回去**。
- **MIFNet**：关键点在 **768×768 网格**里，脚本不会映射回原图（`test_xfeat_mifnet.py:92-94`；`xfeat_engine.py:257` 只修正了 32 对齐带来的比例）。
- **MATLAB 方法**：1-based。

---

## 3. 逐方法总表（精简版；细节和出处见明细文件）

图例：跨模态 = 方法本身是否为跨模态设计；O–SAR 评测 = 官方论文/仓库里是否评测过光学–SAR；权重来源中 GH = GitHub release/仓库，HF = HuggingFace（服务器需走 hf-mirror），GD = Google Drive（服务器大概率访问不了），BD = 百度网盘。

| 方法 | 年份/会议 | 跨模态 | O–SAR 评测 | 语言/许可 | 权重来源 | 输出 | 输入情形 |
|---|---|---|---|---|---|---|---|
| MatchAnything-RoMa | TPAMI 2026 | 是（合成多模态预训练） | 是，visible–SAR 1209 对，SR@10px 93.3% | Py / Apache-2.0 | GD（官方）；HF `vismatch/matchanything-roma`（第三方镜像） | 稠密 warp → 采样 5000 对 | B′ + D(832，拉伸) |
| MatchAnything-ELoFTR | TPAMI 2026 | 是 | 是，72.5% | Py / Apache-2.0 | HF `zju-community/matchanything_eloftr`（官方 transformers 版）；GD | 半稠密 | A + D(832) / K |
| MINIMA-RoMa | CVPR 2025 | 是（MD-syn） | 是，MMIM 遥感 7 类混合：AUC@10 64.38 | Py / Apache-2.0 | GH release `LSXI7/storage` | 稠密 → 采样 | B + D(640→560/864) |
| MINIMA-LoFTR / -SP+LG / -XoFTR | CVPR 2025 | 是 | 同上 | Py / Apache-2.0 | GH release | 半稠密 / 稀疏 | A + D(640) |
| XoFTR | CVPRW 2024 | 是（可见–热红外） | 官方无；第三方 SpaceNet9 3.0 px | Py / Apache-2.0 | GD（官方）；HF `vismatch/xoftr` | 半稠密 | A + D(640) |
| ReDFeat | TIP 2023 | 是 | 是，OSdataset（训练集就是它） | Py / **无 LICENSE** | GH 仓库内 `Pretrained/VIS_SAR.pth` | 稀疏 kp+desc（多尺度） | C + F |
| FHReg (GUSO) | ISPRS 2026 | 光学–SAR 专用 | 是（GUSO / OSdataset / MSAW） | Py / MIT | GD | 半稠密 → FSC | J + F |
| MIFNet | TIP 2025 | 是（单模态训练） | 是，OSdataset 400 对，SRR 36.1–39.7% | Py / MIT | Dropbox；另需 SD-2.1（HF） | 稀疏 | C + D(768，拉伸) |
| VMGGA | ISPRS 2026 | 是（按模态分别训练） | 是 | Py / Apache-2.0 | **仅 BD**（GD 标注 TBA） | 半稠密 | A + E(16) + F |
| CasP | ICCV 2025 | 否（另有 MINIMA 微调版） | 仅 demo gif | Py / Apache-2.0 | **gated**（HF 私有，需要作者给 token） | 半稠密 | A + D(1152) |
| MapGlue | arXiv 2025 | 是 | 是（demo 有 SAR 例子） | 仅 Space | **gated**（HF_TOKEN） | 稀疏 | J |
| RIPE | ICCV 2025 | 否 | 否 | Py / Fraunhofer 学术许可 | 自建服务器 `cvg.hhi.fraunhofer.de` | 稀疏 | C |
| LoFTR | CVPR 2021 | 否 | 否 | Py / Apache-2.0 | GD；HF `vismatch/loftr` | 半稠密 | A + E(8) |
| ELoFTR | CVPR 2024 | 否 | 否 | Py / Apache-2.0 | GD；HF `zju-community/efficientloftr` | 半稠密 | A + E(32) / K |
| RoMa | CVPR 2024 | 否 | 第三方 SpaceNet9 3.0 px | Py / MIT（DINOv2 为 Apache） | GH release | 稠密 | B + D |
| RoMa v2 | arXiv 2025 | 否 | 否（仓库有 SatAst 卫星–宇航员基准） | Py / MIT（DINOv3 自定义许可） | GH release | 稠密 | B + D(800/1280) |
| DKM v3 | CVPR 2023 | 否 | 否 | Py | GH release | 稠密 | B + D(540×720) |
| SP+LightGlue | ICCV 2023 | 否 | 否 | Py / Apache-2.0（SuperPoint 限制性许可） | GH release | 稀疏 | A(提取器内部转灰度) + D(1024) |
| LoMa | ECCV 2026 | 否 | 否 | Py / MIT | GH release | 稀疏 | B + D(1024) |
| RIFT2 | 2023（arXiv/期刊） | 是 | demo 自带 sar-optical 样例 | MATLAB / 无 LICENSE | —（无需权重） | 稀疏 | H |
| RIFT | TIP 2020 | 是 | 同上 | MATLAB / BSD-3 式 | — | 稀疏 | H |
| SRIF | ISPRS 2023 | 是 | 是（自带 Optical-SAR 数据集） | **Windows exe** | — | 稀疏 | I |
| LNIFT | TGRS 2022 | 是 | 是（SRIF 仓库） | **Windows exe** | — | 稀疏 | I |
| 3MRS | 2022 | 是 | 是 | **Windows exe** | — | 稀疏 | I |
| OS-SIFT | TGRS 2018 / 代码 2023 | 光学–SAR 专用 | 是 | MATLAB（CSC_match 为 .p，另有 CSC2.m 源码） | — | 稀疏 + CSC 过滤 | G + F |
| HAPCG | 武大学报 2021 | 是 | 是 | MATLAB，核心 .p | — | 稀疏 | H |
| MS-HLMOv2 | IGARSS 2024 | 是 | 是 | MATLAB，I/O 为 .p | — | 稀疏 | H(p-code) |
| HOMO-Feature | ICCV 2025 | 是（跨任意模态） | 是 | MATLAB，I/O 为 .p | — | 稀疏 | H(p-code) |
| HIMO | TPAMI 2026 | 是 | 是 | MATLAB，I/O 为 .p | — | 稀疏 | H(p-code) |
| sRIFD | IPT 2023 | 是 | 否（LC09 可见–热红外样例） | MATLAB 源码 | — | 稀疏 | H |
| CFOG / HOPC / FED-HOPC | TGRS 2019 / 2017 / 2023 | 是 | 是 | MATLAB + **mexw64** | — | 模板匹配（需初值） | I |
| SAR-SIFT | TGRS 2015 | SAR–SAR | — | **无官方代码** | — | 稀疏 | G |
| POS-GIFT | Inf. Fusion 2023 | 是 | 是 | **未找到公开代码** | — | — | — |

---

## 4. 阻碍 / 风险清单

1. **Google Drive 独占的权重**（服务器大概率访问不了）：XoFTR、ELoFTR、LoFTR、MatchAnything（官方 zip）、GIM、FHReg。其中前四个在 HF `vismatch/*`（第三方镜像，safetensors）或 `zju-community/*`（官方 transformers 版）有替代，可以走 hf-mirror。**第三方镜像要和官方权重核对**（推断：vismatch 可能做过 key 重命名）。**FHReg 没有镜像**，只能在 PC 上下载后 scp 到服务器。
2. **拿不到权重**：CasP（README 明说权重只通过 demo 提供，HF 仓库需要 token）、MapGlue（TorchScript 需要 HF_TOKEN，代码也没开源）。服务器上虽然有 CasP 的代码克隆，但**没有公开权重就无法 zero-shot**。VMGGA 的光学–SAR 权重只在百度网盘，需要人工下载。
3. **只有 Windows 的传统方法**：SRIF / LNIFT / 3MRS 只有 exe；CFOG / HOPC / FED-HOPC 依赖 mexw64。如果服务器是 Linux，需要一台 Windows 机器或 Wine（推断）。另外 CFOG 这类方法需要初始控制点，对大仿射扰动天然不适用。
4. **p-code 黑箱**：HOMO / HIMO / MS-HLMOv2 / HAPCG / CoFSM 的读图和预处理在 `.p` 里，输入归一化无法核实；.p 可以在 Linux MATLAB 上运行。
5. **服务器现有克隆的版本问题**：XoFTR `e9635d8`（2024-09）早于上游修复 `np.float → np.float32`（`e0fbea4`，2025-08，commit message 读到）。MINIMA `796e772` 的 `data_io_roma.py:77`、`data_io_loftr.py:65` 也还用着 `np.float`，在 numpy ≥ 1.24 下会报错（推断：`np.float` 在 1.24 被移除）。MINIMA 的 `load_model()` 会把 `test_orginal_megadepth` 参数传给 `load_xoftr(args)`，但后者不接受这个参数（`load_model.py:143,160`，推断会 TypeError）。RoMa 服务器版本 `edd1b8b`（2025-02）比我读的 `77f8d68`（2026-01）旧，输入 API 需要在服务器上再核对一遍。
6. **SD-2.1 依赖**：MIFNet 要从 HF 下载 `stabilityai/stable-diffusion-2-1`（`README.md:39-44`），这个仓库现在是否还能下载**未核实**。
7. **RIPE 许可证**只允许非商业学术评测（`LICENSE:11`），权重放在 Fraunhofer 自己的服务器上（`vgg_hyper.py:20`）。

---

## 5. 与「光学–SAR zero-shot」最直接相关的第三方证据

Corley, Stoken, Berton, *Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?*（arXiv 2604.10217，2026-04/08；代码 `isaaccorley/rsim@9950822`）：24 个配置做 zero-shot 评测，用 OpenCV `estimateAffine2D`，阈值 3 px，最少 4 个内点。
- SpaceNet9：RoMa 和 XoFTR 3.0 px 并列第一；RoMa+LoFTR 3.3；MA-ELoFTR 3.4；RoMa v2 3.6；LoFTR 5.1。
- SRIF 光学–SAR（600 对）：只有 MINIMA-RoMa 做到 0% 失败（47.0 px，z-score 归一化）。
- 仿射模型本身就把平均误差从 12.3 降到 9.7 px。这支持我们「统一仿射 RANSAC」的设计。
- 他们对输入归一化做了扫参（identity / percentile 2–98 / z-score / CLAHE / SAR-log-percentile，`spacenet9_matcher_benchmark.py:367-480`），结论是**最优归一化因方法而异**（RoMa 系偏好 z-score，XoFTR 偏好 percentile）。

对我们的启示（推断）：「官方输入约定」和「最优输入」可能不同。建议主表严格按官方约定跑，另外单独做一个 SAR 映射方式的小消融（dB 线性映射 vs percentile vs 线性强度），而不是在主表里为每个方法各自调参。

---

## 6. 未完成 / 未核实

- 传统方法在光学–SAR 上的论文数值（SRIF 表、HOMO/HIMO 表）：PDF 有加密，这次没能读出，没有写数字。
- MatchAnything 论文的 visible–SAR 数据集引用 [82]，完整出处没能从 HTML 里读出；评测分辨率论文写 840，脚本写 832，两者不一致。
- DKM、LightGlue 的具体 CUDA/torch 版本：只记了 README/pyproject 里写到的部分。
