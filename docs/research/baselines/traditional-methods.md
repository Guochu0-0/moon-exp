# 传统（手工）方法明细

> 行号对应 [sources.md](sources.md) 中的 commit。「读到」= 直接读了代码；「推断」= 我的判断。
> 这些方法都不需要权重，关键是：代码能不能跑（源码 / p-code / Windows 二进制）、要装哪些 MATLAB 工具箱、输入约定和坐标约定是什么。

通用说明（推断，基于 MATLAB 约定）：
- MATLAB 输出的关键点是 1-based 像素坐标（第一个像素的中心是 (1,1)），进入统一 RANSAC 前要减 1。
- `.p` 文件与平台无关，Linux 版 MATLAB 可以运行，但内容看不到；`.mexw64` 和 `.exe` 只能在 Windows 上运行。
- 以下 demo 普遍用到 Computer Vision Toolbox（`matchFeatures`、`detectFASTFeatures`、`showMatchedFeatures`）和 Image Processing Toolbox（`rgb2gray`、`im2uint8`、`imfilter`、`imadjust`）。

---

## 1. RIFT2 — 推荐

- **出处**：Li, Xu, Hu, Zhang, *RIFT2: Speeding-up RIFT with A New Rotation-Invariance Technique*（README 第 7 行，未写明期刊）。
- **代码**：`LJY-RS/RIFT2-multimodal-matching-rotation@0e980ce`，全部为 `.m` 源码，**没有 LICENSE**。仓库自带 `sar-optical/pair1,2` 样例。
- **输入与流程（读到）**：
  - `im2uint8(imread)`；灰度图复制成 3 通道（`demo_RIFT2.m:6-15`）。
  - `FeatureDetection` 里转回 `rgb2gray`，然后调用 `phasecong3(im,4,6,3,'mult',1.6,'sigmaOnf',0.75,'g',3,'k',1)`，对 PC 最大矩做 min-max 归一化，再用 FAST（`MinContrast`、`MinQuality` 都是 1e-4）取最强的 5000 个点（`FeatureDetection.m:3-12`）。
  - 描述子参数：patch 96，6×6（`demo_RIFT2.m:23-28`）。
  - 匹配：`matchFeatures(...,'MaxRatio',1,'MatchThreshold',100)`，即纯最近邻（第 31 行）。
  - 然后用 FSC（similarity，3 px）去外点（第 40 行）。**统一评测时应该取 FSC 之前的 `matchedPoints1/2`**（推断）。
- **对我们的含义**：相位一致性对局部对比度不敏感，dB→uint8 的映射方式影响小。uint8 量化会损失一些 dB 动态范围（推断）。

## 2. RIFT（TIP 2020）

- `LJY-RS/RIFT-multimodal-image-matching@7ea830e`，`license.txt` 是 BSD-3 式条款（第 1-27 行，条款里的大学名称明显是从模板照抄的）。
- 流程同上，但 demo 版本不做旋转不变（`RIFT_demo.m:1-2,30`），去外点用 FSC（affine，2 px，第 42 行）。README 列出的适用场景包括 SAR–光学。
- 在 SRIF 仓库里有带旋转不变的 RIFT，且部分文件是 `.p`（`algorithms/RIFT` 下有 6 个 `.m`、2 个 `.p`）。

## 3. SRIF（ISPRS 2023）— 推荐，但只有 Windows 版

- **出处**：Li, Hu, Zhang, *Multimodal image matching: A scale-invariant algorithm and an open dataset*，ISPRS JPRS 204 (2023) 77-88（README 第 8 行）。
- **代码**：`LJY-RS/SRIF@88881a3`。SRIF 本身**只提供 `SRIF.exe` + `opencv_world345.dll`**（`algorithms/SRIF/`）。同一仓库还打包了其他方法：
  - LNIFT（exe）、3MRS（exe + `opencv_world320.dll`）
  - MS-HLMO（23 个 `.p`）、CoFSM（9 个 `.m` + 11 个 `.p`）
  - OS-SIFT（13 个 `.m` + 1 个 `.p`）、RIFT
  - SIFT（VLFeat）
  - 以及 Optical-SAR 等 6 类数据集（`dataset/Optical-SAR`）
- **调用方式（读到，`demo_SRIF.m`）**：
  - `uint8(imread)` → `imwrite` 成 PNG（第 17-21 行）。
  - `SRIF.exe 1.png 2.png 128 4 8 5000 1 1 matches.txt`（第 24 行，参数含义没有文档）。
  - 读出 `matches.txt` 的 4 列（第 29-32 行）。
  - demo 用 GT 仿射矩阵、3 px 统计 RMSE 和正确匹配数（第 34-54 行）。
- **坐标约定**：看不到。需要用已知平移的合成对标定（推断）。

## 4. LNIFT（TGRS 2022）

- 官方只有二进制：`LJY-RS/LNIFT_exe@6349a8c`，以及 SRIF 仓库里的 `algorithms/LNIFT/LNIFT.exe`。
- 调用：`LNIFT.exe 1.png 2.png 1 time.txt matches.txt 128 4`（`demo_LNIFT.m:24`）。
- 第三方有一个 Python 复现 `arunsahu159/LNIFT-...`（搜索结果读到，**非官方**，没有核对）。

## 5. 3MRS（2022）

- 只有 `3MRSMatcher.exe`（SRIF 仓库）。调用：`3MRSMatcher.exe 1.png 2.png matches.txt`，输出有表头，用 `importdata(...).data` 读取（`demo_3MRS.m:24,31-34`）。

## 6. OS-SIFT — 推荐（光学–SAR 专用、SAR-aware）

- **出处**：Xiang, Wang, You, TGRS 56(6):3078-3090, 2018。代码 `xym2009/OS-SIFT@631a806`，v1.5（2023-10-27），**没有 LICENSE**。
  - v1.5 不用再手调 Harris 阈值，改为取最强的 5000 个点；并加入了 CSC_match 的源码 `CSC2.m`（README 第 4 行）。
  - v1.4 支持有黑边的影像：用 Mask 去掉边界关键点（README 第 6 行）。
- **输入与流程（读到，`os_sift.m`）**：
  - 注释写「需要灰度图」（第 7 行）。**图 1 = 光学，图 2 = SAR**（第 8-11 行）。
  - `imadjust(im2double(imread))` 对两张图都做 1% 两端饱和拉伸，再各加 0.001「防止分母为 0」（第 14-17 行）。
  - 光学走 `build_scale_opt`，SAR 走 `build_scale_sar`（第 34-35 行）。
  - SAR 梯度 = ROEWA 比值：`Gx = log(M14./M23)`，`Gy = log(M34./M12)`，其中 M 是指数加权的半窗均值（`build_scale_sar.m:20-37`）。
  - **`image == 0.001`（原值为 0）的像素被当作无效区**，膨胀 5×5 后置零（`build_scale_sar.m:7`）。
  - 参数：σ = 2，ratio = 2^(1/3)，8 层，GLOH-like 描述子；匹配和去外点用 `CSC2`，模型为 affine（第 19-26、61 行）。
  - SRIF 仓库里的 demo 版本（`demo_OSSIFT.m`）做 `rgb2gray` + `im2double` + 0.001，**没有 imadjust**；匹配改用 `matchFeatures` 最近邻（第 20-30、63 行）。
- **对我们的含义（推断）**：比值梯度对全局乘性缩放不变，前提是输入是非负的线性量。所以 SAR 侧应该喂**线性 S1**（缩放到 [0,1]），或者幅度 √S1。喂 dB 会破坏乘性斑点模型；而 −25 dB 截断后映射成 0 的像素会被 mask 掉。建议主表跑官方 demo 路径，另做一个「线性 vs dB」的小消融。
- **环境**：`calc_descriptors_parallel` 用了 parfor（没有 Parallel Computing Toolbox 时会串行执行）。`CSC_match.p` 是 p-code，但 `CSC2.m` 是源码。

## 7. SAR-SIFT（Dellinger et al., TGRS 2015）

- **没有官方代码**。第三方实现：`ZeLianWen/Image-Registration`（MATLAB，包括 SIFT、SAR-SIFT、PSO-SIFT）、`yishiliuhuasheng/sar_sift`（Python 2.7），都来自搜索结果，没有深入核对。
- 原本是为 SAR–SAR 设计的；ROEWA 比值梯度的输入要求同 OS-SIFT（推断）。不推荐进主表。

## 8. HAPCG（武汉大学学报 2021）

- `yyxgiser/HAPCG-Multimodal-matching@79aa1ef`。核心函数 `HAPCG_Gradient_Feature`、`HAPCG_Logpolar_descriptors`、`Harris_extreme` 都是 `.p`；非线性尺度空间 `HAPCG_nonelinear_space.m` 是源码。
- 输入：`rgb2gray` + `im2double`（`HAPCG_nonelinear_space.m:43,48`）。
- 参数：`Max = 3`，`scale_value = 2`，`K_weight = 3`，`Path_Block = 42`（`HAPCG_demo.m:20-24`）。匹配 `MatchThreshold = 10`，FSC 为 affine、3 px（第 54、59 行）。
- demo 用 `uigetfile` 交互式选图，需要改成批处理。

## 9. MS-HLMO / MS-HLMOv2

- MS-HLMO（TGRS 2022）原版：`MrPingQi/MS-HLMO_registration` **已经不存在**（GitHub API 返回空）。SRIF 仓库里有 23 个 `.p` 的版本（`demo_MSHLMO.m`：`Preproscessing`，G_resize = 2，σ = 1.6，3 个 octave，N = 5000，patch 96）。第三方复现：`dmwu1115/MS-HLMO`。
- MS-HLMOv2（IGARSS 2024，IEEE 10641671）：`MrPingQi/MS-HLMO_registration-v2.0@c69b13a`。
  - 核心 `.m` 是源码，`Readimage`、`Preproscessing`、`Deal_Extreme` 等 I/O 是 `.p`。
  - 关键点类型 `PC-ShiTomasi`，最多 5000 个；patch 72；模型 affine；Error = 5（`A_main_HLMOv2.m:14-44`）。
  - 作者在 README 里说它已经过时，推荐看 HOMO-Feature。

## 10. HOMO-Feature（ICCV 2025）/ HIMO（TPAMI 2026）— 推荐（二选一，优先 HIMO）

- HOMO：Gao et al., ICCV 2025（CVF open access）。代码 `MrPingQi/HOMO-Feature_ImgMatching@5f64b35`。
- HIMO：Gao, Li, Weng, Tao, Xia, Du, TPAMI 48(8):9001-9018, 2026（PubMed / IEEE 11435911）。代码 `MrPingQi/HIMO_ImgMatching@884297a`。HOMO 的 README 说它已经过时，推荐 HIMO。
- 结构：两者的算法核心 `func_HOMO/`、`func_HIMO/`、`func_Math/` 都是 `.m` 源码；读图、预处理、`Deal_Extreme`、显示、变换等在 `functions/*.p` 里（文件树读到）。
- **输入约定无法完全核实**：`Readimage(file)` → `Deal_Extreme(image,64,512,0)` 求 resample 比例（推断：把尺寸限制在 64–512，对我们的 512² 图大概率不缩放）→ `Preproscessing`（p-code）。
- 参数（`A_HOMO_demo.m:14-41`）：`int_flag = 1`，`rot_flag = 1`，`scl_flag = 0`；3 个 octave，每 octave 2 层；G_resize = 1.2；`PC-ShiTomasi`，最多 5000 个点；patch 72；affine；Error = 5。
- 光学–SAR 数值：论文 PDF 加密，**这次没能读出**。

## 11. sRIFD（Infrared Physics & Technology 2023）

- `bohanlee/sRIFD@a989da4`，MATLAB 源码（自带 `phasecong3.m`、`detectFASTFeatures.m`），基于 RIFT2，**没有 LICENSE**。样例是 LC09 可见光–热红外。

## 12. CoFSM

- 作者仓库 `yyxgiser/CoFSM`，README 写「executable program」。SRIF 仓库里的 CoFSM 部分文件是 `.p`；`demo_CoFSM.m` 做 `rgb2gray`，不做 double 转换（第 20-31 行）。

## 13. 面匹配 / 模板匹配类：CFOG、HOPC、FED-HOPC

- **CFOG**（Ye et al., TGRS 2019）：`yeyuanxin110/CFOG@bdae6ad`。
  - README 写「代码只能在 64 位 Windows 上运行」，已申请专利，只限科研使用。
  - `matchFramework(im_Ref, im_Sen, CP_initial_file, 'CFOG')` 需要**初始控制点文件**来确定搜索区（第 5、122-136 行）。模板大小 100，搜索半径 20（第 64-67 行）。
  - 流程：`rgb2gray` → `double`（未除以 255）（第 72-82 行）→ 分块 Harris 取点 → 基于 FFT 的 SSD 模板匹配（`fftmatch.p`）。
- **HOPC**（Ye et al., TGRS 2017）：`yeyuanxin110/HOPC@bb0aa72`，同一套框架，依赖 `denseBlockHOPC.mexw64`。第三方 Python 版：`yyxgiser/HOPC-Optical-to-SAR-registration`（fork）。
- **FED-HOPC**（2023）：`yyb1234-56/FED-HOPC@c5b510a`（`yyxgiser` 那份是 fork）。依赖 HOPC 核心的 mexw64 + 若干 `.p`。
- **对我们（推断）**：这类方法只在给定初始仿射附近搜平移，属于「精配准」工具，不适合直接和给出全局匹配的方法同表比较。除非统一用单位阵作为初值，并且仿射扰动小于搜索半径。建议不进主表。

## 14. POS-GIFT（Information Fusion 102, 2023）

- 在 GitHub 和作者主页都**没有找到公开代码**（检索了 ScienceDirect / ACM 条目）。排除。

## 15. 其他

- **VSFF**（`yeyuanxin110/VSFF`）：光学–SAR **融合**方法，不是配准，排除。
- **BIFT**（`yyxgiser/BIFT`，2025-11）：目前只有数据集，没有代码。
