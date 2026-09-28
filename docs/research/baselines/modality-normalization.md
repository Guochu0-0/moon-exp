# 跨模态匹配器如何归一化不同传感器的输入（面向 SAR 零样本喂预训练匹配器）

调研日期 2026-09-26。只读调研，未提交。「推断」= 我的判断，非出处原话。

我们的情况：SAR = Mini-RF S1 总功率 → dB → 下限 −25 dB → 用 SAR 自己 Train 的均值/标准差全局标准化得 z（moon-dataset #38 决定，ADR 0001）；光学 uint8，全局标准化 z = (灰度 − 73.786)/26.533。要把这两者喂给**按 gray/255 或 RGB 训练的预训练匹配器**做零样本评测。

候选：
- (a) 用光学全局统计把 SAR z 映回光学灰度尺度：g = (μ_opt + σ_opt·z)/255
- (b) 用自然图像 / ImageNet 统计映射
- (c) 逐图归一化（min-max / 分位拉伸 / 逐图 z-score），两模态各自独立
- (d) 每模态数据集级 mean/std 归一化后直接喂
- (e) 用数据集专用管线把 SAR 转成 8 位（dB + 分位截断到 [0,255]），然后当普通图像

## 0. 内部已有结论（复用，不重做）

- moon-dataset #35 调研（分支 `research/sar-input-preprocessing`，`research/notes/sar-input-preprocessing.md`）：
  - SAR 深度学习数据集主流是 dB + **全局固定常数**（SEN12MS clip [−25,0] dB；SSL4EO-S12 全局 μ±2σ→uint8；torchgeo 全局 z-score）。
  - **光学–SAR 配准数据集把 SAR 做成逐图拉伸的 8 位灰度**：OSdataset 线性振幅 + MATLAB `imadjust`（上下各饱和 1%）；SEN1-2 dB 后截到 ±2.5σ 缩放到 [0,1]（σ 是逐景还是全局未写明）。RIFT/HOPC/CFOG 公开代码都吃 8 位 PNG/TIF；CFOG 丢弃被匹配图上 ≤0 的像素。
  - 预训练权重：LoFTR loader `/255`；RoMa ImageNet mean/std；RIPE 灰度复制 3 通道。
- moon-dataset #38 决定：S1 dB、下限 −25、SAR 全局 z = (S1_dB + 10.976)/2.711；光学全局 z；给传统方法的 8 位版本、多通道留在「雾区」。
- 本仓库 `research-baselines` 工作树 `docs/research/baselines/interfaces-a.md`、`interfaces-b.md`（逐行读过官方代码）：
  - LoFTR、SP+SG、MA-ELoFTR、MINIMA-LoFTR/SP+LG：loader 里只做 `÷255`，forward 无归一化、无 clamp → 对灰度线性尺度**敏感**（interfaces-a §LoFTR `loftr.py:39-75`；§SP `superpoint.py:161-171` 绝对阈值 0.005）。
  - **XoFTR forward 内做逐图 z-score**（`src/xoftr/xoftr.py:39-47`），对每图 a·x+b 不变 → 线性映射的选择对它无影响，非线性（dB vs 线性）仍有影响。
  - MA-RoMa：`convert("RGB")/255`，`NORMALIZE_IMG=False`，**不做 ImageNet 归一化**（interfaces-b §2，`common_data_pair.py:211-212`）；MINIMA-RoMa 则做 ImageNet mean/std（`matcher.py:651-654`）。
  - MatchAnything 官方 visible–SAR 评测：`read_megadepth_gray` 灰度读图 → 拉伸 resize 到 832 → `÷255`（`MA/src/utils/dataset.py:207-265`，`evaluate_datasets.py:117-118,137`）。
  - interfaces-a 的建议：runner 统一给每个方法 **uint8 单通道**（SAR 用固定 dB→uint8 映射），映射方式作为消融项。

## 1. 最直接的证据：第三方零样本 SAR–光学基准（rsim）

Corley, Stoken, Berton, *Are Pretrained Image Matchers Good Enough for SAR–Optical Satellite Registration?*，[arXiv:2604.10217](https://arxiv.org/html/2604.10217v4)，代码 [isaaccorley/rsim](https://github.com/isaaccorley/rsim)。24 个预训练匹配器零样本跑 SpaceNet9、SRIF、SARptical。

代码（`src/rsim/spacenet9_matcher_benchmark.py`，HEAD 读取于 2026-09-26）：
- 读图 `read_image`（:357-364）：rasterio 读出后，**非 uint8 的影像先逐图 min–max 到 uint8**（`_normalize_to_uint8`，:345-354），单通道复制成 3 通道。不做 dB。
- 然后对**每个模态、每张图各自**做一种 normalization（`normalize_image(image, method, modality)`，:367-497；`to_matcher_tensor` :592-594），输出都在 [0,1] 再交给匹配器：
  - `identity`：`clip(x/255, 0, 1)`（:383-384）
  - `percentile`：逐图、逐通道 p2–p98 min–max 截断到 [0,1]（:374-392）
  - `zscore`：逐图、逐通道 z-score，**截到 ±2.5σ 再线性映到 [0,1]**（:476-488）——即逐图 z-score 之后仍映回 [0,1] 值域
  - `clahe`：clipLimit 2.0、8×8（:490-495）
  - 另有 `sarlog_percentile`（仅 SAR 侧 `log1p` 后 p2–p98，:394-401）和几种仿 SpaceNet9 获奖方案的 `winsol_*`（灰度 + 双边滤波 + CLAHE/直方图均衡 + 模糊，:403-474）
- 论文 §3.2：“Each image is optionally normalized (identity, percentile clipping to [2,98], z-score, or CLAHE with clip limit 2.0)”。

结果（论文）：
- Table 2（SpaceNet9）每个匹配器**各自在 3 个带标注训练场景上选最优归一化**，选中结果**因方法而异**：RoMa、RoMaV2、MA-ELoFTR、DISK-LG、SuperPoint-LG、GIM-DKM → Z-Score；XoFTR、MINIMA-XoFTR、MINIMA-RoMa、RoMa+LoFTR、ALIKED-LG、DeDoDe-LG → Percentile；LoFTR、MASt3R → Identity。最好的 RoMa、XoFTR 均 3.0 px，MA-ELoFTR 3.4 px，MINIMA-RoMa 3.4 px。
- Table 3 / Fig. 5（SRIF）归一化 × 匹配器：LoFTR 59.3（Identity）/ 63.1（Z）/ 63.3（CLAHE）px；MINIMA-RoMa 48.2 / 47.0 / 47.7 px。原文：“normalization sensitivity varies substantially across architectures”。
- 摘要：协议选择（几何模型、tile 大小、inlier 门限、归一化）可使同一匹配器平均误差变化高达 33×；归一化单独的贡献论文未单列（WebFetch 读全文未找到逐归一化的 SpaceNet9 数字）。
- 注意：这里所有候选都是**逐图**（c 类或 identity），没有测试「用对方模态/数据集全局统计映射」的 (a)(b)(d)。

## 2. 通用跨模态匹配器（数据引擎类）

### 2.1 MINIMA（LSXI7/MINIMA，main 分支，gh 读取）
- 训练数据 MegaDepth-Syn：从 RGB **生成** 6 种模态，全部存成 8 位图像文件，然后与普通图像一样读入：
  - depth：Depth-Anything-V2 输出**逐图 min–max 到 0–255** → uint8 → `Spectral_r` 伪彩色 3 通道（`data_engine/tools/depth/depth_transfer.py:28-33`）。
  - event：灰度 uint8 取 `log1p` 后按像素差阈值画成红/蓝/白 3 通道图（`data_engine/tools/event/event_transfer.py:15-32`）。
  - infrared：扩散模型（scepter/StyleBooth）输出，PIL 保存（`modality_engine.py:66-67`，`*255 → uint8`）。
- 评测：
  - DIODE 深度：读 `outdoor_depth/` 下已渲染的图像文件（`test_relative_homo_depth.py:122`），`cv2.imread(IMREAD_COLOR 或 GRAYSCALE)` → `/255.`（:342-358）。深度的渲染方式未在该脚本里；推断与数据引擎相同（逐图 min–max + 伪彩色），但下载版未核实。
  - MMIM（含 `RemoteSensing/SAR_Optical`）：直接读 StaRainJ/Multi-modality-image-matching-database 发布的 8 位 PNG（`test_relative_homo_mmim.py:138-146`；README:170-183）。
  - 各匹配器 loader：LoFTR/XoFTR/SP-LG 灰度 `/255`；MINIMA-RoMa 额外 ImageNet mean/std（interfaces-b §4-6，`matcher.py:651-654`）。**推理时没有任何模态特定处理**，两侧同一条管线。

### 2.2 MatchAnything（zju3dv；代码在 HF Space `LittleFrog/MatchAnything`）
- 训练数据（论文 [arXiv:2501.07556](https://arxiv.org/abs/2501.07556) §4.3）：CycleGAN 把可见光翻译成热红外/夜景；深度估计结果 “**rescaled to grayscale images**, which then replace the original images”。训练代码未发布（README “[ ] Data generation and training code”）。
- 评测 visible–SAR（§4.6.1，“Visible-SAR [82] … SAR and visible light image pairs captured from satellite views”）：测试数据是作者预先打包的图像文件（`data/test_data/visible_sar_dataset/eval`，`scripts/evaluate/eval_visible_sar.sh:9-10`），`read_megadepth_gray` 灰度读 → 拉伸到 832 → `÷255`（interfaces-b §1，`MA/src/utils/dataset.py:207-265`）。论文没给 SAR 的 8 位转换细节（WebFetch 全文未找到）。
- 推理：ELoFTR 分支 `÷255` 无 mean/std；RoMa 分支 `convert("RGB")/255`、`NORMALIZE_IMG=False`（不做 ImageNet 归一化）。无模态特定处理。

### 2.3 AnyMatch（[arXiv:2606.31077](https://arxiv.org/abs/2606.31077)，ECCV 2026）
- §3.2 从单视图 RGB 合成 infrared（LDM+LoRA）、depth、normal、event；§4.1 评测 METU-VisTIR、DIODE、DSEC、MMIM，“uniformly resized to 512×512”。论文未写 8 位转换或输入归一化；未找到代码。**无 SAR 专门处理。**

### 2.4 XoFTR（OnderT/XoFTR）
- 推理：forward 内**逐图 z-score**（`src/xoftr/xoftr.py:41-47`）；loader 仍 `/255`（`src/utils/dataset.py:122,227`）。
- 预训练（KAIST 可见–热红外，MIM）：`read_pretrain_gray` 同样 `/255` 后逐图 z-score（`src/utils/dataset.py:258-262`）。
- 微调数据增强 “pseudo-thermal”（论文 [arXiv:2404.09692](https://arxiv.org/abs/2404.09692) §3.5；`src/utils/augment.py:58-82`）：灰度 `/255 − 0.5` → `cos(w·x + phase)`（随机频率/相位，**非单调**灰度映射）→ **逐图 min–max 到 0–255**。即训练时就让网络见过对比度反转/非单调映射，并用逐图 z-score 消掉全局增益与偏移。
- METU-VisTIR 热红外以 8 位 JPG 发布（仓库 `assets/METU_VisTIR_samples/*/thermal/images/*.jpg`）；原始位深与 8 位转换方式论文未写。

## 3. 光学–SAR 专用方法与数据集

### 3.1 方法代码
- **ReDFeat**（ACuOoOoO/ReDFeat，TIP 2022）：训练 `lib/dataset.py:68-69`、推理 `match.py:141,161` 都是 `TF.to_tensor(PIL RGB)` 后**逐图 z-score**（per-channel，H×W 上的 mean/std）。VIS-SAR 数据来自 OSdataset 的 8 位 PNG（Multimodal_Feature_Evaluation README 表：VIS-SAR 1\1 通道，512×512，引用 Xiang et al. 2020）。
- **OSMNet**（zhanghan9718/OSMNet，TGRS）：OSdataset 8 位 `cv2.imread(...,0)`（`folder2lmdb_osdataset.py:39-60`）；`ToTensor` 后用**一对共享的训练集全局常数** `Normalize(0.4309, 0.2236)`（`OSMNet_train_test.py:72-75,169-170`），SAR 与光学用同一组，而非每模态各一组。
- **SOMatch**（system123/SOMatch，Hughes et al.）：`img_as_float(imread(as_gray))` → `rescale_intensity(in_range='dtype')` 到 [0,1]（`datasets/sen12_dataset.py:127-133`）；代码**支持每模态各自 Normalize(mean,std)**（:64-72，即 (d)），但发布的实验配置 `"normalize": {}`（`config/so_match/experiments/backbone_asl1_wml.json:60,78,105`）→ 实际不用。
- **MIFNet**（lyp-deeplearning/MIFNet，TIP 2025，零样本到 Opt-SAR）：示例 `cv2.imread` 8 位 PNG（`scripts/test_xfeat_mifnet.py:28-29,125-127`），无 SAR 特殊处理。
- **SpaceNet9 获奖方案**（SpaceNetChallenge/SpaceNet9，全部用预训练匹配器）：
  - 第 2 名：MINIMA-LoFTR、MINIMA-XoFTR、ALIKED/DISK/SIFT-LG 集成，SAR 与光学都只是 `ToTensor()`（`2nd_motokimura/code/registration/registration_pipeline.py:383-386`；`registration_config.py:12-20`），无额外归一化。
  - 第 3 名：SAR 数组 `astype(float32)/255`，并用 CLAHE 做 TTA（`3rd_TheRealRoman/code/inference.py:51-66,117-120`）。
  - 第 4 名：RoMa，SAR 侧只加高斯模糊（`4th_mawanda-jun/code/matcher_roma.py:24-30`）。
  - 第 1 名：SuperPoint+LightGlue，对光学做 3×3 中值滤波（`1st_handreak80/code/test.py:96-99`）。
  - 推断：这些代码对 SAR 直接 `/255` 或 `ToTensor`，说明 **SpaceNet9 的 Umbra SAR 已以 8 位发布**（官方数据说明未读到，DLR 的 IGARSS PDF 无法解析）。
- 其他（FHReg/GUSO、CMM-Net、3MRS、SOPatch 描述子网络）：本次未找到可读代码，未核实。

### 3.2 数据集（SAR 以什么形式发布）
- **SEN1-2**：VV σ⁰ dB，截到 ±2.5σ 缩放到 [0,1] 后以 8 位 PNG 发布（#35 调研引 arXiv:1807.01569；σ 是逐景还是全局未写明）。
- **OSdataset**：GF-3 线性振幅，MATLAB `imadjust` 逐图饱和上下 1% → 8 位（#35 引 xym2009/OSdataset README）。
- **3MOS**（[arXiv:2404.00838](https://arxiv.org/html/2404.00838) §3.1）：“we first use SRTM DEM for terrain correction … Then, we apply grayscale stretching to enhance the images and **quantize them to 8 bits**”（拉伸算法未写）。
- **SpaceNet6**（SAR-Intensity）：4 极化强度，2×2 multilook，负值置 0，`10·log10`，以 dB 浮点 GeoTIFF 发布（[arXiv:2004.06500](https://arxiv.org/abs/2004.06500)）；官方基线直接对 dB 值套 albumentations `Normalize(mean=0.5, std=0.125, max_pixel_value=255)`（[CosmiQ_SN6_Baseline](https://github.com/CosmiQ/CosmiQ_SN6_Baseline) `baseline.py:293-298`）→ 固定全局常数，属 (d) 的变体。
- **SpaceNet9**：见上，8 位（推断）。
- **MMIM**（StaRainJ）SAR_Optical 子集、**MatchAnything visible-SAR**：以 8 位 PNG/图像文件发布，转换方式未写。
- **QXS-SAROPT、WHU-OPT-SAR、SOPatch**：未找到对 SAR 位深/拉伸的一手说明（QXS 的 arXiv PDF 加密无法读取；WHU-OPT-SAR README 未写）。
- 共同点：**配准/匹配类数据集几乎都在发布前就把 SAR 做成 8 位灰度**（拉伸方式各异，多为逐图），下游匹配器把它当普通图像 `/255`。SAR 深度学习数据集（SEN12MS、SSL4EO、SpaceNet6）则发布 dB 浮点，用全局常数归一化（#35）。

## 4. 总表

| 方法 / 数据集 | 非光学模态怎么变成网络输入（训练） | 推理 / 评测 | 对应候选 | 出处 |
|---|---|---|---|---|
| LoFTR、SP+SG/LG、ELoFTR | 自然灰度 uint8 `/255` | 同上；forward 无归一化，对线性尺度敏感 | 需上游 (e) | interfaces-a `loftr.py:39-75`，`superpoint.py:161-171` |
| RoMa（原版、MINIMA-RoMa） | RGB `/255` + ImageNet mean/std | 同上 | 需上游 (e)，(b) 由框架自己做 | #35；interfaces-b `matcher.py:651-654` |
| MA-RoMa | RGB `/255`，不做 ImageNet 归一化 | 同上 | 需上游 (e) | interfaces-b `common_data_pair.py:211-212` |
| MINIMA（数据引擎） | depth 逐图 min–max→uint8→伪彩；event log1p 差分画图；IR 扩散生成 8 位 | 真实 depth/SAR/IR 均读 8 位图 `/255`，无模态特定处理 | (c)+(e) 于数据侧 | `depth_transfer.py:28-33`；`event_transfer.py:15-32`；`test_relative_homo_*.py` |
| MatchAnything | CycleGAN 生成 thermal/night；深度 “rescaled to grayscale” | visible-SAR 测试集为预制 8 位图，灰度 `/255` | (e) | arXiv:2501.07556 §4.3、§4.6.1；`eval_visible_sar.sh:9-17` |
| AnyMatch | 合成 IR/depth/normal/event，格式未写 | 评测 METU-VisTIR/DIODE/DSEC/MMIM，未写 | — | arXiv:2606.31077 §3.2、§4.1 |
| XoFTR | 8 位 `/255` → **forward 内逐图 z-score**；伪热红外增强 = cos 非单调映射 + 逐图 min–max | 同训练；METU-VisTIR 热红外 8 位 JPG | (c)（网络内） | `xoftr.py:41-47`；`augment.py:58-82`；`dataset.py:258-262` |
| ReDFeat | OSdataset 8 位 → **逐图 z-score** | 同上 | (e)+(c) | `lib/dataset.py:68-69`；`match.py:141,161` |
| OSMNet | OSdataset 8 位 → **共享**训练集全局 Normalize(0.4309, 0.2236) | 同上 | (e)+单一全局常数 | `OSMNet_train_test.py:72-75,169-170` |
| SOMatch | 8 位 → [0,1]；可配每模态 mean/std 但默认关 | 同上 | (e)（(d) 可选未用） | `sen12_dataset.py:64-72,127-133` |
| MIFNet | 单模态训练 | Opt-SAR 8 位 PNG | (e) | `test_xfeat_mifnet.py:28-29` |
| SpaceNet9 获奖方案（预训练匹配器） | 不训练 | SAR 已为 8 位（推断），`/255` 或 `ToTensor`；附加 CLAHE TTA / 高斯模糊 / 中值滤波 | (e)（+局部增强） | SpaceNet9 仓库各队代码，见 §3.1 |
| rsim 第三方基准（24 个预训练匹配器） | — | 非 uint8 先逐图 min–max 到 uint8；再每模态各自选 identity/p2–p98/逐图 z(±2.5σ→[0,1])/CLAHE，**逐方法选最优** | (e)+(c) | `spacenet9_matcher_benchmark.py:345-497`；arXiv:2604.10217 §3.2、Table 2-3 |
| SEN1-2 | dB → ±2.5σ → [0,1] → 8 位 PNG | — | (e) | #35（arXiv:1807.01569） |
| OSdataset | 线性振幅 → `imadjust`（逐图 1%/99%）→ 8 位 | — | (e)+(c) | #35（OSdataset README） |
| 3MOS | 地形校正 → 灰度拉伸 → 8 位 | — | (e) | arXiv:2404.00838 §3.1 |
| SpaceNet6 | dB 浮点发布 | 基线：dB 直接 Normalize(0.5, 0.125, max 255) | (d) 变体（固定常数） | arXiv:2004.06500；`baseline.py:293-298` |
| SEN12MS / SSL4EO-S12 / torchgeo | dB → 固定 [−25,0] 或全局 μ±2σ→uint8 或全局 z | 同 | (d)/(e) | #35 |

## 5. 综合结论

**主流做法**
1. 把预训练匹配器零样本用在 SAR 上时，**主导做法是 (e)**：在匹配器之外先把 SAR 变成一张普通的 8 位灰度图，然后原样走该匹配器自己的 loader（`/255`，RoMa 系再加 ImageNet mean/std）。MINIMA、MatchAnything、SpaceNet9 各获奖方案、MIFNet、ReDFeat/OSMNet 的数据都是这样。**推理阶段没有哪个通用匹配器做 SAR 专用处理。**
2. 8 位化这一步由数据集决定，**多数是逐图拉伸**（OSdataset `imadjust` 1%/99%，3MOS “grayscale stretching”，MINIMA 深度逐图 min–max，rsim 对非 uint8 逐图 min–max）。只有 SEN1-2（±2.5σ，范围不明）和 SSL4EO（全局 μ±2σ）这类可能是全局常数。所以 (e) 在实践中通常等同于 (e)+(c)。
3. **(c) 逐图 z-score 也被部分方法做进了网络里**：XoFTR 在 forward 里做，ReDFeat 在 loader 里做。这类方法对每张图各自的线性映射不敏感，(a)/(c)/(e) 中的线性部分选哪种都一样。仍有影响的是非线性部分（dB 还是线性、截断、CLAHE）（由代码推断）。
4. **(a)、(b)、(d) 在零样本匹配的文献里基本没有先例。** (d) 出现在从头训练的 SAR 网络里（SpaceNet6 基线、SEN12MS、torchgeo；SOMatch 支持但默认关闭；OSMNet 用的是两模态共享的一组常数，不是每模态各一组）。(b) 只是 RoMa 类在 `/255` 之后自己做的一步，不是把 SAR 映射到图像尺度的方法。把 SAR 按光学全局统计映回灰度的 (a)，我没有找到任何出处（WebSearch 未命中）。

**敏感性证据**
- 唯一直接的证据是 rsim（arXiv:2604.10217）。24 个匹配器各自在训练场景上选归一化，**最优选择因方法而异**：RoMa 系和 MA-ELoFTR 选 Z-Score，XoFTR 和 MINIMA-XoFTR/RoMa 选 Percentile，LoFTR 选 Identity。在 SRIF 上，只换归一化带来的差别不大：LoFTR 59.3 到 63.3 px（约 7%），MINIMA-RoMa 47.0 到 48.2 px（约 2.5%）。整套协议（tile、几何模型、门限）的影响可达 33×，归一化只是其中一部分。rsim 没有测试 (a)/(d)。
- 从代码结构推断：LoFTR、SP 系、MA-ELoFTR 没有输入归一化，SuperPoint 还用绝对阈值，所以对灰度尺度最敏感；XoFTR、ReDFeat 对逐图仿射变换不变。

**对我们的含义（推断，供决策参考）**
- (a) 的形式是 g = (73.786 + 26.533·z)/255，本质上是「用一组固定全局常数把 S1_dB 线性映到 8 位」，即 (e) 的全局常数版本。它和 SSL4EO 的 μ±kσ→uint8 同类。但我们的 z 落在 [−5.2, +8.2]，映射后会落到 [0,255] 之外，必须截断，最亮的结构会被截掉。patch 之间的亮度基线差异也会原样保留下来。
- 对 XoFTR 而言，(a)、(c)、(e) 的线性部分等价。对 LoFTR、SP、ELoFTR、RoMa 而言则会有差别，但按 rsim 的量级，这个差别很可能只有几个百分点。
- 与文献最一致的方案：给每个匹配器一份 **SAR 的 8 位视图**，默认用 S1_dB 逐 patch p2–p98（或 p1–p99）拉伸（即 (e)+(c)，和 OSdataset、rsim、MINIMA 的做法同类）。把「全局常数映射 (a)」和「逐图 z-score (±2.5σ→[0,1])」作为消融项。沿用 rsim 的做法，逐方法选归一化，但只在 Train/Val 上选。光学侧保持 uint8 原样（identity）。

