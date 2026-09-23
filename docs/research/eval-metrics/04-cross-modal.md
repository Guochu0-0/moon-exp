# 光学–SAR 配准评价协议调研（四）：跨模态匹配论文组（02_跨模态匹配）

> 目的：弄清已有跨模态匹配工作的评价指标**在源码里是怎么实现的**，重点看把光–SAR 匹配当作评测任务的那些工作。关注点：指标定义、误差算在什么对象上、成功判定、失败样本怎么计入、鲁棒估计器、聚合方式、数据集和真值来源。服务对象是月球 CE-2 / Mini-RF patch 仿射配准的评价协议。
>
> 范围：`D:\科研\光SAR配准\论文\02_跨模态匹配\` 下全部 16 个 PDF。同目录「阅读报告」里的 .md 只用来定位，结论一律以 PDF 和源码为准。
>
> 引用格式：
> - 源码：commit 锁定的 permalink。
> - 论文：「论文 p.X」，X 是 PDF 页码（用 pdftotext 分页符定位）。
> - **代码和论文不一致时，以代码为准，并注明「⚠ 与论文不一致」。**
>
> 所有仓库均 `--depth 1` 克隆到 `D:\Code\refs\<repo>`，commit 见文末「源码清单」。

---

## 0. 筛选表

「光–SAR 评测」一栏：**是** = 论文里有光–SAR（或 SAR–光学）定量结果；**间接** = 光–SAR 只是某个混合基准里的一个子集，论文没有单独报告它。

| # | 论文（文件名） | 光–SAR 评测 | 官方源码 | 处理方式 |
|---|---|---|---|---|
| 1 | RIFT（TIP 2020）`RIFT.pdf` | 是（自建 6 类×10 对，含 SAR-optical 10 对） | https://github.com/LJY-RS/RIFT-multimodal-image-matching （只有 demo） | 精细（§1） |
| 2 | SRIF（ISPRS 2023）`ScaleInvariantMultimodalMatching_ISPRS2023.pdf` | 是（自建 1200 对，其中 Optical-SAR 200 对） | https://github.com/LJY-RS/SRIF | 精细（§2） |
| 3 | GDROS（TGRS 2025）`GDROS_OpticalSAR_2025.pdf` | 是（WHU-OPT-SAR、OS dataset、UBCv2） | https://github.com/Zi-Xuan-Sun/GDROS | 精细（§3） |
| 4 | SOMA（2025）`SOMA_AffineFlow_SAROpt_2025.pdf` | 是（SEN1-2、GFGE SO，外推到 WHU-SEN-City、OSdataset） | https://github.com/traslauc/SOMA | 精细（§4） |
| 5 | Shared Modality（2026）`SharedModality_SAROpt_2026.pdf` | 是（MultiSenGE S1/S2） | https://github.com/BorisovAN/shmod | 精细（§5） |
| 6 | TAR（TGRS 2026）`TAR_TextAssisted_SAROpt_2026.pdf` | 是（SEN1-2、OSdataset） | **未找到**（论文和 arXiv 页都没有链接，搜索也没找到） | 精细，只有论文依据（§6） |
| 7 | RRSI（2026）`RRSIFD_MultimodalDescriptor_2026.pdf` | 是（OSdataset；另有 LLVIP、DroneVehicle） | https://github.com/yeyuanxin110/RRSI （**仓库为空，没有任何 commit**） | 精细，只有论文依据（§7） |
| 8 | MatchAnything（TPAMI 2026）`MatchAnything_TPAMI2026.pdf` | 是（Visible-SAR，1209 对） | GitHub 上只有 README；评测代码在 HF Space https://huggingface.co/spaces/LittleFrog/MatchAnything | 精细（§8） |
| 9 | MINIMA（CVPR 2025）`MINIMA_CVPR2025.pdf` | 间接（MMIM Remote Sensing 共 7 类，SAR_Optical 只有 6 对） | https://github.com/LSXI7/MINIMA | 精细（§9） |
| 10 | AnyMatch（ECCV 2026）`AnyMatch_ECCV2026.pdf` | 间接（沿用 MMIM Remote Sensing） | **未找到**（arXiv 2606.31077 页面没有代码链接） | 精细，只有论文依据（§10） |
| 11 | HOMO-Feature（ICCV 2025）`HOMO-Feature_ICCV2025.pdf` | 是（自建 GCZ，含 VIS-SAR 组） | https://github.com/MrPingQi/HOMO_Feature_ImgMatching （只有 demo，没有评测代码） | 精细（§11） |
| 12 | XoFTR（CVPRW 2024）`XoFTR_CVPRW2024.pdf` | 否 | https://github.com/OnderT/XoFTR | 一行：METU-VisTIR 相对位姿 AUC@5/10/20°；LGHD、FusionDN 用单应角点误差 AUC（论文 Tab.2–4） |
| 13 | LoRetta（arXiv 2026）`LoRetta_2026.pdf` | 否（LEVIR-GM 为光学–光学；表中列出的 SAR 数据集只是综述） | 只有项目页，未找代码 | 一行：所有方法统一走 RANSAC → TPS 拟合 → 稠密 warp，只在可匹配像素上算 PCK@1/2/3/5/10px，AUC 取 T={0.5,1,…,2.5,3,…,10} 上的归一化梯形面积（论文 p.10–11） |
| 14 | MatchAnyEvents（ECCV 2026）`MatchAnyEvents_ECCV2026.pdf` | 否（事件相机） | https://github.com/spikelab-jhu/Match-Any-Events （未克隆） | 一行：事件–事件、事件–图像匹配的位姿/单应 AUC，不涉及遥感 |
| 15 | MatchGS 零样本半稠密（2025）`ZeroShotSemiDense_GaussianSplat_2025.pdf` | 否 | 未查 | 一行：MegaDepth/ScanNet 位姿 AUC@5/10/20°，ZEB 零样本位姿 AUC，HPatches 角点误差 AUC@3/5/10px（论文 p.12 附录） |
| 16 | 综述 `CrossViewMatching_Survey2026.pdf` | 不适用 | 不适用 | 只扫指标：相对位姿用 max(R,t) 角误差 AUC@5/10/20；单应用**平均**角点重投影误差 AUC@3/5/10px；另有视觉定位的阈值召回（论文 p.12） |

---

## 1. RIFT（Li et al., TIP 2020）

**源码**：`LJY-RS/RIFT-multimodal-image-matching` @ `7ea830e`。仓库只有单对图像的 demo（`RIFT_demo.m`），**没有真值文件，也没有评测脚本**。批量评测 RIFT 的代码见 §2 的 SRIF 仓库 `demo_RIFT.m`。

- **数据**：6 类，每类 10 对，共 60 对，其中 SAR-optical 10 对（论文 p.8）。
- **真值来源**：每对人工选 **5 个**分布均匀的亚像素对应点，用它们拟合**仿射**变换，作为「近似真值」（论文 p.8）。
- **指标**：NCM、RMSE、ME（平均误差）、SR（论文 p.8）。
  - 正确匹配：用 NBCS 去除外点后，在真值仿射下残差 **< 3 px** 的点对。
  - **NCM < 4 即判为失败**。
  - 失败样本怎样计入 ME/RMSE，论文**没有明说**。p.12 只写了：SIFT、SAR-SIFT 的 SR 太低，所以「不计算」它们的 ME/RMSE，也就是这两种方法整体不报这两个指标。RIFT 自己 60 对全部成功，所以这个问题在 RIFT 上没有暴露。
  - 全部 60 对的平均：NCM 122.4，ME 1.79 px，RMSE 1.94 px（论文 p.12）。
- **估计器**：评测时用 NBCS 去外点（论文 p.8）。demo 里用的是 FSC、affine、阈值 2 px，然后以估计出的 H 残差 < 3 px 作为 inlier（[RIFT_demo.m#L42-L49](https://github.com/LJY-RS/RIFT-multimodal-image-matching/blob/7ea830e2f13cc3c226f975fe9e98b7666a8f26fb/RIFT_demo.m#L42-L49)）。**demo 的 inlier 是相对估计出的 H 判定的，不是相对真值。**
- **聚合**：按类别对样本取平均（论文 Table 6，p.12）。

---

## 2. SRIF（Li, Hu, Zhang, ISPRS 2023）

**源码**：`LJY-RS/SRIF` @ `88881a3`。仓库里有：
- 8 种方法的 demo 脚本 `demo_*.m`：SRIF、RIFT、LNIFT、SIFT、OS-SIFT、3MRS、CoFSM、MS-HLMO。
- 数据集 `dataset/<类别>/pairN_1.jpg, pairN_2.jpg, gt_N.txt`。**Optical-SAR 有 200 对**（600 个文件）。
- 各脚本 `addpath` 默认指向 `dataset\Optical-Optical\`，评其他类别要手动改路径。

**数据与真值**
- Optical-SAR 子集取自 LNIFT 的 dataset 2：GF-3 SAR 配 Google Earth 光学，覆盖 15 个城市。作者在原有旋转之外又**加了随机尺度 s∈[0.5, 2)**（论文 p.4）。
- 真值是 **2×3 仿射矩阵**，逐对保存在 `gt_N.txt`（例：[dataset/Optical-SAR/gt_1.txt](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/dataset/Optical-SAR/gt_1.txt)），读入后拼成 `H=[gt;0 0 1]`。

**指标（代码：[demo_SRIF.m#L34-L57](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_SRIF.m#L34-L57)）**
1. 用真值 H 投影全部**原始匹配**（暴力最近邻，没有比值检验，也**没有任何 RANSAC**），残差 E<3 的算正确匹配。
2. 按目标点坐标去重（`unique`）。
3. 正确匹配数 `length(E)` 即 NCM。
4. `RMSE = sqrt(mean(E²))`，**只在这些 <3 px 的正确匹配上算**。
5. **NCM < 10 时，RMSE 直接记为 20 px**（惩罚值，不是 ∞，也不剔除）。
6. 结果写入 `RES=[time rmse NCM]`，保存为 `RES_srif.mat`。

- 论文对应定义见 p.8：阈值 ε=3；成功 = NCM ≥ 10；失败样本 RMSE 记 20 px；「不使用 RANSAC 类方法，因为目的是评描述子」。
- **推论**：因为 RMSE 只在 <3 px 的点上计算，成功样本的 RMSE 必然 <3 px；论文也说 SRIF 的 RMSE「在 3 px 阈值下约 2 px」（p.10）。**这个指标主要反映失败率（20 px 惩罚），不反映配准精度。**
- **各方法之间的差异 ⚠**：[demo_RIFT.m#L50-L72](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_RIFT.m#L50-L72) 与 SRIF 的脚本有两处不同：
  - 失败判据写成 `length(E)<=10`（SRIF 是 `<10`）；
  - 对两端都做了 `unique`。

  也就是说，**对比方法的成功阈值差了 1 个点**。
- **SR**：代码里没有计算 SR，是作者从 `RES` 离线统计的（论文 p.8 定义为 NCM ≥ 10）。**聚合脚本未找到。**
- **聚合**：论文 Table 4 按类别取样本均值，「Average」行是 6 类的均值（p.9）。

---

## 3. GDROS（Sun et al., TGRS 2025）

**源码**：`Zi-Xuan-Sun/GDROS` @ `ee6b621`。评测入口是 `test.py`。

**数据与真值**
- WHU-OPT-SAR 切成 512² 后测试集 700 对；OS dataset 测试集 424 对；UBCv2 测试集 1447 对（论文 p.7）。
- 真值是对已配准的 SAR **施加随机仿射**后得到的稠密光流 `truth_flow/*.flo`（论文 p.7；数据加载见 `core/datasets.py` 的 `opt_sar_test`）。
- 测试对是**预先生成好并存盘的**（目录 `sar_warped/`、`truth_flow/`），所有方法用同一批数据。

**评测流程（[test.py#L57-L99](https://github.com/Zi-Xuan-Sun/GDROS/blob/ee6b6216fe780bc03bfb4759ae094daa0cfddc60/test.py#L57-L99)）**
1. 网络输出稠密光流。
2. **LSR**：对光流做**无权最小二乘仿射拟合**（不是 RANSAC），用拟合出的仿射重新生成整幅光流。代码见 [LSmodel.py#L126-L171](https://github.com/Zi-Xuan-Sun/GDROS/blob/ee6b6216fe780bc03bfb4759ae094daa0cfddc60/core/LSRnet/LSmodel.py#L126-L171)。拟合只用行优先展开后的第 10000 到 10000+Npoint 个像素，`Npoint` 默认 200000（[L130-L131](https://github.com/Zi-Xuan-Sun/GDROS/blob/ee6b6216fe780bc03bfb4759ae094daa0cfddc60/core/LSRnet/LSmodel.py#L130-L131)）。
3. 逐像素算 EPE，在**全图所有像素**上取平均，得到该对的 EPE(k)。

**指标定义（[test.py#L40-L45](https://github.com/Zi-Xuan-Sun/GDROS/blob/ee6b6216fe780bc03bfb4759ae094daa0cfddc60/test.py#L40-L45)）**
- `CMR@T = 100·#{k: EPE(k)<T} / N`，T=1,2,3,4,5。这是**图像对级**的成功率。
- `AEPE@T` = 成功子集上 EPE(k) 的均值（代码变量名为 `MAE`）。
- **「RMSE」= `np.var` of EPE(k)**，即**样本间 EPE 的方差**，没有开方。⚠ 论文式(15)把它称作 RMSE，描述为「衡量配准精度的离散程度」（p.7）。从 PDF 文本层无法确认式(15)里有没有开方，**但代码确定是 var**。所以 Table I 里 OS3Flow 那种「RMSE=15524」的数值是 px² 量级的方差。
- 总体 AEPE = 所有对 EPE(k) 的均值（L86）。

**其他要点**
- **失败处理**：稠密光流 + 最小二乘拟合永远有输出，不存在「无输出」的情况。失败样本以大 EPE 计入 AEPE 和方差，在 CMR 中记为不成功。
- **对比方法**：稀疏方法（RIFT2、LNIFT、XoFTR）怎样转成光流或 EPE，**代码中未找到**。论文只说「用官方代码重训」（p.8）。
- **数据来源**：数据集和权重通过百度网盘发布（README）。

---

## 4. SOMA（2025，arXiv 2511.13168）

**源码**：`traslauc/SOMA` @ `481a28c`。评测入口是 `test.py`。

**数据与扰动**
- 论文：主实验用 SEN1-2（SOPatch 精配准版）和 GFGE SO；泛化测试用 WHU-SEN-City 和 OSdataset（p.6–7）。
- 扰动：论文写平移 ≤32 px、尺度 ±0.2、旋转 ±5°；消融实验放宽到 50 px / ±20°（p.6）。
- **代码 ⚠**：测试时**在线随机生成**仿射扰动，范围是 `angle∈U(-20,20)`、`tx,ty∈U(-50,50)`、`scale=1.0`（[datasets/dataloader.py#L75-L93](https://github.com/traslauc/SOMA/blob/481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3/datasets/dataloader.py#L75-L93)）。这是**消融的范围，不是主表的范围**。而且 **`np.random` 没有设种子**，每次运行的测试集都不同。
- 测试集默认 `['SEN1-2','WHU-SEN-City','OSdataset']`，三者混在一个 loader 里（[test.py#L159](https://github.com/traslauc/SOMA/blob/481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3/test.py#L159)）。
- 真值是由仿射矩阵 M 的逆解析算出的稠密 flow（dataloader.py L95-L102）。

**指标（[test.py#L52-L124](https://github.com/traslauc/SOMA/blob/481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3/test.py#L52-L124)）**
- 逐图 `RMSE = sqrt(mean_pixels ||flow_pred − flow_gt||²)`，在**全图全部像素**上计算，**不经过任何估计器**，直接比较网络输出的稠密 flow。
- **CMR@T** = 逐图 RMSE < T 的比例，T = 1..5（L67-L76，L104-L108）。这是**图像对级**的成功率。
- 「Average RMSE」是**按 batch 平均的 batch-RMSE**（L53-L55, L103；batch_size 默认 2），不是逐图 RMSE 的均值。⚠ 论文把 Ravg 定义为「全部测试对 RMSE 的均值」（p.6），和代码的聚合方式略有不同。
- 另外会打印：RMSE<5 的子集上的平均 RMSE；像素级误差 <2 的像素数和平均误差。
- **失败**：稠密输出总是存在，没有剔除。
- **对比方法**（MI、CFOG、DDFN、FFT U-Net、OSMNet 等）的评测代码**未找到**。

---

## 5. Shared Modality（Borisov et al., 2026）

**源码**：`BorisovAN/shmod` @ `e3fa67f`。

**流程**
- 第一步 [compute_keypoints.py](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/scripts/matching_scripts/compute_keypoints.py) 逐对保存匹配点。
- 第二步入口 [compute_matching_metrics.py](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/scripts/matching_scripts/compute_matching_metrics.py) 读取匹配点并计算指标。

**数据与真值**
- MultiSenGE 的 Sentinel-1/2 共配准片，256×256（论文 p.5；[compute_matching_metrics.py#L21](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/scripts/matching_scripts/compute_matching_metrics.py#L21)）。
- **测试时没有施加任何合成几何扰动**：数据加载只读 `s1` 和 `s2` 两个文件夹（[compute_keypoints.py#L58-L93](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/scripts/matching_scripts/compute_keypoints.py#L58-L93)）。所以**真值变换就是恒等变换**。

**指标**
- **CMR@δ / LE@δ**（δ=1..5）：直接用 `‖p_a − p_b‖`（隐含真值为恒等）判定，只针对原始匹配，不经过 RANSAC（[matching/detector.py#L17-L41](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/matching/detector.py#L17-L41)）。
- **ACE**（平均角点误差）：`mean ‖corner − H_est(corner)‖`，在 4 个角点上算（[matching/homography.py#L14-L26](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/matching/homography.py#L14-L26)）。同样隐含真值为恒等。
- **SR** = #(ACE ≤ 40) / N（[compute_matching_metrics.py#L95](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/scripts/matching_scripts/compute_matching_metrics.py#L95)）。论文写作 `ACE < 40`（p.11），⚠ 代码是 `<=`。
- **MMA** = **RANSAC 内点比例**（`inliers_percent`），**不是**相对真值的正确率（[homography.py#L39-L45](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/matching/homography.py#L39-L45)）。论文 p.11 的定义与此一致。

**估计器**
- 所有方法（SIFT、RIFT、RIFT2、DeDoDe、RoMa，含是否做共享模态变换的各组合）共用同一个 `cv2.findHomography(..., cv2.RANSAC)`，参数全部用 OpenCV 默认值：重投影阈值 3 px，maxIters 2000（论文 p.11 写「2000 次迭代、3 px」，与此一致）。
- 模型是**单应**。少于 4 个点返回 None（[homography.py#L32-L45](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/matching/homography.py#L32-L45)）。

**失败计入（[compute_matching_metrics.py#L37-L63, L95-L101](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/scripts/matching_scripts/compute_matching_metrics.py#L37-L101)）**
- 匹配器没有输出时写空文件，对应 `FAILED_MATCHING_STATS`：ACE=NaN，MMA=0，**CMR=0，LE=0**。
- 单应失败或 ACE>40 时，ACE 和 MMA 置为 NaN。
- 聚合：ACE 和 MMA 用 `nanmean`，**只在成功样本上平均**（论文 p.14 也这么写）。CMR 和 LE 用 `mean`，失败样本按 0 计入。⚠ **LE 以 0 计入会把平均定位误差拉低**，论文没有提到这一点。
- 论文 p.11 写「失败时 ACE 视为 ∞」，在代码里等价于「不计入 SR 分子，不参与 ACE 均值」。

**局限**：真值为恒等、又不做扰动，任何输出恒等匹配的方法都会拿到满分。这种协议**无法区分「真正配准」和「近零位移先验」**。

---

## 6. TAR（Cai et al., TGRS 2026）——只有论文

**源码未找到**：论文、arXiv 2605.12064 页面、以及检索结果中都没有代码链接。

- **数据**（论文 p.6）：
  - SEN1-2：训练 12,642 对，测试 1,000 对，256²，10 m。
  - OSdataset（GF-3 聚束模式 + Google Earth）：训练 6,297 对，测试 1,000 对，256²，1 m。
- **扰动**：对 SAR 施加随机仿射，尺度 [0.7,1.3]，旋转 [-35°,35°]，平移为图像尺寸的 10%（p.7）。**没有说明是否固定种子或预存。**
- **指标**（p.7）：
  - `RMSE = sqrt(1/N Σ‖p_i − p_i^gt‖²)`，N 是「匹配点对数」，即在**预测匹配点**上相对真值点计算。
  - `CMR@τ = (1/M) Σ_j I(RMSE_j < τ)`，τ = 1/3/5 px，是**图像对级**成功率。
  - 表中的 RMSE 如何跨样本聚合、失败样本如何计入、是否经过 RANSAC 或仿射拟合、各方法是否共用估计器：**论文未说明**。
- **注意**：表里 RIFT/LNIFT 的 RMSE 为 17–18 px，而 CMR@5 只有约 9%。这说明失败样本很可能以某个大值计入了 RMSE 均值，但**具体规则未找到**。

---

## 7. RRSI（Ye et al., 2026）——只有论文

**源码**：`yeyuanxin110/RRSI` 已克隆，但**仓库为空（没有 commit）**。

- **数据**：OSdataset（1 m，GF-3 + Google Earth，2673 对，按官方划分训练 2011 / 验证 238 / 测试 424）（论文 p.10–11）。
- **扰动**：对预配准的测试对施加随机旋转 ±15°、尺度 [1,1.25]、透视 ±0.08。**每对只生成一次，参数存成协议文件**，例如 OSdataset 的 424 对共用 1 个参数文件，所有方法用同一批变换后的图像（p.11）。
- **估计器**：**所有方法**的匹配点都统一走 **DEGENSAC**，模型为**单应**（p.11）。内点阈值和迭代数**论文未给出**。关键点上限统一为 5000。
- **指标**（p.11）：
  - **RMSE**：用估计的 H 和真值 H_gt 分别投影源图的 **4 个角点**，取两组角点坐标差的 RMSE。**每对截断在 10 px**：超过 10 px 或任何失败（匹配不足、单应失败等）都**记 10 px**。报告值是这些截断值在测试集上的**均值**。
  - **SR(τ)** = 角点 RMSE < τ 的样本比例。
  - **AUC(τ0)**：SR–τ 曲线在 [0,τ0] 上分段线性插值，用梯形法积分后除以 τ0，τ0 = 3/5/10 px。

---

## 8. MatchAnything（He et al., TPAMI 2026）

**源码**
- GitHub `zju3dv/MatchAnything` @ `8cd8c11` **只有 README**（[README.md](https://github.com/zju3dv/MatchAnything/blob/8cd8c1129a6d22dabea9405a869e4fad6ff8b630/README.md)），里面指向 HuggingFace Space。
- 评测代码在 HF Space `LittleFrog/MatchAnything` @ `6a7bcb5` 的 `imcui/third_party/MatchAnything/` 下（下称 `MA/`），不在 GitHub。下面的 permalink 都是 HF 的 blob 链接。

**数据**：Visible-SAR 数据集来自 Xiang et al., TGRS 2023（论文 [82]），共 1209 对，每对带一组**真值对应点**（论文 p.15）。数据以 npz 形式放在 `data/test_data/visible_sar_dataset`，**仓库中不包含数据**。

**评测入口**：[MA/scripts/evaluate/eval_visible_sar.sh](https://huggingface.co/spaces/LittleFrog/MatchAnything/blob/6a7bcb589ec8da3a9e861e799122beaa5eba2193/imcui/third_party/MatchAnything/scripts/evaluate/eval_visible_sar.sh#L14-L17)，调用 `tools/evaluate_datasets.py`，方法名写作 `...@-@ransac_affine`，`--imgresize 832`。

**逐对流程（[MA/tools/evaluate_datasets.py#L146-L222](https://huggingface.co/spaces/LittleFrog/MatchAnything/blob/6a7bcb589ec8da3a9e861e799122beaa5eba2193/imcui/third_party/MatchAnything/tools/evaluate_datasets.py#L146-L222)）**
1. npz 里有 `gt_2D_matches`（N×4）时进入 `gt_match` 模式：源点 = 真值点对的左端，目标 = 右端。
2. 除 FIRE 外，**变换模型一律为仿射**。
3. 仿射估计用 `cv2.estimateAffine2D(..., method=cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99999)`（阈值来自 `--rigid_ransac_thr` 默认值 3.0，L77-L78；函数见 [MA/src/utils/metrics.py#L180-L191](https://huggingface.co/spaces/LittleFrog/MatchAnything/blob/6a7bcb589ec8da3a9e861e799122beaa5eba2193/imcui/third_party/MatchAnything/src/utils/metrics.py#L180-L191)）。maxIters 用 OpenCV 默认值。
4. 用估计的仿射投影真值源点，逐对误差 = 投影点与真值目标点的**平均欧氏距离**（L219-L222）。即**误差算在独立的真值检查点上，不算在方法自己的匹配点上**。
5. 没有真值匹配、只有 GT 单应的数据集：用 4 个角点作为检查点（L158-L161；论文 p.14 描述相同）。

**失败计入**
- `estimateAffine2D` 返回 None 时，H 置为**全零矩阵**（metrics.py L183-L184）。
- `warp_points` 把 |z|<1e-8 的分量置为 1e-8（[MA/src/utils/homography_utils.py#L159-L182](https://huggingface.co/spaces/LittleFrog/MatchAnything/blob/6a7bcb589ec8da3a9e861e799122beaa5eba2193/imcui/third_party/MatchAnything/src/utils/homography_utils.py#L159-L182)），于是所有点被投影到 (0,0)。
- 误差因此等于真值点到原点的平均距离：**是有限的大值，不是 ∞，也不剔除**，在 SR 中自然记为失败。
- 匹配点少于 3 个时 OpenCV 的具体行为**未验证**。

**指标（L231-L236）**
- **SR@{5,10,20} px** = 误差 < 阈值的比例（`error_auc(..., method="success_rate")`）。
- 同时输出 **exact AUC@{5,10,20}**：对排序后的误差做召回曲线的梯形积分，再除以阈值（metrics.py [L361-L396](https://huggingface.co/spaces/LittleFrog/MatchAnything/blob/6a7bcb589ec8da3a9e861e799122beaa5eba2193/imcui/third_party/MatchAnything/src/utils/metrics.py#L361-L396)）。
- 论文 Visible-SAR 表报的是 SR@5/10/20（p.24）。
- **聚合**：全部样本等权。

**其他要点**
- **query_points**：gt_match 模式下真值源点会以 `query_points` 形式传给匹配器。RoMa 版本只拿它算 `query_points_warpped`，评测**没有使用**这个结果，**评测取的仍是 `mkpts0_f/mkpts1_f`，不存在真值泄漏**（third_party/ROMA/roma/matchanything_roma_model.py L92-L97）。
- **对比方法**：这个脚本只能跑 MatchAnything 的两个模型。SIFT、SRIF、RoMa 等基线「用官方代码和权重运行」（论文 p.15）。基线是否走同一个 `ransac_affine` 流程，**代码中未找到**。

---

## 9. MINIMA（Ren et al., CVPR 2025）

**源码**：`LSXI7/MINIMA` @ `796e772`。评测入口是 [test_relative_homo_mmim.py](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py)，参数 `--choose_model 1` 选择 RemoteSensing。

**数据与真值**
- MMIM（Jiang et al. 的多模态匹配数据库）。仓库自带的 zip 只有列表文件：[data/Multi-modality-image-matching-database-metrics-methods.zip](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/data/Multi-modality-image-matching-database-metrics-methods.zip)。
- 解压后 `test_list_2.txt` 包含 7 个 RemoteSensing 子集，各子集的对数为：

  | SAR_Optical | CrossSeason | DayNight | DepthOptical | Infrared_Optical | Map_Optical | Optical_Optical |
  |---|---|---|---|---|---|---|
  | **6** | 5 | 5 | 8 | 4 | 7 | 6 |

  共 41 对。**光–SAR 只有 6 对。**
- 真值是 `.mat` 里的 `T`，经转置后做 1 像素平移修正（MATLAB 1-based 转 0-based），见 [L124-L162](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L124-L162)。论文说真值来自人工标注的匹配（p.6）。
- 真实图像**不施加合成扰动**（MMIM 本身就不对齐）。

**估计器**
- 所有方法统一用 `cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)`，**全部 OpenCV 默认参数**：阈值 3 px、maxIters 2000、conf 0.995（[L381-L389](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L381-L389)）。
- ⚠ 命令行参数 `--ransac_thres`（默认 1.5）**只写进 results.json，并没有传给 RANSAC**（L498, L571, L604）。
- 论文「所有基线用相同的 RANSAC 设置」（p.6）在代码层面成立。
- 匹配点坐标会按缩放比例还原到原图分辨率（例如 `src/utils/data_io_loftr.py` L86-L87），**角点误差在原图像素坐标系下计算**。

**指标（[L277-L295](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L277-L295)）**
- 分别用真值 H 和估计 H 投影 4 个角点，**两组角点各自先经过 `order_corners` 按坐标和/差排序**，再取 4 个角点距离的**平均**。
- ⚠ **排序会掩盖角点对应错误**，例如估计结果差了 90° 旋转或翻转时，误差会被低估。论文只写「四角点平均投影误差」（p.6），没有提到排序。

**失败计入（[L405-L409](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L405-L409)）**
- 单应求解抛异常或返回 None 时，`mean_dist = ∞`，计入 AUC，对所有阈值都算未成功。
- 「Average Mean Dist」只在有限值上平均（L449-L451）。

**AUC 与聚合**
- AUC 用 `error_auc`，计算方式为梯形积分后除以阈值（[src/utils/metrics.py#L160-L176](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/src/utils/metrics.py#L160-L176)）。代码在阈值 {1,3,5,7,10,15,20} 上计算（L421, L458），论文报告的是 @3/5/10px（p.6, p.8 Tab.5）。
- **聚合是两级的**：
  1. 先**在每个子集内**算 AUC；
  2. 再用 `name.split("/")[0]` 把子集归并到 "RemoteSensing"，**对 7 个子集的 AUC 取算术平均**（[L165-L181](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L165-L181), [L461-L463](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L461-L463)）。
- 因此**论文 Tab.5 的 Remote Sensing 数字是按场景宏平均的，光–SAR 在其中只占 1/7 权重**。SAR_Optical 单独的 AUC 只在 results.json 里有，论文没有报。
- **附带指标**：代码还会输出匹配精度 @1/3/5px（相对真值 H，[L298-L320](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L298-L320)），论文没有报。

---

## 10. AnyMatch（Yang et al., ECCV 2026）——只有论文

**源码未找到。**

- **数据**：MMIM 的 13 个跨模态组合，包括遥感和医学，真值为人工标注的对应点（p.11）。遥感结果单独成列（Tab.3, p.14），光–SAR 没有单独报告。
- **指标**：「单应任务报告**四角点最大**重投影误差的 AUC @3/5/10px」（p.11）。
- ⚠ 这与它沿用的 MINIMA 协议不同：MINIMA 代码用的是四角点**平均**。由于没有代码，无法判断论文措辞是否准确。
- **估计器**：「所有鲁棒几何估计使用相同的 RANSAC 超参数」（p.11），但具体数值**论文未给出**。

---

## 11. HOMO-Feature（Gao, Li, Weng, ICCV 2025）

**源码**：`MrPingQi/HOMO_Feature_ImgMatching` @ `5f64b35`。仓库里**只有单对 demo**，大量函数以 `.p` 加密形式提供，**没有 NCM/DCM/RMSE_CP100/SR 的评测代码**。GCZ 数据集「将随论文发布」（论文 p.6），仓库中也没有。

**指标（只有论文依据，p.6）**
- **NCM**：相对标注真值，误差 ≤3 px 的匹配数。
- **DCM** = lg(S_area·S_uniform·NCM + 1)，其中 S_area 是匹配点凸包面积与图像面积之比，S_uniform 是最近邻距离的均匀度。
- **RMSE_CP100**：预设 10×10 共 100 个均匀检查点，用估计的**单应**投影后，与「标签导出的坐标」计算 RMSE。
- **SR**：NCM > 50 的样本比例。
- **失败**：RMSE_CP100 在匹配失败时**记为 5**（p.7：「Matching failures were assigned a high value of 5」）。
- 传统方法使用相同的关键点和 NN 匹配，**所有结果都经过 RANSAC**（p.7）。RANSAC 的参数**论文未给出**。

**代码中可见的匹配器内部参数**（这些是匹配器内部的，不是评测器的）
- demo 默认 `trans_form='affine'`，FSC 外点剔除阈值 `Error=5`（注释写「5 或 3」），见 [A_HOMO_demo.m#L19](https://github.com/MrPingQi/HOMO_Feature_ImgMatching/blob/5f64b3536aae16154c7831936508dd67f97e2be2/HOMO_image_matching_demo/A_HOMO_demo.m#L19)、[#L40](https://github.com/MrPingQi/HOMO_Feature_ImgMatching/blob/5f64b3536aae16154c7831936508dd67f97e2be2/HOMO_image_matching_demo/A_HOMO_demo.m#L40)。
- FSC 迭代 800 次；内点少于 20 时直接判为失败并返回空（[Outlier_Removal.m#L7-L14](https://github.com/MrPingQi/HOMO_Feature_ImgMatching/blob/5f64b3536aae16154c7831936508dd67f97e2be2/HOMO_image_matching_demo/func_HOMO/Outlier_Removal.m#L7-L14)）。

---

## 12. 横向对照表

表头缩写：TP = true positive（相对真值的正确匹配）；px = 像素。

| 论文 | 光–SAR 数据（测试规模） | GT 来源 | 误差算在哪里 | 主指标 | 成功判定 | 失败计入 | 估计器 / 模型 | 各方法是否共用估计器 | 聚合 | 依据 |
|---|---|---|---|---|---|---|---|---|---|---|
| RIFT | 自建 SAR-optical 10 对 | 人工 5 点拟合仿射 | NBCS 后的匹配点（残差<3px 为 TP） | NCM、ME、RMSE、SR | NCM ≥ 4 | 未明说；SR 过低的方法整体不报 ME/RMSE | NBCS（评测）；demo 用 FSC 仿射 | 是（论文） | 按类别取均值 | 论文 |
| SRIF | 自建 Optical-SAR 200 对（加尺度 0.5–2） | 逐对 2×3 仿射 gt.txt | 原始匹配中相对 GT <3px 的点 | NCM、RMSE、SR | NCM ≥ 10（RIFT 脚本写成 >10） | RMSE 记 20px | **不做 RANSAC** | 是（脚本逻辑相同，阈值差 1） | 按类别取均值 | 代码+论文 |
| GDROS | WHU-OPT-SAR 700 / OS 424 / UBCv2 1447 | 合成仿射 → 稠密 flow（预存） | 全图全像素 EPE | CMR@1–5、AEPE@T、AEPE、"RMSE"（实为 EPE 方差） | 逐对 EPE < T | 以大 EPE 计入 | 最小二乘仿射拟合（非鲁棒） | 基线流程未找到 | 按样本 | 代码 |
| SOMA | SEN1-2 / GFGE / WHU-SEN-City / OS | 在线随机仿射 → flow（无种子） | 全图全像素 flow RMSE | CMR@1–5、Ravg | 逐图 RMSE < T | 不存在无输出 | 无（直接比 flow） | 未找到 | 按 batch 取平均 | 代码 |
| SharedMod | MultiSenGE（256²） | **恒等（不扰动）** | 原始匹配（CMR/LE）；4 角点（ACE） | SR、ACE、MMA、CMR@δ、LE@δ | ACE ≤ 40px | ACE/MMA 剔除（nanmean）；CMR/LE 记 0 | cv2 RANSAC 单应，3px，2000 次 | 是 | 按样本 | 代码 |
| TAR | SEN1-2 1000 / OS 1000 | 随机仿射（尺度 0.7–1.3，旋转 ±35°，平移 10%） | 预测匹配点 vs GT 点 | RMSE、CMR@1/3/5 | 逐对 RMSE < τ | 未说明 | 未说明 | 未说明 | 未说明 | 论文 |
| RRSI | OSdataset 424 | 预存随机单应（旋转 ±15°，尺度 1–1.25，透视 0.08） | 4 角点（估计 H vs GT H） | 角点 RMSE（截断 10）、SR(τ)、AUC@3/5/10 | 角点 RMSE < τ | **记 10px（截断值）** | DEGENSAC 单应 | 是 | 按样本 | 论文 |
| MatchAnything | Visible-SAR 1209 | 数据集自带真值对应点 | 独立 GT 点（经估计仿射投影） | SR@5/10/20（另有 AUC） | 平均误差 < 阈值 | 零矩阵 → 点投到原点（有限大值） | cv2 estimateAffine2D RANSAC 3px，conf 0.99999 | 本模型是；基线未找到 | 按样本 | 代码 |
| MINIMA | MMIM SAR_Optical 6（RS 共 41） | MMIM 人工标注 T | 4 角点（**排序后**取平均距离） | AUC@3/5/10（代码 1–20） | — | ∞ 计入 AUC | cv2 RANSAC 单应，默认 3px | 是 | **先按子集算 AUC 再宏平均** | 代码 |
| AnyMatch | MMIM RS | 同上 | 4 角点**最大**误差（论文措辞） | AUC@3/5/10 | — | 未说明 | RANSAC（参数未给） | 是（论文） | 未说明 | 论文 |
| HOMO | GCZ VIS-SAR 组 | 人工配准标签 | 100 个格网检查点（CP100）；NCM ≤3px | NCM、DCM、RMSE_CP100、SR | NCM > 50 | RMSE_CP100 记 5 | RANSAC（参数未给），单应 | 是（论文） | 按组取均值 | 论文 |

**可以直接借鉴的做法**
1. **在独立检查点上计算误差，不在方法自己的匹配点上计算**。MatchAnything 用真值点，RRSI、MINIMA 用角点，HOMO 用 CP100。SRIF、RIFT、TAR 在匹配点上算 RMSE，这会让 RMSE 被内点阈值截断，失去区分度。
2. **所有方法统一过同一个估计器**，模型与任务一致；对仿射任务就用 `estimateAffine2D`（MatchAnything 的做法）。报告时写明阈值、conf、迭代数。MINIMA 的 `--ransac_thres` 参数实际没生效，说明这些参数必须从代码里核对。
3. **失败样本必须以显式的大值或截断值计入**：RRSI 记 10 px，SRIF 记 20 px，MINIMA 记 ∞。同时报 SR 或 AUC。不要像 SharedMod 的 LE 那样按 0 计入，也不要只在成功子集上报均值（SharedMod 的 ACE、GDROS 的 AEPE@T）而不同时报 SR。
4. **扰动参数逐对预生成并存盘**（RRSI、GDROS），**不要在线随机、不设种子**（SOMA）。**不要用恒等真值做评测**（SharedMod）。
5. **聚合方式要写明**：按样本还是按场景宏平均（MINIMA）。两者在小子集上可能差别很大。

---

## 13. 源码清单

| repo URL | commit | 本地路径 | 评测入口文件 |
|---|---|---|---|
| https://github.com/LJY-RS/RIFT-multimodal-image-matching | 7ea830e2f13cc3c226f975fe9e98b7666a8f26fb | D:\Code\refs\RIFT-multimodal-image-matching | 无评测代码（只有 `RIFT_demo.m`）；批量评测见 SRIF 仓库 `demo_RIFT.m` |
| https://github.com/LJY-RS/SRIF | 88881a3a8789d0bed6a8df91b943e02f2593a0cd | D:\Code\refs\SRIF | `demo_SRIF.m`（以及 `demo_<方法>.m`）；真值在 `dataset/Optical-SAR/gt_N.txt` |
| https://github.com/Zi-Xuan-Sun/GDROS | ee6b6216fe780bc03bfb4759ae094daa0cfddc60 | D:\Code\refs\GDROS | `test.py`（加上 `core/LSRnet/LSmodel.py`） |
| https://github.com/traslauc/SOMA | 481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3 | D:\Code\refs\SOMA | `test.py`（加上 `datasets/dataloader.py`） |
| https://github.com/BorisovAN/shmod | e3fa67f2bb2596b4046f5c67944cb9809bf9756f | D:\Code\refs\shmod | `scripts/matching_scripts/compute_matching_metrics.py`（加上 `matching/homography.py`、`matching/detector.py`） |
| https://github.com/yeyuanxin110/RRSI | 无（空仓库） | D:\Code\refs\RRSI | 无 |
| https://github.com/zju3dv/MatchAnything | 8cd8c1129a6d22dabea9405a869e4fad6ff8b630 | D:\Code\refs\MatchAnything | 只有 README |
| https://huggingface.co/spaces/LittleFrog/MatchAnything （HF Space，非 GitHub） | 6a7bcb589ec8da3a9e861e799122beaa5eba2193 | D:\Code\refs\MatchAnything-hf-space | `imcui/third_party/MatchAnything/tools/evaluate_datasets.py`；`scripts/evaluate/eval_visible_sar.sh` |
| https://github.com/LSXI7/MINIMA | 796e7721174f9f829b79b3702bf8c2ae9a3d447a | D:\Code\refs\MINIMA | `test_relative_homo_mmim.py --choose_model 1` |
| https://github.com/MrPingQi/HOMO_Feature_ImgMatching | 5f64b3536aae16154c7831936508dd67f97e2be2 | D:\Code\refs\HOMO_Feature_ImgMatching | 无评测代码（只有 `HOMO_image_matching_demo/A_HOMO_demo.m`） |
| https://github.com/OnderT/XoFTR | e0fbea431b30be9742effbf5577c90aa8eb938f9 | D:\Code\refs\XoFTR | 未深入（不涉及光–SAR） |
| TAR / AnyMatch | — | — | 源码未找到 |

**范围外线索（未核实）**：arXiv 2604.10217《Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?》是一个专门评测「预训练匹配器做光–SAR 配准」的工作，不在本目录中，可作为后续调研对象。
