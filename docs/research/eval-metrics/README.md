# 光-SAR 配准评价指标：实现调研汇总

本目录是 GitHub issue [Guochu0-0/moon-exp#7](https://github.com/Guochu0-0/moon-exp/issues/7)「光-SAR 配准评价指标的实现调研」的产出。调研对象是已有工作在论文和源码里**实际怎样**计算配准评价指标。它为下游 [#2](https://github.com/Guochu0-0/moon-exp/issues/2)「定义评价协议与指标」提供挑选依据，**本文不替 #2 做决定**。

项目背景：月球 CE-2 光学与 Mini-RF SAR 的 patch 对做**仿射**配准。训练不用标注；Val/Test 中大部分 patch 对有人工点对标注。

| 分文件 | 内容 |
|---|---|
| [01-survey-benchmarks.md](01-survey-benchmarks.md) | 光-SAR 综述与 benchmark：MultiResSAR、SRIF、OS-Eval、SOMA-1M、ArePretrainedMatchers(APM)，以及 RoMa、XoFTR、MINIMA |
| [02-theses-a.md](02-theses-a.md) | 学位论文：谢志华、王丽娜 |
| [03-theses-b.md](03-theses-b.md) | 学位论文：张俊、吕宁 |
| [04-cross-modal.md](04-cross-modal.md) | 跨模态匹配：GDROS、SOMA、SharedMod、TAR、RRSI、MatchAnything、MINIMA、HOMO 等 |
| [05-selfsup-homography.md](05-selfsup-homography.md) | 自/无监督匹配与单应估计：RIPE(++)、SiLK、RaCo、GeoFormer、CA-Unsup、S2M2-SAR 等 |
| [sources.md](sources.md) | 合并后的源码清单（repo、commit、本地路径、入口文件）和一键恢复脚本 |

---

## 1. 横向对照表

表格约定：
- **依据**列：「码」表示结论以源码为准（论文和代码不一致时按代码记）；「文」表示只有论文依据。
- 码行里如果某一格来自论文，就在该格末尾标「(文)」。
- 「§」后面是分文件的小节号。

permalink 前缀缩写（完整 commit 见 [sources.md](sources.md)）：
- `SRIF` = `https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/`
- `RSIM` = `https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/`
- `GF` = `https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/`

| # | 工作 | 领域 | 指标 | 误差算在什么上 | 成功阈值 | 失败计入 | 估计器（模型 / 阈值 / 是否统一） | 聚合 | 依据 |
|---|---|---|---|---|---|---|---|---|---|
| <a id="t-multires"></a>R1 | MultiResSAR 综述 | 光-SAR | SR、NCM、RMSE、TM | 人工 GT 控制点，经估计变换映射（GT 误差 ≤1 px） | NCM ≥ 20 且 RMSE ≤ 10 px；单点正确阈值未找到 | SR 分母含失败（记 0）；RMSE/NCM 是否只在成功对上算未找到（推断：只在成功对上） | 各方法自带代码和推荐参数，**不统一**；细节未找到 | "average"，方式未找到 | 文 [01 §1](01-survey-benchmarks.md) |
| <a id="t-srif"></a>R2 | SRIF 仓库 + 论文 | 光-SAR（6 模态） | NCM、RMSE、time；SR 在代码外离线统计(文) | 方法的**原始匹配点**，用 GT 2×3 仿射判定 <3 px 为正确；**RMSE 只在这些正确点上算** | 对级 NCM ≥ 10；RIFT/MSHLMO/OSSIFT 脚本写的是 `<=10` 判失败 | **RMSE 记 20 px** | 评测**不跑 RANSAC**；exe 内部是否剔外点不可审（01 与 04 的说法不同，见 §4 #1） | 代码逐对存 RES；论文按类别取均值(文) | 码 [01 §2](01-survey-benchmarks.md)、[04 §2](04-cross-modal.md)；[demo_SRIF.m#L34-L54](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_SRIF.m#L34-L54) |
| <a id="t-rift"></a>R3 | RIFT 论文 | 光-SAR（60 对，其中 SAR 10 对） | NCM、ME、RMSE、SR | NBCS 之后的匹配点，相对人工 5 点拟合的仿射，<3 px 为正确 | NCM < 4 判失败 | 未明说；SR 过低的方法整体不报 ME/RMSE | NBCS；demo 用 FSC affine 2 px(码，相对估计 H，无 GT) | 按类别均值 | 文 [04 §1](04-cross-modal.md)；demo 码 [01 §3](01-survey-benchmarks.md) |
| <a id="t-oseval"></a>R4 | OS-Eval | 光-SAR | ME（平均欧氏距离，不是 RMSE）、CMR（内点数/全部匹配数） | **独立 GT 检查点**（金属杆）：内点上 LSQ 拟合仿射，再经 RPC+DEM 投影 | 匹配数 ≤ 20 判失败 | ME = 999、CMR = 0；0 匹配时直接 return，**不写结果** | **统一** affine RANSAC 3 px，迭代 min(C(N,3),10000)，每对重复 20 次；只有 4 个指定方法名走这个分支 | 对内平均；跨对未找到 | 码 [01 §4](01-survey-benchmarks.md)；[matchFunc.cpp#L343-L395](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L343-L395) |
| <a id="t-soma1m"></a>R5 | SOMA-1M | 光-SAR | 角点误差 AUC@5/10/20 px | 角点；GT = 合成单应扰动的逆（种子由样本 ID 决定） | AUC 多档 | 未找到 | **统一** H-RANSAC 1.5 px、10k 次、conf 0.9999 | 按测试集和分辨率子集分报；方式未找到 | 文 [01 §8](01-survey-benchmarks.md) |
| <a id="t-apm-sn9"></a>R6 | APM：SpaceNet9 | 光-SAR | MeanErr、S@5/10、Fail；代码另有 median、S@1/2/3、CI | 人工 tie point；tiled **分段模型**；误差是 **resize 后的 px** | 点级：一对图内误差 ≤ τ 的 tie point 比例；"good enough" = 均值 < 8 px(文) | 置 NaN，**均值跳过**，单独报 failure_rate | **统一** cv2 affine RANSAC 3 px（其余用 OpenCV 默认），每个 tile 至少 4 个内点；CLI 默认值与论文不同 | 对内 mean/median，再在成功对上取平均，附 bootstrap CI | 码 [01 §9.1](01-survey-benchmarks.md)；[RSIM…spacenet9_matcher_benchmark.py#L1196-L1227](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L1196-L1227) |
| <a id="t-apm-srif"></a>R7 | APM：SRIF | 光-SAR | 同 R6 | 用 GT 仿射生成 **20×20 网格伪 tie point**（论文写的是 4 角点） | 同 R6 | 同 R6 | 同 R6；代码默认 tiled（论文写不切块） | 按子数据集 | 码 [01 §9.2](01-survey-benchmarks.md)；[srif_matcher_benchmark.py#L171-L204](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/srif_matcher_benchmark.py#L171-L204) |
| <a id="t-gdros"></a>R8 | GDROS | 光-SAR | CMR@1–5（对级成功率）、AEPE@T、AEPE、"RMSE"（**实为 EPE 方差**） | 全图全像素 EPE（光流经 LSQ 仿射重建）；GT = 预存的合成仿射光流 | 逐对 EPE < T | 总有输出，失败以大 EPE 计入；AEPE@T 只在成功子集上算 | 非鲁棒的 LSQ 仿射拟合；基线怎么转换未找到 | 按样本 | 码 [04 §3](04-cross-modal.md)；[test.py#L40-L45](https://github.com/Zi-Xuan-Sun/GDROS/blob/ee6b6216fe780bc03bfb4759ae094daa0cfddc60/test.py#L40-L45) |
| <a id="t-soma"></a>R9 | SOMA（traslauc） | 光-SAR | CMR@1–5、Ravg | 全像素 flow RMSE，不经估计器；GT = **在线随机仿射，没有设种子** | 逐图 RMSE < T | 无剔除 | 无 | batch-RMSE 的均值（论文写的是逐对均值） | 码 [04 §4](04-cross-modal.md)；[test.py#L52-L124](https://github.com/traslauc/SOMA/blob/481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3/test.py#L52-L124) |
| <a id="t-shmod"></a>R10 | SharedMod | 光-SAR | SR、ACE、MMA（= RANSAC 内点比例）、CMR@δ、LE@δ | **GT = 恒等（不加扰动）**；CMR/LE 在原始匹配上算，ACE 在 4 个角点上算 | ACE ≤ 40 px（论文写 < 40） | 无输出时 CMR/LE **记 0**；ACE/MMA 置 NaN 后 nanmean | **统一** cv2 H-RANSAC 3 px、2000 次 | 按样本 | 码 [04 §5](04-cross-modal.md)；[compute_matching_metrics.py#L37-L101](https://github.com/BorisovAN/shmod/blob/e3fa67f2bb2596b4046f5c67944cb9809bf9756f/scripts/matching_scripts/compute_matching_metrics.py#L37-L101) |
| <a id="t-tar"></a>R11 | TAR | 光-SAR | RMSE、CMR@1/3/5 | 预测匹配点与 GT 点比较；GT = 随机仿射 | 对级 RMSE < τ | 未说明（数值上看像是以大值计入） | 未说明 | 未说明 | 文 [04 §6](04-cross-modal.md) |
| <a id="t-rrsi"></a>R12 | RRSI | 光-SAR | 角点 RMSE（截断 10）、SR(τ)、AUC@3/5/10 | 4 角点，估计 H 对 GT H；GT = **预存**的随机单应 | 角点 RMSE < τ | **截断，记 10 px** | **统一** DEGENSAC 单应，阈值未给 | 按样本均值 | 文 [04 §7](04-cross-modal.md) |
| <a id="t-matchanything"></a>R13 | MatchAnything | 光-SAR（1209 对） | SR@5/10/20（另输出 AUC） | **独立 GT 对应点**，经估计仿射投影，取平均欧氏距离 | 平均误差 < 阈值 | 估计失败时 H 为零矩阵，点被投到原点，得到**有限的大值**并计入 | cv2.estimateAffine2D RANSAC 3 px、conf 0.99999；基线是否走同一流程未找到 | 全样本等权 | 码 [04 §8](04-cross-modal.md)；[evaluate_datasets.py#L146-L222](https://huggingface.co/spaces/LittleFrog/MatchAnything/blob/6a7bcb589ec8da3a9e861e799122beaa5eba2193/imcui/third_party/MatchAnything/tools/evaluate_datasets.py#L146-L222) |
| <a id="t-homo"></a>R14 | HOMO-Feature | 光-SAR（GCZ） | NCM(≤3 px)、DCM、RMSE_CP100、SR | 10×10 均匀检查点，经估计单应投影后与标签坐标比较 | SR：NCM > 50 | RMSE_CP100 **记 5** | RANSAC 单应，参数未给；统一 | 按组均值 | 文 [04 §11](04-cross-modal.md) |
| <a id="t-s2m2"></a>R15 | S2M2-SAR | 光-SAR（模板平移） | RMSE(All)、CMR@1/5、RMSE(T=5) | 预测位置与 GT 位置比较 | CMR：误差在 T px 以内 | 总有输出；RMSE(T=5) **剔除**误差 > 5 的样本 | 无（热图 argmax） | 按样本 | 文 [05 §9](05-selfsup-homography.md) |
| <a id="t-xie"></a>R16 | 谢志华 | 光-SAR | NCM（FSC 内点数）、RMSE、时间 | **方法自己的 FSC 内点**，经 20 对人工点拟合的 H 映射后求残差 | NCM ≥ 3 | 记 "*"，不参与平均 | NNDR+FSC 仿射；是否与对比方法统一未说明 | 逐对列表 | 文 [02 §1](02-theses-a.md) |
| <a id="t-wang"></a>R17 | 王丽娜 | 光-SAR | ME、RMSE、NCM、CMR = 2NCM/(N1+N2) | "控制点"；来源和变换是否为估计值都没写明 | NCM ≥ 4（RMSE 16–22 px 的结果照样算成功） | 记 "*"/"---"；平均只在成功组上算 | NNDR+FSC；Ch4 用 SSD 平移；对比方法各用自带流程 | 逐组 | 文 [02 §2](02-theses-a.md) |
| <a id="t-zhang"></a>R18 | 张俊 ch3/ch5 | 光-SAR/多模态 | RMSE、MAE(ch5)、NOCC、MI、RT | 人工 GT 点（ch3 约 30 个；ch5 40–60 个子像素点，是否真在这些点上算是**根据数值推断**的） | 无阈值 | ch3 写 "Failed"，其余栏照报；ch5 **照报数百 px** | RANSAC（ch5 另加 LSQ 6 参数仿射）；对比方法用各自的原设置 | 逐对 | 文 [03 §1.2、§1.4](03-theses-b.md) |
| <a id="t-zhang4"></a>R19 | 张俊 ch4 | SAR-SAR/多模态 | RMSE、NOCC、MI、ROCC、RT | **方法自身匹配对的自洽残差**（Manual 行反而比自动方法差） | 无 | 记 "*" | RANSAC，**所有方法共用**，模型写的是"单应" | 每对跑 10 次取平均 | 文 [03 §1.3](03-theses-b.md) |
| <a id="t-lv"></a>R20 | 吕宁 | 光-SAR | RMSE、NCM（= 剔除外点后的内点数）、Repeatability | 匹配关键点对在（估计的）仿射下的残差 | ch4：正确匹配 < 5 对判失败 | 记 "-"，不计入统计 | NNDR+FSC 仿射，**对比方法用同一实现** | 逐组 | 文 [03 §2](03-theses-b.md) |
| <a id="t-minima"></a>R21 | MINIMA mmim | 跨模态（遥感 7 类，其中光-SAR 6 对） | 角点 AUC@3/5/10（代码算 1–20）、匹配精度@1/3/5 | 4 角点，**经 order_corners 各自重排后**取平均 | AUC 多档 | 记 ∞ 计入 AUC；Average Mean Dist 先过滤 inf | **统一** cv2 H-RANSAC，OpenCV 默认（3 px/2000/0.995）；`--ransac_thres` **没有生效** | 子集内算 AUC，再对 7 个子集**宏平均** | 码 [01 §10](01-survey-benchmarks.md)、[04 §9](04-cross-modal.md)；[L277-L295](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L277-L295) |
| <a id="t-anymatch"></a>R22 | AnyMatch | 跨模态（MMIM） | 角点 AUC@3/5/10 | 4 角点**最大**误差（论文措辞） | AUC | 未说明 | RANSAC 统一，参数未给 | 未说明 | 文 [04 §10](04-cross-modal.md) |
| <a id="t-xoftr"></a>R23 | XoFTR | 跨模态（可见光-热红外，位姿） | pose AUC@5/10/20° | GT 相对位姿 | AUC；阈值参数被**硬编码覆盖** | 记 ∞ 计入 AUC | E-RANSAC 1.5 px（测试）；各方法用自己的 | 场景内算 AUC，再对场景平均 | 码 [01 §5](01-survey-benchmarks.md) |
| <a id="t-xcp"></a>R24 | XCP-Match | 跨模态（可见光-红外） | 单应任务报 **pose AUC**（按平面近似） | 合成单应 GT | AUC@5/10/20 | 未说明 | RANSAC 3 | 未说明 | 文 [05 §8](05-selfsup-homography.md) |
| <a id="t-roma"></a>R25 | RoMa / DKM | 自然图像单应（HPatches） | 角点 AUC@3/5/10 | 4 角点，误差**归一化到短边 480** | AUC | H 置为全零且 [2,2]=1，得到大误差并计入 | H-RANSAC，阈值 3·s px、conf 0.99999 | 全部对算一个 AUC；RoMa 定义了 ignore_seqs 但没有用 | 码 [01 §6](01-survey-benchmarks.md)；[hpatches#L79-L91](https://github.com/Parskatt/RoMa/blob/77f8d68803526dcddfd9b7a46bc76125bdc25f15/romatch/benchmarks/hpatches_sequences_homog_benchmark.py#L79-L91) |
| <a id="t-ripe"></a>R26 | RIPE（glue-factory） | 自然图像单应 | 角点 AUC@1/3/5 | 4 角点 (0..W, 0..H)，短边 480 | AUC | 记 inf，保留 | PoseLib H；RIPE 配置写 0.5 px（论文没写）；`ransac_th=-1` 时**在测试集上扫描阈值** | 全样本 AUC；标量取中位数 | 码 [05 §2](05-selfsup-homography.md)；[GF…eval/hpatches.py#L101-L105](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/hpatches.py#L101-L105) |
| <a id="t-silk"></a>R27 | SiLK | 自然图像单应 | HEA、AUC@1/2/3 | 4 角点 (W−1, H−1)，480 | mean_dist ≤ 阈值 | 记为**阈值 + 1** | cv2 RANSAC 默认 3 px；对比方法的缓存预测走同一流程 | 全样本 | 码 [05 §4](05-selfsup-homography.md)；[hpatches_metrics.py#L204-L248](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/lib/metrics/hpatches_metrics.py#L204-L248) |
| <a id="t-geoformer"></a>R28 | GeoFormer HPatches | 自然图像单应 | Correct@、AUC@1/3/5/10；论文报 AUC@3/5/10 和 mAUC | 4 角点 (w−1, h−1)，480 | AUC | 记 NaN（排序后排在最后，等价于失败） | 入口用 cv RANSAC 3；函数默认却是 degensac 2 | 全样本，另分 i/v 两组 | 码 [05 §11](05-selfsup-homography.md) |
| <a id="t-fire"></a>R29 | GeoFormer FIRE | 单应（视网膜，控制点） | 控制点平均误差、AUC@1..25、Failed/Inaccurate/Acceptable 三类占比 | GT 控制点经估计 H 映射，在原始分辨率上算 | MAE > 50 或 MEE > 20 判为 Inaccurate | 记 **1e6**，归为 Failed | cv2 RANSAC 15 | 按 S/A/P 三组分别算 AUC，再平均得 mAUC | 码 [05 §11](05-selfsup-homography.md)；[fire_helper.py#L159-L182](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/fire_helper.py#L159-L182) |
| <a id="t-caunsup"></a>R30 | CA-Unsup | 单应（视频帧，人工点） | 点 L2 均值；3 px 内点率(文，代码未找到) | 人工 6 点；每个点**取正反两个方向中较小的误差** | — | 直接回归，没有失败分支 | 网络直接回归；基线用 RANSAC/MAGSAC | 每对 6 点取均值，再按 5 个类别求均值 | 码 [05 §10](05-selfsup-homography.md)；[test.py#L146-L152](https://github.com/JirongZhang/DeepHomography/blob/3e811b7d06f84de34eae056baaf90ca886ae3908/Oneline-DLTv1/test.py#L146-L152) |
| <a id="t-raco"></a>R31 | RaCo | 自然图像单应（检测器） | 角点 AUC@1/3 | 4 角点，短边 640 | AUC | 未说明 | GT 对应 + PoseLib DLT，不用 RANSAC；README 里同一方法换估计器后 AUC@1px 从 40.4 变为 44.7 | 未说明 | 文 [05 §5](05-selfsup-homography.md) |

没有单列成行的：CAPS、GIM（信息太少，见 05 §6–7）；APM 的 SARptical 检索任务（01 §9.3）。

---

## 2. 主流做法与分歧点

### 2.1 指标
- **主流**：
  - 光-SAR 传统方法和学位论文：**NCM + RMSE (+SR, 时间)**，例如 [R1](#t-multires)、[R2](#t-srif)、[R16](#t-xie)。
  - 深度方法和单应评测：**角点误差 AUC@多档**，例如 [R5](#t-soma1m)、[R21](#t-minima)、[R26](#t-ripe)。
  - 另有一类是**图像对级成功率** SR/CMR@T，例如 [R13](#t-matchanything)、[R8](#t-gdros)、[R12](#t-rrsi)。
- **分歧（同名不同义）**：
  - 「RMSE」至少有这几种：GT 检查点误差（[R1](#t-multires)、[R18](#t-zhang)）；只在 <3 px 正确点上算（[R2](#t-srif)）；自洽残差（[R19](#t-zhang4)、[R20](#t-lv)）；EPE 方差（[R8](#t-gdros)）。
  - 「CMR」有 4 种含义：2NCM/(N1+N2)（[R17](#t-wang)）；内点数/匹配数（[R4](#t-oseval)）；图像对级成功率（[R8](#t-gdros)、[R9](#t-soma)、[R11](#t-tar)）；恒等 GT 下的匹配级正确率（[R10](#t-shmod)）。
  - 「MMA」在 [R10](#t-shmod) 里是 RANSAC 内点比例，不是相对 GT 的正确率。
- **已知坑**：
  - **残差 RMSE 偏乐观**：RIFT FSC、SAR-SIFT 输出的 rmse 是自洽残差（01 §3、§7）；张 ch4 的 Manual 行反而比自动方法差；吕宁的 RMSE 全部落在 1–2 px。
  - SRIF 的 RMSE 被 3 px 截断，**主要反映失败率**（04 §2 推论）。
  - NCM 如果只是 FSC/RANSAC 内点数，就没有经过 GT 核验（[R16](#t-xie)、[R20](#t-lv)）。

### 2.2 误差算在什么上
- **主流**：
  - (a) **独立人工检查点 + 估计变换**：[R1](#t-multires)、[R4](#t-oseval)、[R13](#t-matchanything)、[R6](#t-apm-sn9)、[R29](#t-fire)。
  - (b) **4 角点，比较 GT 变换与估计变换**：[R5](#t-soma1m)、[R12](#t-rrsi)、[R21](#t-minima)、[R25](#t-roma)–[R28](#t-geoformer)。
- **分歧**：
  - (c) 在 GT 下核验方法的匹配点：[R2](#t-srif)、[R3](#t-rift)、[R11](#t-tar)。这衡量的是匹配器本身（01 §11）。
  - (d) 稠密全像素：[R8](#t-gdros)、[R9](#t-soma)。
  - (e) 自洽残差：[R19](#t-zhang4)、[R20](#t-lv)。
  - 网格检查点：[R14](#t-homo) CP100、[R7](#t-apm-srif) 20×20。
- **已知坑**：
  - **无扰动恒等 GT**：[R10](#t-shmod) 下，"近零位移先验"也能拿满分。
  - 角点定义不统一：W 与 W−1 两种写法（05 §2）。MINIMA 的 order_corners 在大旋转下会**低估**误差。
  - 误差的 px 单位不统一：可能是 resize 后的（[R6](#t-apm-sn9)），也可能归一化到 480（[R25](#t-roma)）。
  - APM tiled 模式实际是分段模型，不是一个全局变换。
  - **人工标注噪声约 1 px**：CA-Unsup v2 标注把误差从 1.82 降到 0.88；MultiResSAR 的 GT 误差 ≤1 px；张 ch5 也自述人工点有误差。
  - CA-Unsup 取双向误差中较小的一个，偏乐观。
  - 谢志华的做法把 RMSE 与 NCM 耦合在一起（02 §3）。
  - 在线随机扰动且不设种子：[R9](#t-soma)。对照做法是预存扰动参数：[R12](#t-rrsi)、[R8](#t-gdros)。

### 2.3 成功阈值
- **主流**：
  - 光-SAR 文献常用 **NCM 下限**：20（[R1](#t-multires)）、10（[R2](#t-srif)）、4（[R3](#t-rift)、[R17](#t-wang)）、3（[R16](#t-xie)）、5（[R20](#t-lv)）、>50（[R14](#t-homo)）。
  - 误差阈值型：
    - 单档或多档 SR：5/10/20（[R13](#t-matchanything)）、1–5（[R8](#t-gdros)、[R9](#t-soma)）、1/3/5（[R11](#t-tar)）。
    - AUC 档位：1/3/5（[R26](#t-ripe)）、1/2/3（[R27](#t-silk)）、3/5/10（[R21](#t-minima)、[R25](#t-roma)、[R12](#t-rrsi)）、5/10/20（[R5](#t-soma1m)）。
- **分歧**：
  - 成功是在对级判定（大多数）还是点级判定（[R6](#t-apm-sn9) S@τ 是一对图内 tie point 的比例）。
  - 附加的"粗失败"门槛：ACE ≤ 40（[R10](#t-shmod)）；MAE > 50 或 MEE > 20（[R29](#t-fire)）；APM 的 "good enough" 是均值 < 8 px。
- **已知坑**：
  - **只按 NCM 判成功**时，RMSE 20 px 的结果也算成功（王 Ch5，02 §3）。
  - 边界写法 `<` 与 `<=` 不一致：SRIF 各脚本之间、SharedMod 的论文与代码之间都有。
  - 阈值参数被硬编码覆盖（XoFTR）。
  - AUC 的积分方式不同：梯形积分（RIPE/MINIMA）、离散均值（FIRE）、分段线性插值（RRSI）。

### 2.4 失败计入
- **主流**：失败**保留在分母里**，但记法不同：
  - ∞ 或 NaN：[R21](#t-minima)、[R26](#t-ripe)、[R23](#t-xoftr)、[R28](#t-geoformer)。
  - 固定惩罚或截断：20 px（[R2](#t-srif)）、10 px（[R12](#t-rrsi)）、5（[R14](#t-homo)）、999（[R4](#t-oseval)）、1e6（[R29](#t-fire)）、阈值+1（[R27](#t-silk)）。
  - 有限的大值：[R13](#t-matchanything)、[R25](#t-roma)。
- **分歧**：
  - **剔除后单独报失败率**：[R6](#t-apm-sn9)、[R10](#t-shmod)（ACE）、[R15](#t-s2m2)（RMSE(T=5)）、MINIMA 的 Average Mean Dist。
  - **剔除且不汇总**：学位论文一律用 "*"/"-"/"---"（[R16](#t-xie)、[R17](#t-wang)、[R20](#t-lv)）。
  - **照报巨大值**：[R18](#t-zhang) ch5。
- **已知坑**：
  - **失败剔除后，均值只反映成功子集**：APM 论文写明持续失败的配置不参与排名；SharedMod 的 LE 把失败记 0，会**拉低**平均误差（04 §5）。
  - OS-Eval 对 0 匹配的对不写结果，相当于**静默剔除**。
  - 惩罚值取多大是任意的（10 / 20 / 999），会直接改变均值。
  - 四种记法得到的数值**不可直接比较**（01 §11）。
  - 另外，"失败"本身的定义也不一致：可能指估计器无输出，也可能指误差超过门槛（[R10](#t-shmod)、[R29](#t-fire)）。

### 2.5 估计器
- **主流**：**评测器统一重新拟合**。
  - 仿射：[R4](#t-oseval) 3 px；[R6](#t-apm-sn9) 3 px；[R13](#t-matchanything) 3 px，conf 0.99999。
  - 单应：[R5](#t-soma1m) 1.5 px；[R21](#t-minima) 默认 3 px；[R10](#t-shmod)；[R12](#t-rrsi) DEGENSAC；[R27](#t-silk)。
  - 学位论文里明确写了共用估计器的只有吕宁（FSC 仿射）和张 ch4（RANSAC）。
- **分歧**：
  - **不统一**：[R1](#t-multires) 用各方法自带代码；GIM 直接抄基线论文的数字；谢、王没有说明。
  - **不用鲁棒估计器**：[R2](#t-srif) 评的是描述子；[R8](#t-gdros) 用 LSQ；[R9](#t-soma) 直接比较 flow；[R31](#t-raco) 用 DLT。
  - 变换模型：仿射（R4/R6/R13/R16/R20）与单应（R5/R12/R21/R25–R28）并存。
  - 阈值从 0.5 到 15 px 都有。
- **已知坑**：
  - **在测试集上扫 RANSAC 阈值**：glue-factory 的 `ransac_th=-1`；RaCo 按方法挑最优阈值，挑选用的数据集没有说明。
  - **代码与论文不一致**：rsim 的 CLI 默认是 homography、min_inliers 8；MINIMA 的 `--ransac_thres` 没有生效；GeoFormer 入口参数与函数默认值不同；RIPE 的 0.5 px 论文里没写；OS-Eval 只对指定方法名跑 RANSAC，而且注释写 10 次、代码是 20 次。
  - 估计器的影响**可能超过匹配器本身**：APM p.5–6 "Protocol sensitivity can exceed matcher differences"；RaCo 同一方法 AUC@1px 从 40.4 变为 44.7。

### 2.6 聚合
- **主流**：**全样本等权**，例如 [R13](#t-matchanything)、[R25](#t-roma)、[R26](#t-ripe)、[R27](#t-silk)。
- **分歧**：
  - **分组宏平均**：[R21](#t-minima) 按 7 个子集，光-SAR 只占 1/7 权重；[R29](#t-fire) mAUC；[R30](#t-caunsup) 按类别；[R2](#t-srif)、[R3](#t-rift) 论文按类别；[R5](#t-soma1m) 按数据集和分辨率。
  - **两级聚合**：[R6](#t-apm-sn9) 先对内取均值，再在成功对上平均并附 CI；[R4](#t-oseval) 对内平均 20 次 RANSAC；[R19](#t-zhang4) 每对跑 10 次。
  - **只逐对列表**：学位论文全部如此，SRIF 代码也是。
- **已知坑**：
  - 宏平均会放大小子集的权重（MINIMA 的光-SAR 只有 6 对）。
  - 代码与论文的聚合口径不一致：SOMA 的 batch-RMSE；APM 的 median 列其实是"每对中位数的均值"。
  - 多篇工作的聚合方式未写明（[R1](#t-multires)、[R5](#t-soma1m)）。

### 2.7 无监督方法的模型选择
- **用带 GT 的验证集选 best**：[RIPE++](#t-ripe) 按 IMW2020 / SCARED val 的 AUC@5° 保存 best，训练虽无标注，选 checkpoint 却用了标签（05 §3）。S2M2-SAR 在带 GT 的 val 上监控指标，是否用来选模型没有说明（05 §9）。
- **只记日志，保存 final**：RIPE（05 §2）。
- **无标注的代理验证**：SiLK 在 COCO 合成单应 val 上监控 `val.f1`；GeoFormer 用合成 `val_loss`（05 §4、§11）。
- **没有验证**：CA-Unsup 每 4000 步存一次，仓库里的 Val_List 没有被用到；吕 ch6 固定 48 epoch，val 只用来画 loss 曲线（03 §2.5）。
- **坑**：张 ch5 **在测试的 4 对图像上**按 NOCC 选 λ（03 §1.4）；glue-factory 和 RaCo 在测试集上挑阈值（见 2.5）。

---

## 3. 「定义评价协议与指标」（#2）的可选项

下面每个维度列出候选项和它们的依据，**不给推荐结论**。

| 维度 | 候选 | 依据（对照表行） |
|---|---|---|
| 指标 | A. 对级成功率 SR@多档 px，并报误差分布 | [R13](#t-matchanything)、[R8](#t-gdros)、[R12](#t-rrsi)、[R11](#t-tar) |
| | B. 误差 AUC@多档（1/3/5、3/5/10、5/10/20） | [R5](#t-soma1m)、[R12](#t-rrsi)、[R21](#t-minima)、[R26](#t-ripe) |
| | C. 平均/中位误差与失败率分列；如需与传统光-SAR 文献对照，可附 NCM 类诊断量 | [R6](#t-apm-sn9)、[R4](#t-oseval)；NCM 见 [R1](#t-multires)、[R2](#t-srif) |
| 误差对象 | A. 人工点对作独立检查点，用估计仿射映射后求距离 | [R1](#t-multires)、[R4](#t-oseval)、[R13](#t-matchanything)、[R6](#t-apm-sn9)、[R29](#t-fire) |
| | B. 由人工点拟合 GT 仿射，再在角点或网格上比较 GT 与估计 | 拟合 GT：[R3](#t-rift)（5 点）、[R16](#t-xie)（20 点）；角点/网格：[R12](#t-rrsi)、[R21](#t-minima)、[R7](#t-apm-srif)、[R14](#t-homo) |
| | C. 在已对齐的对上加预存的合成仿射扰动，用解析 GT 算角点或稠密误差 | [R12](#t-rrsi)、[R8](#t-gdros)、[R5](#t-soma1m)；反例：[R9](#t-soma)（不设种子）、[R10](#t-shmod)（恒等） |
| 成功阈值 | A. 固定一档或少数几档 px 阈值 | [R13](#t-matchanything) 5/10/20、[R11](#t-tar) 1/3/5、[R8](#t-gdros) 1–5 |
| | B. 用 AUC 覆盖一段阈值区间；下限可参考人工标注噪声（约 1 px） | [R26](#t-ripe)、[R27](#t-silk)、[R29](#t-fire)；噪声：[R30](#t-caunsup)、[R1](#t-multires) |
| | C. 在误差阈值之外再加 NCM 下限或粗失败门槛 | [R1](#t-multires)、[R2](#t-srif)、[R10](#t-shmod)、[R29](#t-fire) |
| 失败计入 | A. 固定惩罚或截断值计入均值 | [R12](#t-rrsi) 10、[R2](#t-srif) 20、[R14](#t-homo) 5 |
| | B. 记 ∞，只报 SR/AUC 这类对 ∞ 稳健的量 | [R21](#t-minima)、[R26](#t-ripe)、[R23](#t-xoftr) |
| | C. 从均值中剔除，单独报失败率 | [R6](#t-apm-sn9)、[R10](#t-shmod) |
| 估计器 | A. 评测器统一重跑仿射 RANSAC，阈值和迭代数事先固定 | [R4](#t-oseval)、[R6](#t-apm-sn9)、[R13](#t-matchanything) |
| | B. 同 A，但阈值在 Val 上选定，Test 上冻结（对照：测试集扫阈值的做法） | 反例：[R26](#t-ripe)、[R31](#t-raco)；敏感性证据：[R6](#t-apm-sn9)（01 §9.1） |
| | C. 允许各方法自带估计器，或直接比较稠密输出 | [R1](#t-multires)、[R9](#t-soma)、[R8](#t-gdros) |
| 聚合 | A. 全样本等权 | [R13](#t-matchanything)、[R25](#t-roma)、[R27](#t-silk) |
| | B. 按场景或子集宏平均，同时报各子集 | [R21](#t-minima)、[R29](#t-fire)、[R5](#t-soma1m) |
| | C. 两级聚合（对内 → 跨对），附 bootstrap CI | [R6](#t-apm-sn9)、[R4](#t-oseval) |
| 模型选择 | A. 在有标注的 Val 上用与 Test 相同的指标选 checkpoint | [R26](#t-ripe)（RIPE++，05 §3） |
| | B. 无标注代理：合成扰动 val 上的 loss 或 f1 | [R27](#t-silk)、[R28](#t-geoformer) |
| | C. 固定 epoch 或 final，不做选择 | RIPE（05 §2）、吕 ch6（03 §2.5）、[R30](#t-caunsup) |

---

## 4. 分文件之间的矛盾与笔误（按原样并列，未裁决）

1. **SRIF 各方法是否"共用估计器"**：01 §2 说评测不跑 RANSAC，SRIF/LNIFT 的 exe 内部不可审，结论是"各方法之间**不共用**估计器"。04 §12 的对照表却写"是（脚本逻辑相同，阈值差 1）"。两者都确认评测阶段没有 RANSAC，分歧在于 exe 内部未知时怎么定性。
2. **SRIF 的 RES 列**：01 §2 写的是 `[time, rmse, NCM, 原始匹配数]`，引用 demo_RIFT.m#L76-L77；04 §2 写的是 `RES=[time rmse NCM]`，引用 demo_SRIF.m。可能是不同脚本的列不一样，没有核实。
3. **SRIF 数据集规模**：01 §8 和 04 §2 都说是 6 模态 × 200 = 1200 对，01 §9.2 说 APM 用了 "SRIF（600 对）"。分文件没有说明 600 对是怎么取的子集。
4. **APM 论文与代码**：01 §9 列了 4 处以上不一致（分段模型、resize px、CLI 默认值、max side、SRIF 在论文里是角点而代码里是网格）。04 §13 末尾把 APM 列为"范围外线索（未核实）"，这一条已经被 01 §9 覆盖。
5. **单一工作内部的不一致**（分文件已标注）：综述对 OSEval GT 的描述（角点）与 README（金属杆）不同（01 §1）；吕宁论文里 FSC 有两种全称（03 §2.2）；王丽娜 Ch4 的 CMR 填法与式 2.15 不符，引文 [119] 疑似标错（02 §2.3）；GeoFormer README 的复现值与论文不同（05 §11）；RIPE 论文写 MegaDepth 长边 1200，配置里是 1600（05 §2）。
6. 笔误：02 开头写「博士/硕士论文」，学位类型没有确定；05 §8 XCP-Match 的旋转范围写作「[−10,−10]」，这是论文原文的笔误，分文件已注明。其他明显笔误未发现。

---

## 5. 可追溯性

- **链路**：本 README 中的结论 → 对照表行号（R1–R31，可以锚点链接） → 分文件小节（「01 §2」表示 [01-survey-benchmarks.md](01-survey-benchmarks.md) 的第 2 节） → 分文件中的 commit 锁定 permalink（代码），或 PDF 页码（论文）。
- **PDF 页码**：页码规则以各分文件开头的约定为准。01 和 04/05 用 PDF 物理页。02 的谢志华、王丽娜分别是「PDF = 印刷页 + 17 / +16」。03 的两篇都是「印 = PDF − 17」。PDF 在本地 `D:\科研\光SAR配准\论文\` 下，不进 git。
- **源码**：所有 permalink 都锁定在 [sources.md](sources.md) 列出的 commit 上。本地克隆在仓库外的 `D:\Code\refs\`，可以用 sources.md 末尾的脚本恢复。
- 本 README **没有新增事实**。所有数字和判断都来自 01–05；标有「推断」的地方沿用分文件的原标注。
