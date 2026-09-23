# 光-SAR 配准评价指标：综述与 benchmark 的源码级调研

> 范围：MultiResSAR 综述（首要来源）、SOMA-1M、ArePretrainedMatchers（CVPRW 2026），以及综述实际对比方法和相关数据集里**能拿到的评测代码**。
> 原则：以 PDF 和源码为准，没有参考同目录 AI 生成的 .md 总结。论文和代码不一致时按代码记录，并标注【不一致】。找不到依据的地方写"未找到"，不做推测；标注【推断】的是根据代码或数值推出来的结论，不是原文。
> 调研日期：2026-09-23。所有 permalink 都锁定到下面「源码清单」里的 commit。

PDF 根目录记为 `P = D:\科研\光SAR配准\论文\03_评测基准与数据集\`。页码是 PDF 物理页码。

---

## 1. MultiResSAR 综述（Zhang et al., WHU）

**出处**：`P\Review_MultiResSAR_光SAR配准综述.pdf`，共 48 页。评价指标在 p.31–32，实验设置在 p.30–31，Table 6（方法与代码地址）在 p.31，Table 7（结果）在 p.34–35，数据集与真值在 p.25–29。

**源码**：
- 数据集仓库 `betterlll/Multi-Resolution-SAR-dataset-` @ `6f664903`，只有一个 [README](https://github.com/betterlll/Multi-Resolution-SAR-dataset-/blob/6f664903bc9662a66ac7c2c88a39f5e2d9da1ec4/README.md)，内容是百度网盘链接、"目前公开 1,100 对"。**没有评测代码。**
- 综述里的 16 种方法都是"用作者提供的代码和推荐参数"跑的（p.30–31 原文："the experiments used the code provided by the authors, with the recommended parameter settings"），综述自己的指标计算脚本**没有公开（未找到）**。
- 本地路径：`D:\Code\refs\Multi-Resolution-SAR-dataset-`

### 1.1 综述归纳的指标体系（p.31–32）
- 评价分两类：定性（主观目视）和定量。本文选了 4 个定量指标：**SR、NCM、RMSE、TM**（p.31）。
- **SR（成功率）**，公式 (1)：`I(p_i) = 1 if NCM(p_i) ≥ N_min else 0`，`SR = (1/M)·Σ I(p_i)·100%`，其中 **N_min = 20**，M 是数据集里的图像对总数（p.31）。
- **NCM**：原文是 "the number of image pairs with more than 20 matched corresponding points, while excluding image pairs with RMSE greater than 10 pixels, serving as the count of correct matches"（p.32）。【不一致/歧义】字面上是在数"图像对"，但 Table 7 的 NCM 在 38–590 之间，单位写的是 "number of points"（p.34），所以实际更像是"每对图的正确匹配点数"。从文字能读出的门限是 **>20 点**且 **RMSE ≤ 10 px**，但"单个匹配点算不算正确"用的像素阈值**未找到**。
- **RMSE**，公式 (2)：`RMSE = sqrt( (1/N) Σ_i [(x'_i − x''_i)² + (y'_i − y''_i)²] )`。N 是**真值点个数**，(x'_i, y'_i) 是第 i 个真值点，(x''_i, y''_i) 是它"经过相应匹配变换后"的坐标（p.32）。也就是说，误差是在**人工真值控制点**上，用**估计出来的变换**去映射后计算的。
- **TM**：从输入到输出匹配结果的总耗时，单位是秒（p.32）。

### 1.2 六项要点
| 项 | 结论 | 出处 |
|---|---|---|
| 报了哪些指标 | SR(%)、NCM(点)、RMSE(px)、TM(s)，Table 7 给的是"average results" | 论文 p.31–34 |
| 误差算在什么上 | 人工选取的真值控制点，用估计变换映射后与对应点比较。真值的来源是：先自动配准，再由测绘专业人员挑选"几何相似、受噪声影响最小"的角点等控制点，误差控制在 1 px 以内 | 论文 p.28（§4.3）、p.32 |
| 成功判定与阈值 | NCM ≥ 20 记为成功；NCM 定义里还要求 RMSE ≤ 10 px。只有一档阈值 | 论文 p.31–32 |
| 失败样本怎么计入 | SR 的分母是全部 M 对，失败计 0。RMSE、NCM 的"平均"是只在成功对上算，还是全部对都参与，**未找到**。【推断】Table 7 里 SR 很低的方法（如 LNIFT 的 SR 只有 0.41%）RMSE 仍然只有 4.71 px，说明 RMSE 很可能只在成功对上平均 | 论文 p.34–35 Table 7 |
| 鲁棒估计器 | **未统一**：各方法用作者自带代码和推荐参数（p.30–31）。综述**没有**说明 RANSAC 变体、变换模型和内点阈值（未找到） | 论文 p.30–31 |
| 聚合方式 | Table 7 是全数据集的"average results"（p.34），没有写是按样本平均还是按场景平均（未找到）。另外，850 对 Umbra 0.16 m 的子集只做了定性展示（Fig.10，p.37） | 论文 p.34、p.37 |

补充说明：
- Table 7 的实际数值（p.34–35，按渲染图核对过）：XoFTR SR 40.58% / RMSE 3.03 / NCM 244.26；RoMa 35.26% / 3.15 / 589.70；RIFT 66.51% / 3.58 / 108.40；LNIFT 0.41% / 4.71 / 61.53；等等。
- 【推断】综述作者（李加元为共同作者）同组的 SRIF 仓库，是用"GT 下误差 < 3 px"来判定正确点，并且 RMSE 只在正确点上算（见 §2）。如果综述沿用这套做法，RMSE 应当 ≤ 3 px，但 Table 7 里的 RMSE 全部在 3.03–5.35 px，**和 SRIF 仓库的 3 px 协议对不上**。综述到底用了什么阈值无法确认。
- 【不一致】综述把 OSEval 描述为"用光学/SAR 图像中的**角点**作为真值"（p.26），但 OS-Eval 仓库 README 写的是用**金属杆（meta poles）**作为真值。见 §4。

---

## 2. SRIF（Li, Hu, Zhang, ISPRS 2023）：综述对比方法之一，同时也是数据集和对比脚本

**出处**：综述 Table 6（p.31）。SRIF 数据集也被 SOMA-1M 和 ArePretrainedMatchers 用作测试集。
**源码**：https://github.com/LJY-RS/SRIF @ `88881a3a8789d0bed6a8df91b943e02f2593a0cd`，本地 `D:\Code\refs\SRIF`（本地已有，直接复用）。评测入口是 `demo_*.m`，每个方法一个脚本（SRIF/RIFT/LNIFT/3MRS/CoFSM/MSHLMO/OSSIFT/SIFT）。

| 项 | 结论（以代码为准） | permalink |
|---|---|---|
| 指标 | 每对图保存 `[time, rmse, NCM(=正确点数), 原始匹配数]`，只输出 `RES` 矩阵，**没有 SR 汇总** | [demo_RIFT.m#L76-L77](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_RIFT.m#L76-L77)、[demo_SRIF.m#L56-L61](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_SRIF.m#L56-L61) |
| 误差算在什么上 | 真值是 2×3 仿射 `gt_i.txt`（[样例](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/dataset/Optical-SAR/gt_1.txt)）。做法是：用 GT 变换把**方法输出的匹配点**从图 1 映射到图 2，和匹配点在图 2 的位置算欧氏距离，**< 3 px 的算正确匹配**；去重后，**RMSE 只在这些正确匹配上算**。所以它衡量的是匹配点的精度，不是估计变换的精度 | [demo_SRIF.m#L34-L54](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_SRIF.m#L34-L54) |
| 成功判定与阈值 | 单点：GT 误差 < 3 px。单对图：正确点 < 10（部分脚本是 ≤ 10）就判失败。【不一致】各脚本的边界写法不一样：SRIF/LNIFT/3MRS/CoFSM/SIFT 用 `<10`，RIFT/MSHLMO/OSSIFT 用 `<=10` | [demo_SRIF.m#L50-L54](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_SRIF.m#L50-L54)、[demo_RIFT.m#L69-L73](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_RIFT.m#L69-L73) |
| 失败怎么计入 | **RMSE 直接记为固定惩罚值 20 px**，NCM 保留实际计数 | 同上 |
| 鲁棒估计器 | **评测环节不跑 RANSAC**，直接用 GT 去筛原始匹配。RIFT 脚本用的是 `matchFeatures(...,'MaxRatio',1,'MatchThreshold',100)` 得到的原始最近邻匹配。SRIF 和 LNIFT 调用的是 exe（例如 SRIF.exe 参数中有 `5000`），exe 内部有没有做外点剔除**未找到**（二进制，不可审）。各方法之间**不共用估计器** | [demo_RIFT.m#L43-L46](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_RIFT.m#L43-L46)、[demo_SRIF.m#L24](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_SRIF.m#L24) |
| 聚合方式 | 代码只逐对保存 `RES_*.mat`，均值/中位数**未找到**。另外所有脚本都写死了 `addpath dataset\Optical-Optical\`，换模态需要手动改 | [demo_SRIF.m#L2](https://github.com/LJY-RS/SRIF/blob/88881a3a8789d0bed6a8df91b943e02f2593a0cd/demo_SRIF.m#L2) |

---

## 3. RIFT（Li et al., TIP 2020）：综述 SR 最高的传统方法

**源码**：https://github.com/LJY-RS/RIFT-multimodal-image-matching @ `7ea830e2f13cc3c226f975fe9e98b7666a8f26fb`，本地 `D:\Code\refs\RIFT-multimodal-image-matching`（已有，复用）。入口是 `RIFT_demo.m`，只处理单对图，**没有真值评测**。

| 项 | 结论 | permalink |
|---|---|---|
| 指标 | demo 不算任何 GT 指标。`FSC` 会返回内点上最小二乘拟合的**自洽残差 rmse**，这个值没有用到 GT | [FSC.m#L96-L99](https://github.com/LJY-RS/RIFT-multimodal-image-matching/blob/7ea830e2f13cc3c226f975fe9e98b7666a8f26fb/FSC.m#L96-L99) |
| 误差算在什么上 | 没有 GT。内点的判定依据是估计出的仿射模型 | [RIFT_demo.m#L42-L49](https://github.com/LJY-RS/RIFT-multimodal-image-matching/blob/7ea830e2f13cc3c226f975fe9e98b7666a8f26fb/RIFT_demo.m#L42-L49) |
| 成功判定 | 未找到 | — |
| 失败怎么计入 | 未找到 | — |
| 鲁棒估计器 | **FSC**（RANSAC 的变体），模型是 **affine**，FSC 内点阈值 **2 px**，然后再用 **< 3 px** 筛一遍内点。迭代次数取 `min(C(M,3), 10000)` | [RIFT_demo.m#L42-L47](https://github.com/LJY-RS/RIFT-multimodal-image-matching/blob/7ea830e2f13cc3c226f975fe9e98b7666a8f26fb/RIFT_demo.m#L42-L47)、[FSC.m#L1-L18](https://github.com/LJY-RS/RIFT-multimodal-image-matching/blob/7ea830e2f13cc3c226f975fe9e98b7666a8f26fb/FSC.m#L1-L18)、[FSC.m#L77](https://github.com/LJY-RS/RIFT-multimodal-image-matching/blob/7ea830e2f13cc3c226f975fe9e98b7666a8f26fb/FSC.m#L77) |
| 聚合方式 | 未找到 | — |

---

## 4. OS-Eval（Xiang et al., TGRS 2023）：综述列出的亚米级光-SAR 评测数据集

**出处**：综述 §4.1 第 8 项（p.26）。
**源码**：https://github.com/xym2009/OS-Eval @ `4df7d5a780d4c2d5b913ac7b7efa83ef44c43850`，本地 `D:\Code\refs\OS-Eval`。评测入口是 `CalME/src/Main.cpp` 和 `CalME/src/matchFunc.cpp`（C++，依赖 GDAL 和 OpenCV），参数在 `CalME/config.txt`。匹配点生成的示例是 `TestCode/RIFT_TEST_RE.m`。

| 项 | 结论 | permalink |
|---|---|---|
| 指标 | **ME**（Mean Error，**GT 检查点的平均欧氏误差，不是 RMSE**）、**CMR**（RANSAC 内点数 / 全部匹配数）。对非指定方法报的是 ME 和 NCM | [matchFunc.cpp#L362](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L362)、[#L387](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L387)、[#L394](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L394) |
| 误差算在什么上 | **独立的 GT 检查点**（`*-GT.txt`，README 说是金属杆）。先在 RANSAC 内点上用最小二乘拟合仿射 `T1`，把光学侧的 GT 点经 `T1` 变换，再经 RPC 和 DEM 投影到 SAR 图像坐标，和 SAR 侧的 GT 点求距离后取平均 | [matchFunc.cpp#L685-L729](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L685-L729) |
| 成功判定 | 没有显式的成功率。只有一个门槛：匹配数 ≤ 20 时直接判失败 | [matchFunc.cpp#L366-L367](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L366-L367) |
| 失败怎么计入 | 匹配数 ≤ 20 时记 **ME = 999（最大值惩罚）、CMR = 0**。匹配文件里 0 个匹配时直接 `return`，**不写结果**，相当于被剔除 | [matchFunc.cpp#L313-L316](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L313-L316)、[#L366-L367](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L366-L367) |
| 鲁棒估计器 | 评测器**统一**重新跑 RANSAC：模型 affine，阈值 `threshold:3`（来自 config），迭代数 `min(C(N,3),10000)`，每对**重复 20 次取平均**（注释写的是 10 次，代码是 `rN=20`）。【不一致】只有 CFOG/OSMNet/RIFT/MatchosNet 这些方法名会走 RANSAC 分支，其他方法名**不做 RANSAC**，直接用全部匹配拟合 | [config.txt#L2](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/config.txt#L2)、[matchFunc.cpp#L137-L155](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L137-L155)、[#L343-L395](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/CalME/src/matchFunc.cpp#L343-L395) |
| 聚合方式 | 同一对图内：GT 点取平均，20 次 RANSAC 再取平均。跨对：只按对写日志，数据集层面的聚合代码**未找到** | 同上 |

补充：示例 `RIFT_TEST_RE.m` 写出的是**未剔除外点的原始最近邻匹配**，外点剔除交给评测器统一做（[RIFT_TEST_RE.m#L69-L77](https://github.com/xym2009/OS-Eval/blob/4df7d5a780d4c2d5b913ac7b7efa83ef44c43850/TestCode/RIFT_TEST_RE.m#L69-L77)）。

---

## 5. XoFTR（Tuzcuoğlu et al., CVPRW 2024）：综述 SR 最高的深度方法

**源码**：https://github.com/OnderT/XoFTR @ `e0fbea431b30be9742effbf5577c90aa8eb938f9`，本地 `D:\Code\refs\XoFTR`（已有，复用）。入口是 `test_relative_pose.py`（METU-VisTIR 可见光-热红外数据）。这是**相对位姿**评测，和光-SAR 仿射配准不是同一类任务，这里只记录它的协议结构。

| 项 | 结论 | permalink |
|---|---|---|
| 指标 | 位姿 AUC@5/10/20°，位姿误差取 `max(R_err, t_err)` | [test_relative_pose.py#L211-L213](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/test_relative_pose.py#L211-L213) |
| 误差算在什么上 | GT 相对位姿 | [metrics.py#L119-L152](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/src/utils/metrics.py#L119-L152) |
| 成功判定 | AUC 本身就是多档阈值（5/10/20°）。【不一致】`error_auc` 在函数内部把 `thresholds` **硬编码覆盖成 [5,10,20]**，传进来的参数不起作用 | [metrics.py#L157-L174](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/src/utils/metrics.py#L157-L174) |
| 失败怎么计入 | 匹配点 < 5 或估计出的 E 为 None 时记 **∞**，计入 AUC 的分母（相当于拉低 recall） | [metrics.py#L90-L105](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/src/utils/metrics.py#L90-L105)、[test_relative_pose.py#L176-L182](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/test_relative_pose.py#L176-L182) |
| 鲁棒估计器 | `cv2.findEssentialMat` RANSAC，置信度 0.99999。VisTIR 测试的阈值是 **1.5 px**，训练 config 里是 0.5 px | [test_relative_pose.py#L222](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/test_relative_pose.py#L222)、[default.py#L150-L151](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/src/config/default.py#L150-L151) |
| 聚合方式 | 先在每个场景文件（npz）内算 AUC，再对同名场景取平均 | [test_relative_pose.py#L115-L131](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/test_relative_pose.py#L115-L131) |

---

## 6. RoMa / DKM（Edstedt et al.）：综述对比方法中与单应和角点评测最接近的实现

**源码**：
- RoMa：https://github.com/Parskatt/RoMa @ `77f8d68803526dcddfd9b7a46bc76125bdc25f15`，本地 `D:\Code\refs\RoMa`，入口 `romatch/benchmarks/hpatches_sequences_homog_benchmark.py`
- DKM：https://github.com/Parskatt/DKM @ `ef57565db52684e661052ba82eb361329c63af3d`，本地 `D:\Code\refs\DKM`，入口 `dkm/benchmarks/hpatches_sequences_homog_benchmark.py`

两者都是在 HPatches 自然图像上的评测，这里只作为协议写法的参考。

| 项 | 结论 | permalink |
|---|---|---|
| 指标 | 单应**角点误差的 AUC**，阈值 1..10 px，报告 @3/@5/@10 | [RoMa hpatches#L106-L113](https://github.com/Parskatt/RoMa/blob/77f8d68803526dcddfd9b7a46bc76125bdc25f15/romatch/benchmarks/hpatches_sequences_homog_benchmark.py#L106-L113)、[pose_auc](https://github.com/Parskatt/RoMa/blob/77f8d68803526dcddfd9b7a46bc76125bdc25f15/romatch/utils/utils.py#L135-L147) |
| 误差算在什么上 | 图像的 **4 个角点**，分别经 GT 单应和估计单应映射后求平均距离。误差按 `min(w2,h2)/480` **归一化到 480 px 短边** | [RoMa hpatches#L92-L104](https://github.com/Parskatt/RoMa/blob/77f8d68803526dcddfd9b7a46bc76125bdc25f15/romatch/benchmarks/hpatches_sequences_homog_benchmark.py#L92-L104) |
| 成功判定 | 通过 AUC 体现多档阈值，没有单独的成功率 | 同上 |
| 失败怎么计入 | 估计失败时把 `H_pred` 设成**全零矩阵并令 [2,2]=1**。这个矩阵会把所有点映射到 (0,0)，于是得到一个很大的误差并**计入 AUC**，不会被剔除 | [RoMa hpatches#L79-L91](https://github.com/Parskatt/RoMa/blob/77f8d68803526dcddfd9b7a46bc76125bdc25f15/romatch/benchmarks/hpatches_sequences_homog_benchmark.py#L79-L91) |
| 鲁棒估计器 | 先用 `model.sample` 采 5000 个匹配，再跑 `cv2.findHomography` RANSAC，置信度 0.99999，阈值 `3·min(w2,h2)/480` | [RoMa hpatches#L75-L86](https://github.com/Parskatt/RoMa/blob/77f8d68803526dcddfd9b7a46bc76125bdc25f15/romatch/benchmarks/hpatches_sequences_homog_benchmark.py#L75-L86) |
| 聚合方式 | 所有图像对的误差放在一起算一个 AUC。【不一致】RoMa 定义了 `ignore_seqs`，但**没有使用**（[#L18-L30](https://github.com/Parskatt/RoMa/blob/77f8d68803526dcddfd9b7a46bc76125bdc25f15/romatch/benchmarks/hpatches_sequences_homog_benchmark.py#L18-L30)）；DKM 确实跳过了这些序列（[DKM#L62](https://github.com/Parskatt/DKM/blob/ef57565db52684e661052ba82eb361329c63af3d/dkm/benchmarks/hpatches_sequences_homog_benchmark.py#L62)） | — |

---

## 7. SAR-SIFT、LNIFT、MOSS 及综述中其余方法

- **SAR-SIFT**（综述引用的是 `yishiliuhuasheng/sar_sift` @ `6601368b`，本地 `D:\Code\refs\sar_sift`）：**没有 GT 评测**。自带的 RANSAC 是 affine 模型、**800 次迭代**、阈值 `error_threshold=1`。【注意】这个阈值比较的是 `sqrt(Σ残差²/2)`，也就是逐轴 RMS，**不是欧氏距离**。它输出的 rmse 是内点上最小二乘拟合的自洽残差。见 [ransac.py#L13-L14](https://github.com/yishiliuhuasheng/sar_sift/blob/6601368b5bf9cff418068721e2f2de1aa3ecc81d/ransac.py#L13-L14)、[#L49-L50](https://github.com/yishiliuhuasheng/sar_sift/blob/6601368b5bf9cff418068721e2f2de1aa3ecc81d/ransac.py#L49-L50)、[#L85-L86](https://github.com/yishiliuhuasheng/sar_sift/blob/6601368b5bf9cff418068721e2f2de1aa3ecc81d/ransac.py#L85-L86)、[sar_sift.py#L40](https://github.com/yishiliuhuasheng/sar_sift/blob/6601368b5bf9cff418068721e2f2de1aa3ecc81d/sar_sift.py#L40)。
- **LNIFT**（综述引用的 `arunsahu159/LNIFT-...` @ `1ffa3528`，本地 `D:\Code\refs\LNIFT`）：README 自称是 "my own implementation"，也就是**非官方复现**，只有一个 notebook，**没有评测代码**。官方的 LNIFT 以 exe 形式放在 SRIF 仓库里（见 §2）。
- **MOSS**（`betterlll/MOSS_data` @ `25c1cfcd`，本地 `D:\Code\refs\MOSS_data`）：只有数据，48 对图，每对附一个 GT 检查点 txt，格式是首行点数、之后每行 `x1 y1 x2 y2`（[1.txt](https://github.com/betterlll/MOSS_data/blob/25c1cfcd1f0afc34a78af27dccd2637adf139a6e/moss_datasets/1.txt)）。**没有评测代码。**
- **HOWP**：地址是项目网页，不在 GitHub 上。**ASS**：地址在 gitee。这两个都**没有获取**。
- **SuperGlue、LightGlue、LoFTR、Efficient LoFTR、SGM-Net、XFeat**：官方评测都是 MegaDepth/ScanNet 的位姿 AUC（自然图像），和本任务无关，**没有克隆和审查**（未找到对综述协议有用的信息）。

---

## 8. SOMA-1M（Wu et al., 2026 预印本，WHU）

**出处**：`P\SOMA-1M_2026.pdf`，共 21 页。匹配评测协议在 p.11–12（§4.1.1–4.1.2），Table 2 在 p.11，分辨率子集 Table 6 在 p.18，数据集构建时的配准阈值在 p.8–9。
**源码**：https://github.com/PeihaoWu/SOMA-1M @ `b86137c92b225402a4706eaee4bb7a61f11baabf`，本地 `D:\Code\refs\SOMA-1M`，**只有 README**（提供测试集下载链接）。论文说"按 MapGlue 的标准协议"，但 https://github.com/PeihaoWu/MapGlue @ `81b06ec0f95c6f635aa6486a66147fdf974f6e2c`（本地 `D:\Code\refs\MapGlue`）**也只有 README**。**评测代码未找到**，下表全部依据论文。
（注：`D:\Code\refs\SOMA` 是另一个仓库 `traslauc/SOMA`，和这篇论文无关。）

| 项 | 结论 | 出处 |
|---|---|---|
| 指标 | **角点误差 AUC@5/10/20 px**（百分比） | 论文 p.12 |
| 误差算在什么上 | 测试图对本身是**已对齐**的。评测时对左图施加随机单应扰动，扰动参数是：旋转 [−108°, 108°]、平移 ±30% 宽/高、缩放 0.7–1.3，随机种子由样本 ID 决定。GT 取扰动矩阵的逆，`H_gt = T⁻¹`。误差在**角点**上算。具体用哪几个角点、是否归一化，**未找到** | 论文 p.11 式(1) |
| 成功判定 | 通过 AUC 体现 5/10/20 px 三档，没有单独的成功率 | 论文 p.12 |
| 失败怎么计入 | **未找到** | — |
| 鲁棒估计器 | 所有方法**统一用 RANSAC 估单应**，重投影阈值 **1.5 px**，最多 **10,000** 次迭代，置信度 **0.9999**。稀疏方法的关键点上限统一为 2,048，输入统一 resize 到 512×512（p.12 写在训练设置里，测试时是否也 resize 没有明说） | 论文 p.11–12 |
| 聚合方式 | 按测试集分别报告：SOMA-Test、OSdataset（662 对 GF-3）、SRIF（6 模态 × 200 对）。SOMA-Test 另外按 Low/Mid/High 分辨率子集报告（Table 6）。是按样本还是按场景平均，**未找到** | 论文 p.11、p.18 |

补充：构建数据集时的精配准，全局阶段用 RANSAC 10 px，局部阶段用 1.5 px 求局部单应（p.8–9）。这决定了 GT 本身的精度上限，但不属于评测协议。

---

## 9. ArePretrainedMatchersGoodEnough（Corley, Stoken, Berton, CVPRW 2026）

**出处**：`P\ArePretrainedMatchersGoodEnough_CVPRW2026.pdf`（arXiv:2604.10217v4），共 11 页。协议在 p.3–4（§3.2–3.3），Table 2 在 p.5，扫参在 p.5–6，代码链接在 p.5。
**源码**：https://github.com/isaaccorley/rsim @ `9950822cd7c9eaa23c233cc94099c1882ec6d2cc`，本地 `D:\Code\refs\rsim`。匹配器统一通过 https://github.com/gmberton/vismatch @ `9d49b892ed21625cee00bc7797a45d352bf659a7`（本地 `D:\Code\refs\vismatch`）调用。评测入口是 `src/rsim/spacenet9_matcher_benchmark.py`、`src/rsim/srif_matcher_benchmark.py`、`src/rsim/sarptical_pair_eval.py`，复现脚本在 `scripts/`。

下文 permalink 前缀 `R = https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/`。

### 9.1 SpaceNet9（主实验，3 对大场景，人工 tie point）
| 项 | 结论（以代码为准） | permalink |
|---|---|---|
| 指标 | 每对图算 `mean_error_px`、`median_error_px`、`success_at_{1,2,3,5,10}px`；汇总层面再算 `failure_rate`，外加 bootstrap 95% CI。论文只报了 MeanErr、S@5、S@10、Fail | [R/src/rsim/spacenet9_matcher_benchmark.py#L648-L680](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L648-L680) |
| 误差算在什么上 | **人工 tie point**：把 SAR 侧 tie point 经预测变换映射到光学侧，和光学侧 tie point 求欧氏距离。【不一致 1】论文写的是"把估计变换作用在 tie point 上"，但 tiled 模式下代码**并没有一个全局变换**：每个 tile 各拟合一个模型，每个 tie point 取覆盖它的 tile 模型，按 `inliers/(1+到 tile 中心距离)` 加权后混合，实际上是**分段模型**。【不一致 2】误差的"px"是 **resize 之后**的像素（tie point 坐标乘了 `max_side` 的缩放系数），不是原始分辨率的像素 | [#L893-L927](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L893-L927)、[#L1105-L1114](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L1105-L1114) |
| 成功判定与阈值 | S@τ 是**一对图内误差 ≤ τ 的 tie point 所占比例**，τ ∈ {1,2,3,5,10}（论文只报 5 和 10）。"good enough" 的定义是 SpaceNet9 上平均误差 < 8 px（论文 p.2） | [#L648-L652](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L648-L652) |
| 失败怎么计入 | 以下情况把状态记为非 `ok`，**所有指标置为 NaN**：没有合格 tile（`no_tiles`）、投影出现非有限值、平均误差 > 1e6（`unstable_homography`）、异常、模型初始化失败。汇总时 pandas 的 `mean` **跳过 NaN**，所以 MeanErr 和 S@k **只在成功对上平均**，失败只体现在单独的 `failure_rate` 里。论文也说 "Configurations with persistent full failure are excluded from the ranking"（p.5） | [#L597-L615](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L597-L615)、[#L966-L975](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L966-L975)、[#L1196-L1227](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L1196-L1227) |
| 鲁棒估计器 | rsim 丢掉 vismatch 内部的 RANSAC，拿**原始匹配 `matched_kpts`** 统一重新拟合，所有匹配器**共用同一个估计器**：`cv2.estimateAffine2D` 或 `cv2.findHomography`，方法是 `cv2.RANSAC`，只设置了 `ransacReprojThreshold`，迭代数和置信度用 OpenCV 默认值（代码里没有显式设置）。主协议是 affine、3.0 px、每个 tile 至少 4 个内点（不足则丢弃该 tile）。【不一致 3】CLI 默认值和论文主协议不同：默认是 homography、tile 1024、min_inliers 8，论文数值要靠脚本传参才能复现。【不一致 4】论文 §5.2 说 max side 扫的是 {1024, 1536}，脚本里写的是 `MAX_SIDES = [1024, 2048]` | [#L531-L572](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L531-L572)、[#L851-L852](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L851-L852)、[#L98-L113](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L98-L113)、[R/scripts/run_protocol_sweep_top_matchers.py#L33-L36](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/scripts/run_protocol_sweep_top_matchers.py#L33-L36)、[R/scripts/run_extended_transfer_ablations.py#L44-L49](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/scripts/run_extended_transfer_ablations.py#L44-L49) |
| 聚合方式 | 先在一对图内对 tie point 取 mean 和 median，再按 `(matcher, normalization, mode, geometry)` 分组，对**成功的图对**取平均。median 列其实是"每对中位数的均值"。CI 来自对成功对的 bootstrap | [#L1196-L1227](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L1196-L1227)、[#L249-L273](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/spacenet9_matcher_benchmark.py#L249-L273) |

阈值扫描用的是 {0.5,1,2,3,5,8,10,15,20} px（[run_extended_transfer_ablations.py#L82](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/scripts/run_extended_transfer_ablations.py#L82)），论文 p.6 的结论是 "Protocol sensitivity can exceed matcher differences"。

### 9.2 SRIF（600 对，GT 是 2×3 仿射）
| 项 | 结论 | permalink |
|---|---|---|
| 指标 | 和 SpaceNet9 相同的一组列，另外按 `subdataset` 分组 | [R/src/rsim/srif_matcher_benchmark.py#L367-L402](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/srif_matcher_benchmark.py#L367-L402) |
| 误差算在什么上 | 【不一致】论文 p.4 说"把估计变换作用于**四个角点**，报告平均角点重投影误差"，但代码是用 GT 仿射在图 1 上生成 **20×20 网格点**，只保留落在图 2 内的点作为伪 tie point（`--srif-grid-size 20`）。代码里**没有角点计算** | [#L171-L204](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/srif_matcher_benchmark.py#L171-L204)、[#L102-L107](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/srif_matcher_benchmark.py#L102-L107) |
| 成功判定、失败、估计器、聚合 | 复用 SpaceNet9 的 `evaluate_pair`、`empty_metric_row` 和 NaN 跳过逻辑。CLI 默认是 affine、3 px、tiled 模式、tile 1024、min_inliers 8。【不一致】论文说 SRIF "processed as single images without tiling"（p.4），而代码默认是 `tiled`；不过 SRIF 图小于 tile 尺寸时只会切出一块，效果接近单图，但 min_inliers 仍然生效。复现 SRIF 的脚本**未找到**（`scripts/` 里没有调用它的地方） | [#L56-L95](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/srif_matcher_benchmark.py#L56-L95) |

### 9.3 SARptical（40 个查询、525 个候选对，检索任务）
- 论文 p.3 说"affine RANSAC（3 px），按**内点数**排序"。【不一致】代码的排序分数是**内点比例** `num_inliers/num_matches`。另外，vismatch 路径下的 `num_inliers` 来自 vismatch 内部的 **USAC_MAGSAC 单应**（3 px、2000 次迭代、置信度 0.995），不是仿射模型。只有 Kornia 基线走的是 affine 3 px。见 [R/src/rsim/sarptical_pair_eval.py#L337-L347](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/sarptical_pair_eval.py#L337-L347)、[#L378-L381](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/sarptical_pair_eval.py#L378-L381)、[vismatch base_matcher.py#L39-L41](https://github.com/gmberton/vismatch/blob/9d49b892ed21625cee00bc7797a45d352bf659a7/vismatch/base_matcher.py#L39-L41)、[#L97-L107](https://github.com/gmberton/vismatch/blob/9d49b892ed21625cee00bc7797a45d352bf659a7/vismatch/base_matcher.py#L97-L107)。
- 报告的指标是 AUROC、AUPRC（在全部候选对上算）和 Recall@1/5/10（按查询平均）（[#L493-L531](https://github.com/isaaccorley/rsim/blob/9950822cd7c9eaa23c233cc94099c1882ec6d2cc/src/rsim/sarptical_pair_eval.py#L493-L531)）。

---

## 10. 补充参考：MINIMA 多模态单应评测（SOMA-1M 的基线之一；不在本次指定范围内，仅作为"角点 AUC"的开源对照）

**源码**：https://github.com/LSXI7/MINIMA @ `796e7721174f9f829b79b3702bf8c2ae9a3d447a`，本地 `D:\Code\refs\MINIMA`（已有，复用）。入口是 `test_relative_homo_mmim.py`。
- 误差：4 个角点经 GT 单应和估计单应映射后求平均距离，**没有归一化**（[#L277-L295](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L277-L295)）。【注意】两组角点是**各自**经 `order_corners` 按坐标和/差重新排序后才配对的（[#L184-L195](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L184-L195)）。旋转较大时会配错角点，从而**低估**误差。【推断】
- 估计器：`cv2.findHomography(..., cv2.RANSAC)`，阈值用 OpenCV 默认值（[#L381-L382](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L381-L382)）。
- 失败：估计失败时记 `inf`，**计入 AUC**。但 "Average Mean Dist" 这一项会先**过滤掉 inf** 再平均（[#L405-L409](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L405-L409)、[#L449-L451](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L449-L451)）。
- AUC 阈值是 {1,3,5,7,10,15,20} px（[#L421-L422](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L421-L422)）。另外还报了 GT 下误差 ≤ 1/3/5 px 的匹配精度（[#L298-L320](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/test_relative_homo_mmim.py#L298-L320)）。

---

## 11. 横向对比表

| 工作 | 主指标 | 误差算在什么上 | 成功/正确判定 | 失败怎么计入 | 估计器（是否统一） | 聚合 | 依据 |
|---|---|---|---|---|---|---|---|
| MultiResSAR 综述 | SR、NCM、RMSE、TM | 人工 GT 控制点 + 估计变换 | NCM ≥ 20（加上 RMSE ≤ 10） | SR 计 0；RMSE/NCM 怎么处理未找到 | 各方法自带，不统一，细节未找到 | "average"，方式未找到 | 论文 |
| SRIF 仓库 | NCM、RMSE、time | 匹配点 + GT 仿射 | 单点 < 3 px；单对 < 10（或 ≤ 10）点即失败 | RMSE 记 20 px | 评测不做 RANSAC；各方法 exe 不统一 | 逐对保存，无汇总 | 代码 |
| RIFT demo | 无 GT 指标 | — | — | — | FSC affine 2 px，≤ 10000 次迭代 | — | 代码 |
| OS-Eval | ME（平均误差）、CMR | 独立 GT 检查点，经 RPC 和 DEM 投影 | 匹配数 ≤ 20 即失败 | ME = 999、CMR = 0；0 匹配的对被剔除 | 统一 affine RANSAC 3 px、≤ 10000 次迭代、重复 20 次（仅限指定方法名） | 对内平均；跨对未找到 | 代码 |
| XoFTR | 位姿 AUC@5/10/20° | GT 位姿 | AUC 多档 | ∞，计入 AUC | E-RANSAC 1.5 px（每个方法用自己的） | 场景内 AUC，再对场景平均 | 代码 |
| RoMa/DKM HPatches | 角点 AUC@3/5/10 | 4 个角点，归一化到 480 px | AUC 多档 | 退化 H 得到大误差，计入 AUC | H-RANSAC 3·s px，置信度 0.99999 | 全部对一起算 AUC | 代码 |
| SOMA-1M | 角点 AUC@5/10/20 px | 角点，GT = 合成扰动的逆 | AUC 多档 | 未找到 | 统一 H-RANSAC 1.5 px、10k 次迭代、置信度 0.9999 | 按数据集和分辨率子集；方式未找到 | 仅论文（代码未公开） |
| APM：SpaceNet9 | MeanErr、S@5/10、Fail | 人工 tie point；tiled 分段模型；resize 后的 px | 点级 ≤ τ 的比例 | NaN，从均值中**剔除**，单独报 Fail 率 | 统一 affine RANSAC 3 px（OpenCV 默认迭代数），每 tile 至少 4 个内点 | 对内均值，再在成功对上平均，附 bootstrap CI | 代码 |
| APM：SRIF | 同上 | 论文说 4 角点，**代码是 20×20 网格点** | 同上 | 同上 | 同上 | 按子数据集 | 代码 |
| MINIMA mmim（补充） | 角点 AUC、匹配精度 | 4 角点（重新排序后配对） | AUC 多档；点级 1/3/5 px | AUC 中计 ∞；平均距离中剔除 | H-RANSAC，OpenCV 默认值 | 按场景 | 代码 |

**对本项目评价协议的直接启示**（事实层面的对比，不代表对具体方案的建议）：
1. "误差算在什么上"主要有三类：(a) GT 下的匹配点，如 SRIF，衡量的是匹配器本身；(b) 估计变换映射后的独立检查点或角点，如综述公式、OS-Eval、SOMA、APM，衡量的是配准结果；(c) 估计器自身的拟合残差，如 RIFT FSC 和 SAR-SIFT，**不能**当作精度指标。
2. 失败处理有四种做法：固定惩罚（20 px 或 999）、∞ 计入 AUC、剔除后单独报失败率、剔除且不报。不同做法下的数值**不可直接比较**。
3. 估计器是否统一：OS-Eval、SOMA-1M、rsim 统一；综述和 SRIF 不统一。rsim 的扫参显示，RANSAC 阈值和几何模型的影响可以超过换匹配器的影响（APM 论文 p.5–6）。

---

## 12. 源码清单

| repo URL | commit | 本地路径 | 评测入口文件 |
|---|---|---|---|
| https://github.com/betterlll/Multi-Resolution-SAR-dataset- | 6f664903bc9662a66ac7c2c88a39f5e2d9da1ec4 | D:\Code\refs\Multi-Resolution-SAR-dataset- | 无（只有 README） |
| https://github.com/LJY-RS/SRIF | 88881a3a8789d0bed6a8df91b943e02f2593a0cd | D:\Code\refs\SRIF | demo_SRIF.m / demo_RIFT.m / demo_LNIFT.m 等 |
| https://github.com/LJY-RS/RIFT-multimodal-image-matching | 7ea830e2f13cc3c226f975fe9e98b7666a8f26fb | D:\Code\refs\RIFT-multimodal-image-matching | RIFT_demo.m（无 GT 评测）、FSC.m |
| https://github.com/xym2009/OS-Eval | 4df7d5a780d4c2d5b913ac7b7efa83ef44c43850 | D:\Code\refs\OS-Eval | CalME/src/Main.cpp、CalME/src/matchFunc.cpp |
| https://github.com/OnderT/XoFTR | e0fbea431b30be9742effbf5577c90aa8eb938f9 | D:\Code\refs\XoFTR | test_relative_pose.py、src/utils/metrics.py |
| https://github.com/Parskatt/RoMa | 77f8d68803526dcddfd9b7a46bc76125bdc25f15 | D:\Code\refs\RoMa | romatch/benchmarks/hpatches_sequences_homog_benchmark.py |
| https://github.com/Parskatt/DKM | ef57565db52684e661052ba82eb361329c63af3d | D:\Code\refs\DKM | dkm/benchmarks/hpatches_sequences_homog_benchmark.py |
| https://github.com/yishiliuhuasheng/sar_sift | 6601368b5bf9cff418068721e2f2de1aa3ecc81d | D:\Code\refs\sar_sift | 无 GT 评测（ransac.py） |
| https://github.com/arunsahu159/LNIFT-Locally-Normalized-Image-for-Rotation-Invariant-Multimodal-Feature-Matching | 1ffa3528cd88a655d9fb7e99dafb1a27ebe8d1ca | D:\Code\refs\LNIFT | 无（非官方 notebook） |
| https://github.com/betterlll/MOSS_data | 25c1cfcd1f0afc34a78af27dccd2637adf139a6e | D:\Code\refs\MOSS_data | 无（只有数据和 GT 点） |
| https://github.com/PeihaoWu/SOMA-1M | b86137c92b225402a4706eaee4bb7a61f11baabf | D:\Code\refs\SOMA-1M | 无（只有 README） |
| https://github.com/PeihaoWu/MapGlue | 81b06ec0f95c6f635aa6486a66147fdf974f6e2c | D:\Code\refs\MapGlue | 无（只有 README） |
| https://github.com/isaaccorley/rsim | 9950822cd7c9eaa23c233cc94099c1882ec6d2cc | D:\Code\refs\rsim | src/rsim/spacenet9_matcher_benchmark.py、srif_matcher_benchmark.py、sarptical_pair_eval.py |
| https://github.com/gmberton/vismatch | 9d49b892ed21625cee00bc7797a45d352bf659a7 | D:\Code\refs\vismatch | vismatch/base_matcher.py（统一匹配接口） |
| https://github.com/LSXI7/MINIMA（补充） | 796e7721174f9f829b79b3702bf8c2ae9a3d447a | D:\Code\refs\MINIMA | test_relative_homo_mmim.py |

没有获取的：HOWP（项目网页 skyearth.org）、ASS（gitee）。SuperGlue、LightGlue、LoFTR、Efficient LoFTR、SGM-Net、XFeat 的评测都是自然图像位姿任务，没有克隆。
