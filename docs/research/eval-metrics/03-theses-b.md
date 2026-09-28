# 评价协议调研 03：硕博论文（B 组：张俊、吕宁）

> 用途：为月球 CE-2 光学 / Mini-RF SAR patch 对仿射配准定义评价协议，收集已有光-SAR / 多模态配准工作的指标定义与计算细节。
> 范围：
> - 张俊，《基于深度无监督迁移学习的多模态图像配准与融合》，西安电子科技大学博士学位论文，2022-12（下称【张】）
> - 吕宁，《面向场景语义融合的多源遥感影像配准》，西安电子科技大学博士学位论文（下称【吕】）
>
> 页码约定：写作 `PDF p.N（印 p.M）`。两篇的正文印刷页 1 都对应 PDF 第 18 页（印 = PDF − 17）。
> 读法：两份 PDF 都有文字层（PyMuPDF 抽取），公式和表格用渲染后的页面图像核对过。只精读了评价、实验和方法中与评价相关的部分，没有通读全文。
> 本文中「未找到」表示已在相关章节找过、论文没有写，不是没去查。

---

## 1. 张俊（2022）

全文共 5 个技术章节。第二章做描述符学习（patch 级），第三～五章做配准，第六章做融合（与配准评价无关，只一笔带过）。各章的评价协议不一样，下面分章记录。

### 1.0 与本项目最相关的结论（先读这里）

- **RMSE 用的点对，各章定义不一样**：
  - 第三章：人工选取的 N 对真值点，经估计变换后算残差，属于真正的 GT 检查点误差。
  - 第四章：原文是「自动配准方法从图像对中选择 N 个点对」，即**方法自己匹配出的点对**在自己估计的变换下的残差（自洽残差，不是 GT 误差）。
  - 第五章：公式同第四章，但另写了「对应匹配对是手工确定的，每对选 40–60 个均匀分布的子像素点作为基准」。SIFT 等方法的 RMSE 高达 300 多像素，说明这一章的误差实际是在人工检查点上算的。
  （PDF p.60 / p.73 / p.89–90）
- **失败样本不剔除，也不做统一处理**：
  - 第三章：表中写「Failed」，但同一行仍报 NOCC 和 RT。
  - 第四章：写「*」，并注明「结果不适合比较，或与正确结果有很大差别」。
  - 第五章：大多数失败**直接报出巨大的 RMSE**（如 SIFT 337.38 px），少数写「*」。
  - **全程没有成功率指标，也没有像素阈值档位。**
- **无监督模型选择**：第五章训练轮次固定为 20，温度 τ=0.5 取自文献。领域自适应系数 λ 的取法是：在**测试用的 4 对大规模图像**上画「NOCC–λ」曲线（图 5.14），再从曲线上选 λ=0.1。没有独立验证集，也没有 early stopping 或 checkpoint 选择的说明（PDF p.89、p.97）。
- **聚合方式**：全部按图像对逐对列表，没有跨图像对的平均或中位数。第四章写了「每个图像对取 10 次的平均值」（PDF p.73）。

### 1.1 第二章 RDLNet（描述符学习，patch 匹配）

与仿射配准评价关系不大，只简记。

- **数据集**（PDF p.43–44）：
  - UBC PhotoTour：Liberty / Notredame / Yosemite，64×64 patch，训练 50 万对，测试 10 万对。
  - HPatches：116 个序列。
  - RGB-NIR Scene：477 张，9 类。训练用 country，在其余 8 类上测试，并做交叉实验。
- **指标**（PDF p.43–44）：
  - FPR95：UBC 和 RGB-NIR 用，越小越好。RGB-NIR 同时报 8 个场景的 mean 和 std。
  - mAP：HPatches 的 verification / matching / retrieval 三个任务。
  - 论文没有给 FPR95 或 mAP 的公式，只有文字说明。**公式：未找到。**
- **对比方法**（PDF p.45–46）：SIFT、DeepDesc、MatchNet、L2Net、CS L2Net、HardNet、HardNet-GOR、SATN、SOSNet、TFeat、PN-Net、Q-Net、DOAP、Exp-TLoss、GeoDesc 等。

### 1.2 第三章 基于图像迁移（Deep Image Analogy）+ SIFT/SURF 的配准

- **数据集**（PDF p.59–60，印 p.42–43）：3 对图像，均来自 Ye 等 HOPC_ncc 的测试集 [143]。
  - (a) Daedalus 可见光/红外，512×512，0.5 m。
  - (b) LiDAR/机载可见光，524×524，2.5 m。
  - (c) Google Earth 光学 528×524 与 TerraSAR-X SAR 534×524，3 m。
  - 预处理：图像对「已经消除了明显的比例、平移和旋转变换，但仍然存在较小的位置偏移（几个像素）」。
- **GT 来源**：人工选取 N 个对应点 (x_i,y_i)⇔(x'_i,y'_i) 表示真实映射（PDF p.60）。表 3.1 的 Manual 行 NOCC=30，推测 N=30（PDF p.61）。
- **指标**（PDF p.60，式 3-4）：
  - RMSE：
    $$RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\hat x_i)^2+(y_i-\hat y_i)^2}$$
    其中 $(\hat x_i,\hat y_i)$ 是 $(x'_i,y'_i)$ 经估计变换后的坐标。误差在**人工真值点对**上算。
  - NOCC（正确匹配数）：论文说它「用来评价稳健性」，但**没有给出「正确」的判定阈值**（未找到）。
  - RT：运行时间（秒）。
- **成功判定**：没有阈值定义。SIFT 和 SURF 在 (b)(c) 上标为「Failed」/「Faild」，但仍列出 NOCC（例如 7、10、5、4）和 RT（PDF p.61 表 3.1）。
- **失败计入**：在 RMSE 栏写「Failed」，其它栏照常报数，不参与任何平均（本章本来也没有平均）。
- **鲁棒估计器**：引入 RANSAC，d_ratio（NNDR）= 0.85（PDF p.60）。RANSAC 的阈值、迭代数、变换模型都未找到。RANSAC 是否对所有对比方法共用，原文没有明说。
- **聚合**：3 对图像逐对列表。
- **对比方法**：Manual、SIFT、SURF、HOPC_ncc、Proposed+SURF、Proposed+SIFT（PDF p.60–61）。

### 1.3 第四章 TSRM：两步配准（VGG16 深度特征区域匹配 + SIFT 局部特征）

- 对应发表：Ma W, Zhang J, et al., TGRS 2019, 57(7):4834–4843（论文参考文献 [103]）。
- **数据集**（PDF p.72–73）：
  - (a)(b)(c) Radarsat-2 SAR 对：黄河口，four-look 作参考、one-look 作待配准，8 m，尺寸 600×500 / 400×400 / 400×400。
  - (d) Google Earth 光学（5 m）与 Radarsat-2 SAR（400×400）。
  - (e) 腾讯街景地图与光学，800×800。
  - (f) 腾讯与 Google 光学，500×500。
- **微调数据**（PDF p.68）：10 幅已精确配准的遥感图像，裁出 1000 个 224×224 不相交块。每块作为一类，用随机变换扩成 400 个样本，按分类任务微调。这是有监督的代理任务。
- **GT 来源**：「图像对的实际变换矩阵是由 30 个人工标注的匹配对计算出来」（PDF p.76，印 p.59），用于表 4.6 的区域匹配误差。
- **指标**（PDF p.73–74，式 4-4～4-8）：
  - RMSE（式 4-4）：
    $$RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\tilde x'_i)^2+(y_i-\tilde y'_i)^2}$$
    原文写的是「自动配准方法从图像对中选择 N 个点对」，即**在方法自己输出的匹配对上**计算变换后的残差。这是自洽残差，不是 GT 误差。Manual 行的 RMSE 为 1.78～2.57，反而比所有自动方法差（PDF p.75–76），也印证了这一点。
  - NOCC：正确匹配对数量。**「正确」的阈值：未找到。**
  - MI（式 4-5～4-7），直方图估计：
    $$MI(X,Y)=H(X)+H(Y)-H(X,Y)$$
    $$H(X)=-\sum_i p_X(i)\log p_X(i)$$
    $$H(X,Y)=-\sum_{i,j} p_{XY}(i,j)\log p_{XY}(i,j)$$
    直方图的 bin 数和对数底：未找到。
  - ROCC（式 4-8）：
    $$ROCC=\frac{NOCC}{NOCC+NOFC}$$
    其中 NOFC 是错误匹配数。
  - RT：运行时间。
  - 区域匹配误差「误差(x/y)」：相对 30 点 GT 变换，分 x 和 y 两个方向报（PDF p.77，表 4.6）。
- **成功判定**：无阈值。(d)(e)(f) 上的「SIFT-like」各栏全写「*」，表示「获得的结果不适合比较，或者……与正确结果有很大差别」（PDF p.74–76）。
- **失败计入**：写「*」。
- **鲁棒估计器**：「由于 RANSAC 在单应矩阵估计上的有效性，它被应用于所有的配准方法中」，即各方法共用 RANSAC（PDF p.74）。所有方法 d_ratio = 0.9。局部匹配半径 R = 6。SAR 对取 top-10 近邻，多模态对取 top-30（PDF p.75）。RANSAC 的内点阈值和迭代数：未找到。变换模型：式 4-3 写作 A=HB，称「单应矩阵」（PDF p.70、p.74）。
- **聚合**：每个图像对跑 10 次取平均，再逐对列表（PDF p.73）。
- **对比方法**：Manual、SURF、SIFT、RSCJ、PSO-SIFT、SAR-SIFT，以及 Proposed+SURF / +SIFT / +SAR-SIFT（PDF p.75）。

### 1.4 第五章 领域自适应对比注意力学习（无监督）+ 渐进匹配

- **训练（无监督）**（PDF p.83–85、p.89）：
  - 对比损失（式 5-2，InfoNCE 形式，cos 相似度）。
  - 注意力图位置损失（式 5-5）：利用增强时记录的变换 H_a，这是自监督。
  - 领域混淆损失（式 5-6）。
  - 总损失（式 5-7）：L_a·L_c + λL_d。
  - 网络 VGG-A 加 BN；约 5000 个 256×256 多模态对；增强为随机裁剪（旋转、缩放）、颜色失真、高斯模糊；epoch 20，batch 64，Adam lr 1e-4，τ=0.5，λ=0.1。
  - 原文说训练数据「是通过对参考图像和待配准图像的随机裁剪和扩展得到的」（PDF p.89），另一处又写「实践中通常只有两幅图像需要配准」（PDF p.83）。据此理解，训练数据可能来自待配准图像本身（直推式），但原文没有说明训练集和测试集是否分开。**训练/测试划分：未找到。**
- **模型选择 / 超参选择**（本项目重点）：
  - τ=0.5 直接沿用文献 [86]（SimCLR）的设定（PDF p.97）。
  - λ：图 5.14 以 λ 为横轴、NOCC 为纵轴，画的是「Large-Scale Multimodal Results」中 Image Pair (a)～(d) 的曲线，也就是 5.3.5 节的 4 对测试图像。据此选 λ=0.1（PDF p.97）。**超参在测试集上选取，没有独立验证集。**
  - 训练 epoch 固定为 20。没有 checkpoint 选择、早停或验证损失的说明（未找到）。
  - 各组件的消融（域适应、对比注意力、精细化）只给定性图，或给 MI / NOCC 曲线（PDF p.96–99）。
- **匹配与估计**（PDF p.86–89）：
  - Conv4 区域匹配（NNDR），逐层位置调整到 Conv1。
  - 在 conv1 特征上做泰勒展开，得到亚像素精细化（式 5-9～5-13）。
  - RANSAC 去外点，再用最小二乘估计 **6 参数仿射**（式 5-14 / 5-15，h00..h12）。
  - RANSAC 的阈值和迭代数：未找到。
- **数据集**：
  - 5.3.3 Nirscene（RGB-NIR）：RGB 作参考，NIR 随机缩放 ±0.2、旋转 ±20°，按 @0°/5°/10°/20° 报平均 NOCC（PDF p.91，表 5.1）。GT 是**合成变换**。
  - 5.3.4 多模态遥感 8 对 (a–h)（PDF p.91，表 5.2），均为「小像素偏移」对：
    - (a) Radarsat-2 与合成图像，400²
    - (b) LiDAR/可见光，480×550
    - (c) Landsat5 TM 红外/可见光
    - (d) Google Maps
    - (e) Google Earth / TerraSAR-X，618×628，3 m
    - (f) Landsat5 TM / Sentinel-1A，688×500，30 m
    - (g) Daedalus 光学/红外，512²
    - (h) Landsat-5 热红外/光学，412×300
  - 5.3.5 大规模 4 对（PDF p.94，表 5.4）：
    - (a) Radarsat-2 黄河口，800²
    - (b) 腾讯地图/街景，700²
    - (c) DFC2018 Houston，1202²
    - (d) Google Earth，1048×724
- **GT 来源**：「一般来说，对应匹配对是手工确定的。在每对参考图像和待配准图像之间选择 40 到 60 个均匀分布的匹配子像素点作为基准」（PDF p.90，印 p.73）。表 5.5 中 Manual 行 NOCC = 40（PDF p.94）。
- **指标**（PDF p.89–90，式 5-16、5-17）：
  - RMSE：
    $$RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left((x_i-\tilde x_i)^2+(y_i-\tilde y_i)^2\right)}$$
  - MAE（逐点欧氏距离的均值）：
    $$MAE=\frac{1}{N}\sum_{i=1}^{N}\sqrt{(x_i-\tilde x_i)^2+(y_i-\tilde y_i)^2}$$
  - NOCC：「越高越容易获得准确结果」。**正确的阈值：未找到。**
  - MI：沿用第四章的定义，本章不再重复公式。
  - RT：运行时间。
  - 定义里的点对写作「配准方法从图像对中选择 N 个点对」，而基准点是人工的 40–60 点。从 SIFT 的 RMSE=337.38、RDLNet 的 RMSE=285.49 这类数值（PDF p.93 表 5.3）判断，误差是**在人工检查点上用估计变换算的**，否则误匹配方法不会出现数百像素的误差。原文没有把两处说法统一起来。
- **成功判定**：无阈值，也不报成功率。
- **失败计入**（表 5.3，PDF p.93）：
  - 失败方法**照常报巨大的 RMSE 和 MAE**，不剔除、不截断。例如 SIFT 在 b/c/e/f 上为 337/388/313/276 px，RDLNet 在 b/c/e/f 上为 285/306/291/337 px。
  - 表 5.5 中 RDLNet 在 (b) 上各栏写「*」（PDF p.94）。
  - 文中报「相对前最好方法 RMSE 降低 x%」，允许出现负值（例如 −0.88%、−21.08%）（PDF p.91）。
- **鲁棒估计器与公平性**：
  - 本方法：RANSAC + 最小二乘仿射。
  - SIFT、HOPC、PSO、WOA、Powell「参数设置参考对应的工作 [21]」，用 MATLAB R2018a 实现。
  - TSRM、HardNet、RDLNet「根据原始论文中的参数设置」，用 PyTorch 实现（PDF p.89）。
  - 是否共用同一估计器：**未找到**。
- **聚合**：逐对列表，没有跨对平均。表 5.1 的「Average NOCC」是 Nirscene 在同一旋转角下对多对图像取平均（PDF p.91）。
- **对比方法**：SIFT、HOPC、PSO（模板匹配 PSO [19]）、WOA（Yan 等，transfer optimization [21]）、Powell（[188] 强度法）、TSRM、HardNet、RDLNet（PDF p.89）。

### 1.5 第六章 TCGAN 融合

这是融合任务，评价用融合质量指标（TNO / RoadScene / Harvard），与配准无关，不展开（PDF p.109–118）。

---

## 2. 吕宁

各章主题：
- 第二章：数据来源
- 第三章：PSGF（凸显结构描述符）
- 第四章：EC-RIFT（异质边缘完整性）
- 第五章：SrGAN 数据增强（评价的是分割质量，不是配准）
- 第六章：对比表示学习配准（自监督）

### 2.0 与本项目最相关的结论（先读这里）

- **RMSE 一律在「匹配的关键点对」上算**。做法是把待配准点经仿射变换映射到参考图，再算残差，N 取「匹配的点对数量」（PDF p.59，式 3-11）。**没有独立的人工检查点，也没有写明映射用的是 GT 变换还是估计变换**，看起来是估计变换下的内点残差。所以报出的 RMSE 全部落在 1–2 px 区间（PDF p.64、p.84、p.144）。
- **NCM 的定义是「迭代剔除误差较大的点对后剩下的点」**（PDF p.141），实际就是外点剔除后的内点数，并没有用 GT 核验。
- **失败的判定与计入**：失败写「-」。第四章明确定义「-」为「算法失效或者正确匹配的同名点对数量少于 5 个」（PDF p.83–84）。「-」不参与任何统计。第四章比较平均运行时间时，失败方法「未进行比较」（PDF p.85）。
- **估计器**：全篇用 NNDR + **FSC**（fast sample consensus）估计**仿射**变换，并且对比方法「外点移除和匹配部分……都是用同样的代码实现」（PDF p.58、p.60、p.78、p.83）。这是两篇论文中唯一明确写了「共用估计器」的地方。
- **无监督（自监督）模型选择**：第六章训练 48 epoch，batch 16，lr 1e-3，τ=0.1，wd 1e-4（PDF p.141，表 6.1）。SEN1-2 验证集有 1980 对，但只用来画训练/验证损失曲线（图 6.8，PDF p.133）。**checkpoint 或超参的选择准则：未找到。**

### 2.1 数据来源总表（第二章，PDF p.41–42，表 2.1）

| 平台 | 尺寸 | 分辨率 | 区域 |
|---|---|---|---|
| Google Earth / TerraSAR-X | 528×520 / 534×513 | 3 m | 城市 |
| Google Earth / TerraSAR-X | 516×574 / 634×562 | 1 m | 某机场 |
| Google Earth / TerraSAR-X | 628×618 | 3 m | 城市区域 |
| TM band3 / TerraSAR-X | 600×600 | 30 m | 平原 |
| Sentinel-1 / Sentinel-2 | 256×256 | 10 m | 城市区域 |
| Google Earth / 高分三号 | 256² 或 512² | 1 m | 全世界多个城市 |
| 高分一号 / 高分三号 | 128² 或 256² | 2 m | 江西、安徽 |

原文称这些数据「都包含可以进行预配准的地理信息，或者已经经过预配准」（PDF p.42）。

### 2.2 第三章 PSGF（凸显结构 WIV 检测 + MRI 滤波描述符）

- **数据集**（PDF p.59）：
  - SAR/光学对，覆盖城区、郊区、平原、河流、人工岛、火山岛等场景（表 3.2 共 6 组）。
  - 数据源为 TerraSAR-X、高分、Google Earth、Landsat，分辨率 3–10 m。
  - 每个场景的对数：表中每个场景 1 列，推测每场景 1 对，原文未明确（未找到）。
- **GT 来源**：未找到（没有提到人工检查点或合成变换）。
- **指标**（PDF p.59，式 3-10、3-11）：
  - Repeatability（式 3-10）：
    $$Repeatability=\frac{N_c}{(N_1+N_2)/2}$$
    N₁、N₂ 是两图各自检测到的特征点数，N_c 是潜在同名点对数。判定规则：匹配后特征点的欧氏距离小于阈值，同源配准取 2 px，**多源配准本文取 3 px**。
  - RMSE（式 3-11）：
    $$RMSE=\sqrt{\frac{\sum_{i=1}^{N}(x_{2i}-x_{1i})^2+(y_{2i}-y_{1i})^2}{N}}$$
    $(x_{1i},y_{1i})$ 是参考图关键点，$(x_{2i},y_{2i})$ 是待配准关键点经**仿射变换**映射到参考空间后的坐标，N 是**匹配的点对数量**。
  - 运行时间：只报了 PSGF 的平均值 12.53 s（PDF p.65）。
- **成功判定**：无阈值。OS-SIFT「在 4 组场景中都没有匹配成功」，表中记「-」（PDF p.64，表 3.2）。
- **鲁棒估计器**：NNDR 匹配，加 FSC 外点剔除并估计变换（PDF p.58）。对比方法「外点移除和匹配部分的算法都是用同样的代码实现」，对比算法的其它参数用原文献参数（PDF p.60）。特征点上限 5000（PDF p.60）。FSC 阈值和迭代数：未找到。
  - 注：论文 3.2.2 节把 FSC 写成「Feature Selection Criterion」并引用 [131]《Feature Selection: A Data Perspective》（PDF p.53），但 3.3.2 节又写作「快速一致性采样算法（Fast Sample Consensus, FSC）」（PDF p.58）。前后不一致，按后者理解。
- **聚合**：逐场景列表，没有平均。
- **对比方法**：HAPCG、OS-SIFT、RIFT、RIFT2、ASS、LNIFT（PDF p.60）。检测器消融另比了 PC 与 WIV（表 3.1）。

### 2.3 第四章 EC-RIFT（去噪/增强预处理 + OLG 滤波 + 扇区 GLOH 型描述符）

- **数据集**（PDF p.79）：数据取自对比方法文献中公开的多源影像，组 1–10。
  - 组 1–7：SAR/光学，来源为 TerraSAR-X、高分、Google、航空，场景包括城区、洪水、河流、农田。
  - 组 8：红外/光学。组 9：日/夜。组 10：深度/光学。
  - 尺寸、分辨率见表 2.1。
- **GT 来源**：未找到。
- **指标**（PDF p.79–80）：
  - Repeatability：同式 3-10，但潜在点判定收紧为下式（式 4-17）：
    $$\sqrt{(x_{2i}-x_{1i})^2+(y_{2i}-y_{1i})^2}<2$$
    其中 $(x_{2i},y_{2i})$ 为「待配准影像经过仿射变换映射到参考影像」的坐标。这个仿射是 GT 还是估计的：**未找到**。
  - RMSE：同式 3-11。
  - T_proc：运行时间（秒）。
- **成功判定与失败计入**：「-」表示「算法失效或者正确匹配的同名点对数量少于 5 个」（PDF p.83–84）。平均运行时间统计时，失败的方法「未进行比较」（PDF p.85）。
- **鲁棒估计器**：NNDR + FSC，估计**仿射**模型（PDF p.78）。「实验测试的所有参考算法都使用相同的匹配方法，并且相关的算法参数都使用各自文献中推荐的设置」（PDF p.83）。
- **聚合**：逐组列表（表 4.1～4.4）。运行时间另外对所有组求平均（图 4.13）。
- **对比方法**：HAPCG、OS-SIFT、LNIFT、RIFT（PDF p.80）。
- **适用范围**（PDF p.87–88）：
  - 论文假设输入已经过地理预配准，残余主要是平移。
  - 对缩放 ≤0.5 或 ≥1.5、旋转 >10° 的情况，方法开始不稳定。

### 2.4 第五章 SrGAN（遥感数据语义增强）

这一章评价的是**生成图像质量**，用下游分割/解译来衡量，**不是配准指标**（PDF p.104–105）。
- **指标**：
  - Acc_class（式 5-4）
  - Acc_pixel（式 5-5）
  - IoU_class（式 5-6）
  - T_gen（生成时间）
  - 解译评分 S_interp（式 5-7，图斑/非图斑两类 F1 的平均）
- **评价方式**：用 FCN-8s 分割生成图像（PDF p.104）。
- **数据**：高分数据集（2 m，13989×9359）、Google 地图分割数据集（1200 张，600²）、Mnih Massachusetts 建筑数据集（PDF p.102–104）。

与本项目无直接关系。

### 2.5 第六章 基于对比表示学习的多源配准（自监督）

- **训练**（PDF p.129–141）：
  - 逐块正负样本采样：锚块在 SAR 上，同位置的光学块作正样本，其余块作负样本。
  - 正样本做弱增强：噪声、高斯模糊、边缘检测、cutout，外加随机镜像、翻转、旋转。
  - 解耦 InfoNCE（式 6-10、6-11），加难分正样本期望（式 6-12～6-14）。
  - 网络：U-Net 编码器，下采样用 Inception，跳连接上加残差空间注意力，并加激活正则。
  - 训练依赖 SEN1-2 已配准对的「空间对齐」作为约束，原文称之为「半监督约束」（PDF p.135），因此不是完全无标注。
- **推理流程**（PDF p.139–140）：
  1. 预处理增强后，候选点检测（WIV / 异质边缘 / SIFT / ORB）。
  2. 在编码图上取邻域，用 HOG 或第三、四章的描述符描述。
  3. 匹配、外点剔除，估计空间变换。
- **数据集**（PDF p.140–141）：
  - SEN1-2（Sentinel-1 IW SAR 与 Sentinel-2 RGB，10 m，256×256）：训练 6579 对，验证 1980 对。光学转灰度。
  - 测试按 6 种地形各取一组：道路、水体、建筑、农田、山区、城郊。每组对数：未找到。
  - 多源泛化测试：深度/光学、红外/光学、地图/光学、SAR/光学、日/夜（Suomi NPP VIIRS），先在少量新测试影像上「迁移训练」再测试（PDF p.147）。
  - 另有实拍航空 SAR/光学两例（PDF p.149）。
- **GT 来源**：SEN1-2 自带已配准对，但测试是否在其上叠加了合成变换：**未找到**。多源测试影像的 GT：**未找到**。
- **指标**（PDF p.141）：
  - NCM：「在特征点匹配过程中会迭代的剔除误差较大的匹配点对，剩下的满足精度要求的特征点对即为正确匹配点数」。**「精度要求」的阈值：未找到。**
  - RMSE：原文说「依然采用和前述章节相同」的定义，即式 3-11。第三、四章实际并没有用 NCM，这里的说法与前文不一致。
- **成功判定与失败计入**：失败写「-」，原文说明为「匹配失败」或「NCM 不足」，NCM 下限没有另外定义（表 6.2–6.7，PDF p.142、p.144、p.147）。第四章的「<5 对」定义是否沿用到这里：未找到。
- **鲁棒估计器**：Pix2Pix、Pix2PixHD、SrGAN 先把 SAR 转成光学，再「均使用同样的特征提取算法（ORB）和图 6.13 中相同的匹配流程进行配准，所有算法使用的测试图像与相关的参数设置也保持一致」（PDF p.144）。外点剔除的具体算法和阈值：本章未写，按前文应为 FSC，但**未找到明确说明**。
- **模型选择**：固定 48 epoch（表 6.1）。验证集只用于观察损失曲线（图 6.8）。**checkpoint 或超参选择：未找到。**
- **聚合**：逐地形、逐模态列表，没有平均。
- **对比方法**：RIFT、RIFT2、ASS、Pix2Pix、Pix2PixHD、SrGAN（表 6.4、6.5）；多源泛化测试只比了 Pix2PixHD（表 6.6、6.7）。基于描述符的方法「需要手动调参」，所以在多源泛化测试中不参与对比（PDF p.147）。

---

## 3. 横向对比表

| 章节 | 数据 / GT | 精度指标（误差在什么上算） | 其它指标 | 成功阈值 | 失败计入 | 估计器 / 变换 / 是否共用 | 聚合 | 模型选择 |
|---|---|---|---|---|---|---|---|---|
| 张 ch3 | 3 对（可见/红外、LiDAR/光学、GE/TerraSAR-X），GT 为人工 N 点（≈30） | RMSE，**人工真值点** | NOCC（无阈值）、RT | 无 | 「Failed」，其余栏照报 | RANSAC，ratio 0.85；模型未写；是否共用未写 | 逐对 | 不涉及 |
| 张 ch4 | 3 对 SAR-SAR 与 3 对多模态；GT 为 30 个人工点（仅用于区域匹配误差） | RMSE，**方法自身匹配对**（自洽）；区域匹配 x/y 误差对 GT | NOCC、MI、ROCC、RT | 无 | 「*」 | RANSAC，**所有方法共用**，ratio 0.9，「单应」 | 每对 10 次平均，逐对列 | 微调用人工构造的有监督数据 |
| 张 ch5 | Nirscene（合成旋转/缩放）；8 对与 4 对多模态，每对人工 40–60 个子像素点 | RMSE、MAE，**人工检查点**（依数值推断） | NOCC、MI、RT | 无 | 失败**照报巨大 RMSE**（数百 px），偶有「*」 | RANSAC 加 LSQ **6 参数仿射**；对比方法用各自原设置 | 逐对 | epoch 固定 20；**λ 在测试对上按 NOCC 选** |
| 吕 ch3 | 6 场景 SAR/光学；GT 未写 | RMSE，**匹配的关键点对**（估计仿射下） | Repeatability（3 px） | 无 | 「-」 | NNDR 加 **FSC**，仿射，**共用同一实现** | 逐场景 | 不涉及 |
| 吕 ch4 | 10 组（7 组 SAR/光学与 3 组其它）；GT 未写 | RMSE，同 ch3 | Repeatability（2 px）、T_proc | **正确匹配 <5 对即失败** | 「-」，平均时不计入 | NNDR 加 FSC，仿射，共用 | 逐组；时间取平均 | 不涉及 |
| 吕 ch6 | SEN1-2（训练 6579 / 验证 1980），6 地形测试组，外加 5 种多源 | RMSE（同式 3-11）、NCM = 剔除后内点 | — | 未定义（「NCM 不足」） | 「-」 | 生成类方法统一用 ORB 加相同流程；外点剔除未明写 | 逐组 | epoch 固定 48，验证集只看 loss |

对本项目的启示（据以上事实归纳，属于推论）：
1. 这两篇论文都**没有成功率、阈值档位或 CDF 类指标**，失败处理也不统一：张俊的写法有照报数百 px、写「*」、写「Failed」三种，吕宁写「-」并剔除。我们的协议需要自己明确规定失败的计法。
2. 「RMSE」这个名字下至少有 3 种算法：
   - 人工 GT 点上的误差（张 ch3、ch5）
   - 方法自身内点的残差（张 ch4、吕全篇）
   - 相对 GT 变换的 x/y 误差（张 ch4 表 4.6）
   其中第二种系统性偏小，跨论文比较数值没有意义。
3. 无监督方法的模型和超参选择，在这两篇里都**没有干净的验证协议**：张 ch5 在测试对上选 λ，吕 ch6 固定 epoch。
4. 只有吕宁明确写了「对比方法共用同一匹配和外点剔除实现」（FSC、仿射）；张 ch4 写了共用 RANSAC。

---

## 4. 源码清单

「官方」指作者本人或其机构账号。以下全部 `git clone --depth 1` 到 `D:\Code\refs\`。

| 方法（出现章节） | repo URL | commit | 本地路径 | 备注 |
|---|---|---|---|---|
| RIFT（吕 ch3/4/6） | https://github.com/LJY-RS/RIFT-multimodal-image-matching | 7ea830e2f13cc3c226f975fe9e98b7666a8f26fb | D:\Code\refs\RIFT-multimodal-image-matching | 已存在，复用；作者 Li Jiayuan 官方 |
| RIFT2（吕 ch3/6） | https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation | 0e980ce4124d2abd62727dcf040198db0a18b269 | D:\Code\refs\RIFT2-multimodal-matching-rotation | 官方 |
| LNIFT（吕 ch3/4） | https://github.com/LJY-RS/LNIFT_exe | 6349a8c47b4fc3c58c28c8aa526096831de06bdd | D:\Code\refs\LNIFT_exe | 官方，按 repo 名看是可执行文件发布；另外 D:\Code\refs\LNIFT 是第三方实现（arunsahu159），不是本调研克隆的 |
| HAPCG（吕 ch3/4） | https://github.com/yyxgiser/HAPCG-Multimodal-matching | 79aa1efee0039a9220897dba6f65faeb6b49bba6 | D:\Code\refs\HAPCG-Multimodal-matching | 账号对应一作 Yao Yongxiang（按账号名判断） |
| OS-SIFT（吕 ch3/4） | https://github.com/xym2009/OS-SIFT | 631a8060042ed89ea11e43d97e44b387443b1eb6 | D:\Code\refs\OS-SIFT | 与 OSdataset 同一账号（Xiang Yuming），视为作者发布 |
| HOPC / HOPC_ncc（张 ch3/5） | https://github.com/yeyuanxin110/HOPC | bb0aa72e9cbaf70a9892a4bb709fd8ae435beb8f | D:\Code\refs\HOPC | 作者 Ye Yuanxin 账号 |
| Deep Image Analogy（张 ch3 的迁移模块） | https://github.com/msracver/Deep-Image-Analogy | 632b9287b42552e32dad64922967c8c9ec7fc4d3 | D:\Code\refs\Deep-Image-Analogy | MSRA 官方 |
| HardNet（张 ch2/5） | https://github.com/DagnyT/hardnet | b1e9967299e36741bcc01d27c2312e91633d8443 | D:\Code\refs\hardnet | 官方 |
| Pix2Pix（吕 ch6） | https://github.com/phillipi/pix2pix | 89ff2a81ce441fbe1f1b13eca463b87f1e539df8 | D:\Code\refs\pix2pix | 官方（Lua/Torch 原版） |
| Pix2PixHD（吕 ch6） | https://github.com/NVIDIA/pix2pixHD | 14b3b3c7fff413086e3b58df52096f16b6891172 | D:\Code\refs\pix2pixHD | 官方 |

未找到官方源码（未克隆）：
- ASS（Xiong et al., TGRS 2022）：搜索引擎和 `gh search repos` 都没有找到。
- RDLNet、TSRM、张 ch5 的方法、PSGF、EC-RIFT、吕 ch6 的方法（论文作者本人的方法）：论文中没有给链接，`gh search repos "RDLNet"` 只搜到无关或空的 repo。
- PSO-SIFT、SAR-SIFT：只找到第三方重实现，没有克隆，只登记：
  - https://github.com/ZeLianWen/Image-Registration（含 SIFT、SAR-SIFT、PSO-SIFT）
  - https://github.com/yishiliuhuasheng/sar_sift（别的 agent 已放在 D:\Code\refs\sar_sift）
- FSC（Wu et al.）：只找到第三方 https://github.com/ShowStopperTheSecond/FSC，未克隆。
- RSCJ、PSO 模板匹配 [19]、WOA / transfer optimization [21]、Powell [188]：未找到。
- SIFT、SURF：用 OpenCV 自带实现即可，不单列。

检索来源：
- [LJY-RS GitHub](https://github.com/LJY-RS)
- [RIFT2 repo](https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation)
- [LNIFT_exe](https://github.com/LJY-RS/LNIFT_exe)
- [xym2009/OSdataset](https://github.com/xym2009/OSdataset)
- HAPCG 论文页：[武汉大学学报](https://ch.whu.edu.cn/en/article/doi/10.13203/j.whugis20200702)
- ASS 论文页：[IEEE Xplore 9852267](https://ieeexplore.ieee.org/document/9852267/)
- 其余 repo 通过 `gh search repos` 和 `gh repo view` 确认
