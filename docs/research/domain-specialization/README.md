# 其他垂域的深度学习怎样针对自己的数据做特化（#142）

> **2026-10-16 核对说明**：本报告没有一条读过全文，机制都据摘要或题录，没标【摘要】的条目也不能当作读过原文。「结论速览」里的归纳多为推断：「领域先验落在六处」「最常见的套路是化为单模态」「侧视几何几乎都靠 DEM」（只有 R17、R18、P7 三条）、「论文常见写法」，都不能当作文献结论。已知错误：
> - 「近期稠密方法（GDROS、SOMA）都用『全局仿射 + 稠密流细化』」不对。按 arXiv 摘要，GDROS 是先出稠密光流，再回归一个仿射去约束光流；只有 SOMA 是「全局仿射 + 局部流」。
> - 五个领域实际有 61 条，不是「约 50 条」。
> - 「没有找到用深度学习做 Mini-RF 与光学配准的工作」只是本次检索的结果，检索范围没有记录；Wang 2025（遥感学报，月球可见光与 SAR 匹配方法比较）正文未读到。
>
> 以 #142 票里修订后的「结论」为准。

> 问题：其他垂直领域的深度学习工作，怎样针对自己领域的数据做特化？范围以配准、匹配为主，覆盖医学多模态配准（含形变场输出）、地球光学-SAR、红外-可见光、显微 / 病理、行星科学。另外归纳一个框架：领域先验放在哪里，论文怎么包装这类贡献。
>
> 约束（来自地图 Notes「数据特性驱动的改进」）：先广撒网，**不评估**与我们方法的契合度和实现难度。依赖辅助数据（DEM、成像元数据、分割标签、目录等）的思路照列，用 **【辅助】** 标出，并写明需要什么数据。
>
> 出处核对情况：带 arXiv 号的条目，标题和号码都已通过 arXiv API 逐条核对。没有 arXiv 版的条目给出会议页、DOI 或 IEEE 页。机制描述依据论文摘要和作者对原文的了解，本次**没有逐篇重读全文**。标 **【摘要】** 的条目只看过摘要或检索摘要，细节以原文为准。

## 结论速览

1. **领域先验落在六个地方**：①输入表示（换成对模态不敏感的特征或可学的公共表示）、②网络结构（分支是否共享权重、梯度 / 相位等手工先验嵌进网络、粗到细级联）、③训练数据（模态翻译、物理仿真、按领域规律合成）、④损失和自监督信号（换相似度度量、学一个跨模态相似度、利用噪声统计）、⑤输出表示（仿射 → 网格单应 → 稠密形变场，加光滑、可逆、局部刚性等约束）、⑥训练流程（困难负样本、课程、推理时逐对优化）。见第 1 节。
2. **多模态配准里最常见的套路是「把多模态问题变成单模态问题」**，有两条路：翻译成对方模态（Merkle 2018 光学→SAR、UMF-CMGR 可见光→伪红外、Arar 2020），或者学一个公共表示（CoMIR、ContraReg）。有一篇专门的对比研究（Lu 2022）发现翻译式在显微数据上常常不如直接学公共表示，这类工作的效果分歧较大。
3. **医学形变配准的特化主要在输出端**：输出稠密形变场，再给它加领域规律（光滑、微分同胚、逆一致，以及骨头局部刚性、滑动界面等生物力学约束），损失换成 MIND、NMI、对比学习相似度这类跨模态度量。SynthMorph 走得更远：完全用随机合成的「标签图→随机外观」训练，不碰真实影像，获得对对比度不敏感的能力。
4. **光学-SAR 的特化集中在三点**：外观差异（伪孪生结构、梯度 / 相位先验嵌入网络，如 SOMA、RIFT/CFOG 的思路）、斑点噪声（MERLIN、SAR2SAR 用噪声统计做自监督）、侧视几何（主要靠 DSM/DEM 仿真 SAR 或成像角条件生成，**几乎都要辅助数据**）。近两年的稠密方法（GDROS、SOMA）都采用「全局仿射 + 稠密流细化」的输出结构。
5. **红外-可见光的特化**：伪红外数据增强 + 掩码预训练（XoFTR），配准与融合联合训练、输出双向形变场（SuperFusion、RFNet/MURF、ReCoNet）。
6. **显微 / 病理的特化**：染色颜色增强（Tellez 2019）、先粗对齐旋转再仿射再非刚性的分级流程（DeepHistReg 系），连续切片用自监督逐对位移场 + 多次投票提高稳健性（SEAMLeSS）。
7. **行星科学**：DL 配准工作很少，多数是检测（陨石坑）或用仿真补数据。特化手段主要是「用真实正射图 + 光照仿真生成训练对」（Geo-LoFTR）、物理噪声模型合成训练数据（HORUS），以及以陨石坑为匹配单元做导航（LunaNet）。这几类基本都**依赖 DEM、相机元数据或陨石坑目录**。没有找到用深度学习做 Mini-RF 与光学配准的工作，现有的 Mini-RF 配准用 DEM 仿真 SAR + NCC。
8. **论文的包装方式**：常见的是「指出通用方法的一个领域失效现象 → 用领域物理或统计规律解释它 → 把这条规律变成某个可学习的模块、损失或数据生成器 → 在领域数据集上报告提升，并用消融证明是这条先验起了作用」。题目里通常直接写出领域性质（如 contrast-invariant、modality-invariant、speckle、gradient prior、geometry-aided、biomechanically-constrained）。见第 1.2 节。

---

## 1. 归纳框架

### 1.1 领域先验放在哪里

| 位置 | 典型做法 | 代表条目（见第 2 节） |
|---|---|---|
| **A. 输入表示** | 把原图换成对模态不敏感的表示：手工结构特征（MIND、相位一致性、方向梯度）、可学的公共表示（对比学习）、SAR 取对数 / 极化分解 / 复数输入 | MIND、CoMIR、ContraReg、RIFT、CFOG、CV-FCN |
| **B. 网络结构** | 两支是否共享权重（伪孪生）；把梯度、相位等手工先验做成网络里的固定或可学的滤波层；粗到细级联；与融合 / 翻译网络联合；加入几何约束层（用回归得到的仿射约束稠密流） | Hughes 2018、SOMA、GDROS、Recursive Cascade、Zampieri 2018、MURF |
| **C. 训练数据：增强 / 仿真 / 合成** | 模态翻译生成配对（光学→伪 SAR、可见光→伪红外、MegaDepth→多模态）；物理仿真（DEM 渲染光照、DSM 仿真 SAR、相机噪声模型）；按领域规律随机合成（随机标签图→随机外观）；领域特有增强（染色颜色抖动、斑点噪声） | MINIMA、MatchAnything、XoFTR、Merkle 2018、SynthMorph、Geo-LoFTR、HORUS、Tellez 2019 |
| **D. 损失 / 自监督信号** | 换成跨模态相似度（MIND、NMI、可微分 MI、SSIM / 结构特征）；学出来的相似度度量；利用噪声统计做自监督（实部虚部独立、多时相）；困难负样本 | DRMIME、Simonovsky 2016、MU-Net、MERLIN、SAR2SAR、Hughes 2018b |
| **E. 输出表示与形变模型** | 仿射 / 单应 → 网格单应 → B 样条控制点 → 稠密位移场 → 微分同胚（速度场积分）→ 隐式神经表示；正则：光滑、逆一致、可逆、局部刚性、滑动界面；关键点 + 闭式解（仿射 / TPS） | VoxelMorph、Dalca 2018、GradICON、DLIR、Nie 2022、KeyMorph、IDIR、Kheil 2025 |
| **F. 训练流程** | 先合成有真值的预训练再在真实数据上无监督微调；粗到细分阶段；超参条件化（一个网络覆盖多个正则强度）；推理时逐对优化；多次投票 | HyperMorph、DeepHistReg、SEAMLeSS、IDIR |

### 1.2 论文通常怎么包装这类贡献

以下是从各条目的标题和摘要归纳出来的写法，不是某篇论文的原话。

- **「通用方法在本领域失效 + 原因是某条领域性质」开头。** 例：SOMA 摘要先说深度方法在 SAR-光学上表现不好，再指出梯度线索在手工描述子里一直有效却很少被深度模型用到；GDROS 列出「非线性辐射差异、几何畸变、噪声差异」三个性质；Geo-LoFTR 指出「光照差异大」。
- **把性质写进方法名或标题。** contrast-invariant（SynthMorph）、modality invariant（MINIMA）、geometry-preserving（Arar 2020）、geometry-guided（GDROS）、feature gradient enhanced（SOMA）、biomechanically-constrained（Kheil、Spine）、physical noise model（HORUS）、environment-driven（同上）。
- **把先验做成一个可插拔模块，并用消融证明它起作用。** 例：SOMA 的「梯度增强模块」与「全局-局部仿射-流匹配器」两个模块；Spine 论文强调两个刚性惩罚可以接到别的 DL 配准方法上。
- **「免标注 / 免真实数据」也是常见卖点。** SynthMorph 标题即「without acquired images」，SEAMLeSS、MERLIN、SAR2SAR、VoxelMorph 都把「不需要真值对应 / 干净图」写在摘要里。
- **「任务互相促进」的包装**：配准与融合（RFNet、MURF、SuperFusion）、配准与翻译（Arar、UMF-CMGR）联合训练，宣称两边都变好。
- **配一个领域数据集或基准一起发。** SEN1-2、ANHIR、ACROBAT、Learn2Reg、MINIMA 的多模态数据集，常常与方法一起构成贡献。

---

## 2. 逐条列表（按领域）

每条：出处 / 针对的数据特性 / 机制。【辅助】= 需要影像以外的数据；【摘要】= 只看了摘要。

### 2.1 医学多模态配准（含形变场输出）

| # | 出处 | 针对的数据特性 | 机制 |
|---|---|---|---|
| M1 | VoxelMorph: A Learning Framework for Deformable Medical Image Registration，IEEE TMI 2019，[arXiv:1809.05231](https://arxiv.org/abs/1809.05231)，代码 [voxelmorph](https://github.com/voxelmorph/voxelmorph) | 解剖结构的非刚性形变；没有形变真值 | 网络直接输出稠密位移场，用空间变换层 warp 后算图像相似度，加位移场梯度的光滑项，完全无监督。可选地用分割标签做辅助监督（【辅助】标签）。 |
| M2 | Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration，MICCAI 2018，[arXiv:1805.04605](https://arxiv.org/abs/1805.04605) | 解剖拓扑不应撕裂或折叠 | 网络输出速度场，用 scaling-and-squaring 积分成微分同胚形变，保证可逆、无折叠，同时给出不确定性。 |
| M3 | SynthMorph: learning contrast-invariant registration without acquired images，IEEE TMI 2022，[arXiv:2004.10282](https://arxiv.org/abs/2004.10282) | 不同 MR 序列 / 模态对比度差异大，配对数据少 | 随机生成标签图，再给每个标签随机赋强度、加模糊和偏置场，合成「同结构、外观随机」的图像对；只在合成图上训练，用标签重合度做损失，获得对对比度不敏感的配准。 |
| M4 | Weakly-Supervised CNNs for Multimodal Image Registration（MR-TRUS），MedIA 2018，[arXiv:1807.03361](https://arxiv.org/abs/1807.03361) | MR 与超声外观无法直接比较 | 不比较图像强度，只用两侧对应的解剖标签（腺体、标志点）的重合度监督形变场。【辅助】需要分割标签。 |
| M5 | Unsupervised Deformable Registration for Multi-Modal Images via Disentangled Representations，IPMI 2019，[arXiv:1903.09331](https://arxiv.org/abs/1903.09331) | 多模态外观差异 | 把图像拆成与模态无关的「形状」编码和与模态有关的「外观」编码，在形状空间里做单模态配准。 |
| M6 | A Deep Metric for Multimodal Registration，MICCAI 2016，[arXiv:1609.05396](https://arxiv.org/abs/1609.05396) | 手工相似度（MI 等）在 T1-T2 之类跨模态上不够好 | 用已对齐数据训练一个 CNN 判断两块是否对齐，把它当相似度度量嵌入传统优化。【辅助】需要已对齐的训练对。 |
| M7 | ContraReg: Contrastive Learning of Multi-modality Unsupervised Deformable Image Registration，MICCAI 2022，[arXiv:2206.13434](https://arxiv.org/abs/2206.13434) | 跨模态相似度难以手工设计 | 用块级对比学习把两种模态投影到共享特征空间，在这个空间里算相似度来训练形变网络。 |
| M8 | MIND: Modality independent neighbourhood descriptor，MedIA 2012，[DOI 10.1016/j.media.2012.05.008](https://doi.org/10.1016/j.media.2012.05.008) | 跨模态强度关系非线性 | 用每个像素与邻域块的自相似模式作描述子（自相似对模态不敏感）。本身不是 DL，但被大量 DL 配准方法用作损失（如 Learn2Reg 的参赛方法）。 |
| M9 | A Deep Learning Framework for Unsupervised Affine and Deformable Image Registration（DLIR），MedIA 2019，[arXiv:1809.06130](https://arxiv.org/abs/1809.06130) | 先有全局错位，再有局部形变 | 级联：先一个网络回归仿射，再多级网络输出 B 样条控制点位移，逐级无监督训练。 |
| M10 | Recursive Cascaded Networks for Unsupervised Medical Image Registration，ICCV 2019，[arXiv:1907.12353](https://arxiv.org/abs/1907.12353) | 大形变一步回归不准 | 同一个子网络递归多次，每次预测剩余的小形变，只在最后的 warp 结果上算损失。 |
| M11 | HyperMorph: Amortized Hyperparameter Learning for Image Registration，IPMI 2021，[arXiv:2101.01035](https://arxiv.org/abs/2101.01035) | 光滑项强度对不同数据最优值不同 | 用超网络让配准网络以正则权重为条件，一个模型覆盖整段正则强度，测试时再选。 |
| M12 | Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration，[arXiv:1809.03443](https://arxiv.org/abs/1809.03443)；GradICON，CVPR 2023，[arXiv:2206.05897](https://arxiv.org/abs/2206.05897) | A→B 与 B→A 应当互逆 | 同时预测双向形变并惩罚两者复合偏离恒等（GradICON 惩罚复合的 Jacobian 偏离单位阵），作为正则替代光滑项。 |
| M13 | TransMorph，MedIA 2022，[arXiv:2111.10480](https://arxiv.org/abs/2111.10480) | 形变需要长程上下文 | 用 Swin Transformer 编码器替换 CNN，输出位移场或微分同胚场。 |
| M14 | DiffuseMorph，ECCV 2022，[arXiv:2112.05149](https://arxiv.org/abs/2112.05149) | 形变的连续变化 | 用扩散模型的得分函数来生成形变场，可在固定图与运动图之间连续插值。 |
| M15 | Implicit Neural Representations for Deformable Image Registration（IDIR），MIDL 2022，[PMLR v172](https://proceedings.mlr.press/v172/wolterink22a.html) | 逐对形变精细、无需训练集 | 每对图单独拟合一个带周期激活的小 MLP 表示形变（坐标→位移），加正则，推理时优化。 |
| M16 | A Robust and Interpretable Deep Learning Framework for Multi-modal Registration via Keypoints（KeyMorph），MedIA 2023，[arXiv:2304.09941](https://arxiv.org/abs/2304.09941) | 多模态下直接回归变换不稳健、难解释 | 网络在两图各输出一组顺序对应的关键点，再用闭式最小二乘求仿射或 TPS，整体端到端训练。 |
| M17 | SAM: Self-supervised Learning of Pixel-wise Anatomical Embeddings，IEEE TMI 2022，[arXiv:2012.02383](https://arxiv.org/abs/2012.02383) | 解剖位置在不同人之间对应 | 像素级对比学习（同一图的不同增强版本），得到每个像素的「解剖位置」嵌入，再用最近邻找对应点或初始化配准。 |
| M18 | uniGradICON: A Foundation Model for Medical Image Registration，[arXiv:2403.05780](https://arxiv.org/abs/2403.05780) | 每个解剖部位都要单独训练 | 在多部位、多数据集上训练一个通用配准模型，必要时逐对做实例优化。 |
| M19 | Biomechanical Constraints Assimilation in Deep-Learning Image Registration，[arXiv:2504.05444](https://arxiv.org/abs/2504.05444)【摘要】 | 骨头应局部刚性、器官之间会滑动 | 在形变场上算应变张量，对骨头区域惩罚其特征值（局部刚性），在器官交界处允许切向滑动。【辅助】需要区域分割。 |
| M20 | Weakly-supervised Biomechanically-constrained CT/MRI Registration of the Spine，MICCAI 2022，[arXiv:2205.07568](https://arxiv.org/abs/2205.07568)【摘要】 | 椎骨在 CT-MR 间只有刚体运动 | 两个刚性惩罚（刚性 Dice 与刚性形变场损失）约束每节椎骨的形变接近刚体。【辅助】需要运动图的标签。 |
| M21 | Learn2Reg 挑战，IEEE TMI 2022，[arXiv:2112.04489](https://arxiv.org/abs/2112.04489) | — | 多任务配准基准（含 MR-CT、MR-US），参赛方法中 MIND 特征 + 实例优化、粗到细是常见做法；可作为「医学配准里哪些先验有效」的总览。 |

### 2.2 地球遥感光学-SAR

| # | 出处 | 针对的数据特性 | 机制 |
|---|---|---|---|
| R1 | Identifying Corresponding Patches in SAR and Optical Images with a Pseudo-Siamese CNN，IEEE GRSL 2018，[arXiv:1801.08467](https://arxiv.org/abs/1801.08467) | 两种传感器成像机理不同，共享权重不合适 | 两支结构相同但**不共享权重**（伪孪生），后接融合层做「是否对应」二分类。 |
| R2 | The SEN1-2 Dataset for Deep Learning in SAR-Optical Data Fusion，ISPRS Annals 2018，[arXiv:1807.01569](https://arxiv.org/abs/1807.01569) | 缺少大规模配对数据 | 发布 Sentinel-1/2 配对块数据集，后续很多光学-SAR 匹配方法在其上训练评测。 |
| R3 | Exploring the Potential of Conditional Adversarial Networks for Optical and SAR Image Matching，IEEE JSTARS 2018，[DOI 10.1109/JSTARS.2018.2803212](https://doi.org/10.1109/JSTARS.2018.2803212) | 外观差异让 NCC/SIFT 失效 | 用条件 GAN 从光学块生成「像 SAR」的块，再用 NCC、SIFT、BRISK 等单模态方法与真实 SAR 匹配。 |
| R4 | Mining Hard Negative Samples for SAR-Optical Image Matching Using GANs，Remote Sensing 2018，[DOI 10.3390/rs10101552](https://doi.org/10.3390/rs10101552)【摘要】 | 高分 SAR 难区分的错误匹配多，标注少 | 用对抗训练的 VAE 生成逼真的 SAR 块，当作困难负样本训练匹配网络，降低误匹配率。 |
| R5 | A deep learning framework for matching of SAR and optical imagery，ISPRS J. P&RS 2020，[DOI 10.1016/j.isprsjprs.2020.09.012](https://doi.org/10.1016/j.isprsjprs.2020.09.012)【摘要】 | 稀疏、不可靠的对应 | 三步：先预测可匹配区域，再在特征空间做互相关得到对应，最后用网络剔除外点。 |
| R6 | Fast and Robust Matching for Multimodal Remote Sensing Image Registration（CFOG），IEEE TGRS 2019，[arXiv:1808.06194](https://arxiv.org/abs/1808.06194) | 非线性辐射差异但结构相似 | 逐像素的方向梯度通道特征 + 频域快速模板匹配。非 DL，但「方向梯度结构特征」后来被当作 DL 损失或模块的先验。 |
| R7 | RIFT: Multi-modal Image Matching Based on Radiation-invariant Feature Transform，IEEE TIP 2020，[arXiv:1804.09493](https://arxiv.org/abs/1804.09493) | 辐射差异 | 用相位一致性（对亮度和对比度不敏感）检测特征点，用最大索引图做描述子。非 DL，是「相位先验」的来源。 |
| R8 | MU-Net: A Multiscale Unsupervised Network for Remote Sensing Image Registration，ISPRS Archives 2022，[链接](https://isprs-archives.copernicus.org/articles/XLIII-B3-2022/537/2022/)，代码 [MU-Net](https://github.com/yeyuanxin110/MU-Net)【摘要】 | 无真值；多模态 | 多尺度级联网络直接回归变换参数，无监督，损失基于结构相似度设计，以适应光学-SAR 等多模态。 |
| R9 | Coarse to fine non-rigid registration: a chain of scale-specific neural networks for multimodal image alignment with application to remote sensing，ECCV 2018，[arXiv:1802.09816](https://arxiv.org/abs/1802.09816) | 地图与影像之间的局部非刚性错位 | 每个尺度一个全卷积网络直接输出位移场，逐尺度串起来，从粗到细。 |
| R10 | GDROS: A Geometry-Guided Dense Registration Framework for Optical-SAR Images under Large Geometric Transformations，IEEE TGRS 2025，[arXiv:2511.00598](https://arxiv.org/abs/2511.00598)【摘要】 | 辐射差异、几何畸变、噪声差异 | CNN-Transformer 提特征、多尺度 4D 相关体迭代细化出稠密光流；再用最小二乘回归一个仿射并把它施加到最终光流上，限制光流发散。 |
| R11 | SOMA: Feature Gradient Enhanced Affine-Flow Matching for SAR-Optical Registration，AAAI 2026，[DOI 10.1609/aaai.v40i12.37939](https://doi.org/10.1609/aaai.v40i12.37939)【摘要】 | 深度特征缺少梯度结构信息 | 多尺度、多方向梯度滤波 + 注意力和重建，把梯度先验注入深度特征；全局仿射 + 局部光流的粗到细匹配器。 |
| R12 | MINIMA: Modality Invariant Image Matching，CVPR 2025，[arXiv:2412.19412](https://arxiv.org/abs/2412.19412) | 多模态配对数据少 | 用生成模型把 RGB 匹配数据集（带几何真值）翻译成红外、深度、SAR 风格等多种模态，在合成的跨模态对上训练通用匹配器（AnyMatch 系列权重即属此类）。 |
| R13 | MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training，[arXiv:2501.07556](https://arxiv.org/abs/2501.07556) | 同上 | 多种来源的数据（多视图、视频、图像翻译生成的跨模态对）大规模预训练匹配器，以覆盖医学、遥感等跨模态任务。 |
| R14 | As if by magic: self-supervised training of deep despeckling networks with MERLIN，IEEE TGRS 2022，[arXiv:2110.13148](https://arxiv.org/abs/2110.13148) | SAR 斑点噪声；没有干净图 | 利用单视复数 SAR 的实部和虚部斑点相互独立：用一部分预测另一部分，构成自监督去斑损失。需要复数数据。 |
| R15 | SAR2SAR: a semi-supervised despeckling algorithm for SAR images，IEEE JSTARS 2021，[arXiv:2006.15037](https://arxiv.org/abs/2006.15037) | 同上 | 先用合成斑点训练，再用同一地点的多时相 SAR 对做 Noise2Noise 式训练，并补偿时相变化。【辅助】需要多时相数据。 |
| R16 | Pixel-Wise PolSAR Image Classification via a Novel Complex-Valued Deep FCN，[arXiv:1909.13299](https://arxiv.org/abs/1909.13299) | 极化 SAR 的复数和相位信息 | 网络直接在复数域运算（复数卷积、激活），保留相位。不是配准，是「极化 / 复数输入」特化的代表。 |
| R17 | Stereo Radargrammetry Using Deep Learning-Based Image Matching with Fine-Tuned Model on Synthetic Aperture Radar Images，Remote Sensing 2026，[DOI 10.3390/rs18101662](https://doi.org/10.3390/rs18101662)【摘要，二手检索摘要】 | SAR 侧视几何导致的非线性畸变；光学预训练迁移差 | 用参考 DSM + SAR 成像模型自动生成带真值的 SAR 块训练集，微调 RoMa；在斜距图上直接匹配以免重采样损失。（SAR-SAR 立体，不是光学-SAR。）【辅助】DSM + 成像几何。 |
| R18 | GC-Pix2Pix 等「以投影局部入射角为条件」的 SAR→光学翻译，ISPRS Archives 2026，[PDF](https://isprs-archives.copernicus.org/articles/XLIX-B3-2026/163/2026/isprs-archives-XLIX-B3-2026-163-2026.pdf)【摘要，二手检索摘要】 | 侧视几何使 SAR 与光学错位，像素级损失受害 | 把局部入射角图作为生成器条件，帮助保留结构。【辅助】需要入射角 / DEM。 |
| R19 | Deep Learning Meets SAR（综述），IEEE GRSM，[arXiv:2006.10027](https://arxiv.org/abs/2006.10027)；Multi-Resolution SAR and Optical Remote Sensing Image Registration Methods: A Review，[arXiv:2502.01002](https://arxiv.org/abs/2502.01002) | — | 两篇综述，可当作 SAR 领域 DL 特化手段（斑点、复数、几何）和光学-SAR 配准方法的索引。 |

### 2.3 红外-可见光

| # | 出处 | 针对的数据特性 | 机制 |
|---|---|---|---|
| I1 | XoFTR: Cross-modal Feature Matching Transformer，CVPR Workshop 2024，[arXiv:2404.09692](https://arxiv.org/abs/2404.09692) | 热红外与可见光纹理和亮度差异大，配对数据少 | 两步：先用跨模态掩码图像建模预训练；再在可见光数据上用「伪热红外」增强（模拟热图外观）做有真值的微调；结构上在 LoFTR 基础上加强细级匹配。 |
| I2 | Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration（UMF-CMGR），IJCAI 2022，[arXiv:2205.11876](https://arxiv.org/abs/2205.11876) | 两种模态不能直接比；源图有轻微非刚性错位 | 先把可见光翻译成伪红外，再在红外与伪红外之间做单模态配准，输出形变场，然后融合。 |
| I3 | Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation，CVPR 2020，[arXiv:2003.08073](https://arxiv.org/abs/2003.08073) | 翻译网络可能偷偷改几何 | 翻译网络与配准网络联合训练，要求「先翻译再变形」与「先变形再翻译」结果一致，迫使翻译只改外观不改几何。 |
| I4 | SuperFusion，IEEE/CAA JAS 2022，[DOI 10.1109/JAS.2022.106082](https://doi.org/10.1109/JAS.2022.106082)【摘要】 | 错位导致融合伪影 | 配准网络估计双向形变场，用光度约束和端点约束训练；配准与融合对称执行，加单模态一致性约束。 |
| I5 | RFNet，CVPR 2022，[CVF 页](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_RFNet_Unsupervised_Network_for_Mutually_Reinforcing_Multi-Modal_Image_Registration_and_CVPR_2022_paper.html)；MURF，[IEEE 10145843](https://ieeexplore.ieee.org/document/10145843)【摘要】 | 配准误差与融合质量互相影响 | 粗到细配准（共享信息提取 → 多尺度粗配准 → 细配准与融合），融合结果反馈回来提升配准；MURF 用对比学习做特征提取。 |
| I6 | ReCoNet，ECCV 2022，[ECVA 页](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3864_ECCV_2022_paper.php)【摘要】 | 源图轻微错位、计算量大 | 加形变模块显式补偿几何畸变，注意力抑制重影，循环结构减少计算。 |
| I7 | SA-DNet: A on-demand semantic object registration network adapting to non-rigid deformation，[arXiv:2210.09900](https://arxiv.org/abs/2210.09900)【摘要】 | 非刚性形变；只关心部分语义目标 | 按语义目标做匹配和非刚性配准，只对需要的目标区域求形变。 |

### 2.4 显微 / 病理

| # | 出处 | 针对的数据特性 | 机制 |
|---|---|---|---|
| H1 | CoMIR: Contrastive Multimodal Image Representation for Registration，NeurIPS 2020，[arXiv:2006.06325](https://arxiv.org/abs/2006.06325) | 明场与二次谐波等显微模态外观完全不同 | 用已对齐的配对训练两个网络，以对比损失（InfoNCE）把两种模态映射到相似的稠密表示，并加旋转等变约束；之后用常规单模态配准（如 α-AMD、SIFT）在表示上配准。也在遥感数据上验证。【辅助】训练需要已对齐配对。 |
| H2 | Is Image-to-Image Translation the Panacea for Multimodal Image Registration? A Comparative Study，PLOS ONE 2022，[arXiv:2103.16262](https://arxiv.org/abs/2103.16262) | — | 在显微、遥感等数据上系统比较 GAN 翻译式与 CoMIR 式公共表示；翻译式在不少数据上效果不稳定。可作为「翻译 vs 公共表示」路线的对照证据。 |
| H3 | DRMIME: Differentiable Mutual Information and Matrix Exponential for Multi-Resolution Image Registration，MIDL 2020，[arXiv:2001.09865](https://arxiv.org/abs/2001.09865) | 跨模态强度关系未知 | 用神经网络估计互信息（MINE）作为可微损失；变换用矩阵指数参数化仿射 / 单应，多分辨率逐对优化。 |
| H4 | DeepHistReg，Computer Methods and Programs in Biomedicine 2021，[DOI 10.1016/j.cmpb.2020.105799](https://doi.org/10.1016/j.cmpb.2020.105799)；后续 DeeperHistReg，[arXiv:2404.14434](https://arxiv.org/abs/2404.14434)【摘要】 | 切片任意旋转、染色不同、组织局部变形 | 分级：先粗对齐（含旋转搜索）→ 仿射网络 → 无监督非刚性网络，逐级训练。 |
| H5 | Quantifying the effects of data augmentation and stain color normalization in CNNs for computational pathology，MedIA 2019，[arXiv:1902.06543](https://arxiv.org/abs/1902.06543) | 不同实验室染色颜色差异 | 系统比较 HSV、HED 颜色空间的颜色扰动增强与颜色归一化，发现颜色增强对跨中心泛化最关键。（是分类任务，代表「领域特有增强」。） |
| H6 | ANHIR 挑战，IEEE TMI 2020，[DOI 10.1109/TMI.2020.2986331](https://doi.org/10.1109/TMI.2020.2986331)；ACROBAT 2022 挑战，[arXiv:2305.18033](https://arxiv.org/abs/2305.18033) | — | 不同染色切片的配准基准，两者都以人工标志点误差评测；参赛方法的分级流程可作索引。 |
| H7 | Siamese Encoding and Alignment by Multiscale Learning with Self-Supervision（SEAMLeSS），[arXiv:1904.02643](https://arxiv.org/abs/1904.02643)；后续 Nature Communications 2023 管线，[bioRxiv](https://www.biorxiv.org/content/10.1101/2022.03.25.485816v1) | 连续电镜切片之间的形变不能用仿射近似；裂纹、折痕、缺失 | 孪生编码 + 多尺度位移场，用自监督（warp 后相似度 + 光滑）训练；后续加「多次预测投票」应对伪影，并把块级形变带衰减地组合成全局对齐。 |

### 2.5 行星科学

| # | 出处 | 针对的数据特性 | 机制 |
|---|---|---|---|
| P1 | Geometry-aided Vision-based Localization of Future Mars Helicopters in Challenging Illumination Conditions（Geo-LoFTR），[arXiv:2502.09795](https://arxiv.org/abs/2502.09795)【摘要】 | 机载图与轨道图光照差异大、地表缺纹理 | 以 LoFTR 为基础、加入几何信息的匹配器；用真实轨道图（正射影像 + 地形）搭仿真器，渲染不同太阳角下的训练图像。【辅助】需要 DEM。 |
| P2 | Lunar Crater Identification via Deep Learning（DeepMoon），Icarus 2019，[arXiv:1803.02192](https://arxiv.org/abs/1803.02192) | 陨石坑是主要地物，圆形 | U-Net 在 DEM 上分割坑缘，再用模板匹配提取圆、与目录比对。【辅助】输入是 DEM。 |
| P3 | Lunar Terrain Relative Navigation Using a CNN for Visual Crater Detection（LunaNet），ACC 2020，[arXiv:2007.07702](https://arxiv.org/abs/2007.07702) | 光照变化下手工坑检测不稳 | CNN 在光学图上检测陨石坑，把检测到的坑与目录中的坑匹配作为导航特征，进入卡尔曼滤波。【辅助】需要陨石坑目录和位置先验。 |
| P4 | Automated Crater Detection from Co-registered Optical Images, Elevation Maps and Slope Maps using Deep Learning，[arXiv:2012.15281](https://arxiv.org/abs/2012.15281) | 单一光学图上坑的可见度受光照影响 | 光学 + DEM + 坡度图多通道输入 Mask R-CNN 检测坑。【辅助】DEM。 |
| P5 | Extreme Low-Light Environment-Driven Image Denoising over Permanently Shadowed Lunar Regions with a Physical Noise Model（HORUS），CVPR 2021，[CVF PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Moseley_Extreme_Low-Light_Environment-Driven_Image_Denoising_Over_Permanently_Shadowed_Lunar_Regions_CVPR_2021_paper.pdf) | 永久阴影区极暗，相机噪声复杂 | 用相机的物理噪声模型 + 真实暗帧噪声样本合成训练数据，用光线追踪挑选合适的训练场景，并以相机温度等元数据为条件。【辅助】相机元数据。 |
| P6 | Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data，[arXiv:2509.04775](https://arxiv.org/abs/2509.04775)【摘要】 | 跨传感器月面影像；极区光照、阴影 | 评测 SIFT/ASIFT/AKAZE/RIFT2/SuperGlue，并提出强度归一化与阴影校正预处理；极区光照下传统方法明显变差。 |
| P7 | Automatic Registration of Mini-RF S-Band Level-1 Data，IGARSS 2024，[IEEE 10642532](https://ieeexplore.ieee.org/document/10642532)；A Robust Digital Elevation Model-Based Registration Method for Mini-RF/Mini-SAR Images，Remote Sensing 2025，[DOI 10.3390/rs17040613](https://doi.org/10.3390/rs17040613)【摘要】 | Mini-RF 成像偏移、侧视几何 | 非 DL：用 LOLA DEM 仿真 SAR 图，与真实 Mini-RF 做 NCC 求偏移，再与光学 / DEM 对齐。【辅助】DEM。本次没有找到用 DL 做 Mini-RF 与光学配准的工作。 |

### 2.6 跨领域可借的输出表示（一般视觉）

| # | 出处 | 针对的数据特性 | 机制 |
|---|---|---|---|
| G1 | Unsupervised Deep Homography，IEEE RA-L 2018，[arXiv:1709.03966](https://arxiv.org/abs/1709.03966) | 没有单应真值 | 网络回归 4 角点位移，经可微 DLT 和空间变换层 warp，用光度误差无监督训练。 |
| G2 | Content-Aware Unsupervised Deep Homography Estimation，ECCV 2020，[arXiv:1909.05983](https://arxiv.org/abs/1909.05983) | 场景中有不服从全局单应的区域 | 网络同时学一张权重掩码，在特征空间算损失时降低这些区域的权重，避免被它们带偏。 |
| G3 | Depth-Aware Multi-Grid Deep Homography Estimation with Contextual Correlation，IEEE TCSVT 2022，[arXiv:2107.02524](https://arxiv.org/abs/2107.02524) | 场景有深度变化，单一单应不够 | 输出网格上每个顶点的位移（网格单应 / 分块变换），介于全局变换和稠密光流之间，并加形状保持约束。 |
| G4 | Pyramid Stereo Matching Network，CVPR 2018，[arXiv:1803.08669](https://arxiv.org/abs/1803.08669) | 已校正立体对的对应只在一个方向上 | 只沿水平方向建 1D 代价体（视差），而不是 2D 搜索。是「已知位移主要沿某一轴」这一先验在结构上的典型写法。 |

---

## 3. 按数据特性反查（不评估是否适用）

| 数据特性 | 出现过的特化手段 | 条目 |
|---|---|---|
| 跨模态外观 / 辐射差异 | 不共享权重；翻译成对方模态；对比学习公共表示；形状 / 外观解耦；梯度、相位、自相似等手工结构先验做输入、模块或损失；可微 MI；学出来的相似度 | R1, R3, I2, I3, H1, M7, M5, R6, R7, R11, M8, H3, M6 |
| 配对 / 真值数据少 | 生成式翻译合成跨模态训练对；随机合成外观；伪模态增强；掩码预训练；困难负样本生成；只用合成数据训练 | R12, R13, M3, I1, R4 |
| 局部非刚性形变 | 稠密位移场 + 光滑；微分同胚；逆一致；B 样条控制点；网格单应；级联 / 递归细化；隐式表示逐对优化；全局仿射 + 稠密流细化 | M1, M2, M12, M9, G3, M10, R9, M15, R10, R11, H7 |
| 部分结构应保持刚性 | 应变张量的局部刚性损失；逐结构刚性惩罚；关键点 + 闭式仿射 / TPS | M19, M20, M16 |
| 部分区域不服从全局变换 | 学一张掩码降低这些区域的损失权重；按语义目标分别配准 | G2, I7 |
| 位移主要沿一个方向 | 1D 代价体（立体视差） | G4 |
| 斑点 / 传感器噪声 | 利用噪声统计做自监督（实部虚部独立、多时相）；物理噪声模型合成 | R14, R15, P5 |
| 光照 / 阴影（行星） | 用正射图 + DEM 按太阳角渲染训练对；强度归一化与阴影校正 | P1, P6 |
| 侧视几何 / 地形畸变（SAR） | 用 DSM/DEM 仿真 SAR 生成训练对；以入射角为条件的生成；DEM 仿真 SAR 做 NCC | R17, R18, P7 |
| 极化 / 复数 | 复数网络；复数实部虚部分开利用 | R16, R14 |
| 陨石坑为主的地物 | 以坑为检测和匹配单元（与目录比对） | P2, P3, P4 |
| 颜色 / 染色差异 | 领域专用颜色空间增强 | H5 |

## 4. 依赖辅助数据的条目汇总

| 条目 | 需要的辅助数据 |
|---|---|
| M1（可选分割监督）、M4、M19、M20 | 解剖分割标签 |
| M6、H1 | 已对齐的配对（训练时） |
| R15 | 多时相 SAR |
| R14 | 单视复数 SAR（实部、虚部） |
| R17、R18、P7 | DSM / DEM、SAR 成像几何或入射角 |
| P1、P2、P4 | DEM |
| P3 | 陨石坑目录、位置先验 |
| P5 | 相机元数据（温度等） |
