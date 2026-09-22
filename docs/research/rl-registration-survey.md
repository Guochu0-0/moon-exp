# 强化学习用于图像配准 / 特征匹配：文献现状调研

> 调研日期：2026-09-23
> 说明：**「读到的」= 我直接读到了论文原文/摘要原句；「推断」= 我的判断，已显式标注。**
> IEEE Xplore 直连被 AWS WAF 拦截，本报告通过 r.jina.ai 阅读器代理、arXiv、CVF Open Access、Semantic Scholar / OpenAlex / Crossref API 取证。

---

## A. 用户给的那篇论文：已定位

**IEEE 文档号 11445864 = RIPE: Reinforcement Learning on Unlabeled Image Pairs for Robust Keypoint Extraction**

| 项 | 内容 |
|---|---|
| 作者 | Johannes Künzel, Anna Hilsmann, Peter Eisert（Fraunhofer HHI + 洪堡大学柏林） |
| 会议 | **ICCV 2025**（IEEE/CVF International Conference on Computer Vision），2025-10 |
| 预印本 | arXiv:2507.04839（2025-07-07 v1，2025-07-14 v2） |
| 代码 | https://github.com/fraunhoferhhi/RIPE |

来源：[IEEE Xplore 11445864 页面](https://ieeexplore.ieee.org/document/11445864/)（经阅读器代理读到标题为 "RIPE: ... | IEEE Conference Publication"，面包屑显示 "2025 IEEE/CVF International C..."）、[arXiv:2507.04839](https://arxiv.org/abs/2507.04839)、[CVF 开放获取 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Kunzel_RIPE_Reinforcement_Learning_on_Unlabeled_Image_Pairs_for_Robust_Keypoint_ICCV_2025_paper.pdf)、[GitHub](https://github.com/fraunhoferhhi/RIPE)。

### A.1 它解决什么任务

**稀疏关键点的检测 + 描述子学习**（SuperPoint / DISK / ALIKED / DeDoDe 这一赛道），不是配准变换回归。目标是：只用「这两张图是不是同一场景」的**二值标签**训练一个关键点提取器，彻底摆脱 depth / pose / 人工单应增强。

> 原文："RIPE requires only a binary label indicating whether paired images represent the same scene." —— [arXiv abs](https://arxiv.org/abs/2507.04839)

### A.2 RL 的 state / action / reward

读自 [arXiv HTML 全文 §3.3](https://arxiv.org/html/2507.04839v2)：

- **State**：输入图像 `I` 本身（"the input image I representing the state"）。
- **Policy**：encoder-decoder（hourglass）网络 `d_θ(e_θ(I))` 输出 logit 热图 `H`；热图切成 `m×m` 规则网格，**每个 cell 内的 logit 构成一个 categorical 分布**，从中采样恰好一个关键点位置。再对 logit 过 sigmoid 得到「接受指示 `a_i = σ(l_i)`」，最终概率 `p_i = σ(l_i)·p̂_i`（采样概率 × 保留概率）。
- **Action**：**关键点的位置**（"the keypoint localization corresponds to an action"）。一张图 C 个 cell 各采一个点，等价于采样多条 trajectory 来近似期望。
- **Reward**：两图描述子做 **L2 互最近邻（mutual nearest neighbor）** → **8 点法 + RANSAC 估基础矩阵 F** 过滤 → 通过极线约束的匹配对给 `sign(λ)·ρ`，其余为 0。`λ=+1`（同场景）给正奖励，`λ=−1`（不同场景）给同等负奖励；作者说这种对称信号 "beneficial during training"。
- **算法**：**REINFORCE**（Williams），`ĝ = Σ_c ∇_θ(log p_c ⊕ log p'_c) R`，`⊕` 为外和（把两图所有 cell 组合展成 C×C）。正文未提 critic 或 baseline。
- **辅助损失**：可微的 margin 式描述子损失 `L_desc`（正对 `max(0, μ + δ₊ − δ_h)`，负对 `max(0, μ − δ₊)`，N 为 RANSAC 内点数）。

### A.3 reward 是否依赖 ground truth

**不依赖任何几何真值**。这是全文最核心的卖点：不需要 depth、不需要 relative pose、不需要像素级对应、不需要人工单应增强。**唯一监督是「这对图是不是同一场景」的二值标签**——负样本对可随意构造，正样本对在 place-recognition / 自动驾驶数据里本就存在。

作者明确对比前作（[§2 "SOTA Limitations"](https://arxiv.org/html/2507.04839v2)）：

> "RL also remains underutilized, as depth (DISK), pose (Reinforced Feature Points) or artificial augmentations (DEAL) are still required."

### A.4 作者自己给的「为什么用 RL 而不是直接梯度下降」

两条，都写在正文里：

1. **关键点选择本身不可微**："to address the non-differentiable nature of the keypoint selection process, we introduce a probabilistic formulation for keypoint selection via Reinforcement Learning (RL)"。argmax / top-k 取点没有梯度，概率化 + REINFORCE 是标准解法（DISK、Reinforced Feature Points 同理由）。
2. **reward 链路里的 MNN + RANSAC 是黑箱**："The reward is computed using mutual nearest-neighbor estimation and RANSAC filtering, which, **despite being non-differentiable**, are used here solely for the reward calculation, and thus do not require gradient computation."

**→ 这正是「用不可微 reward 论证 RL 必要性」的教科书案例，见 C.1。**

### A.5 局限（作者自报 + 我读到的）

- Aachen Day-Night 上加太多 Tokyo 24/7 数据反而变差：Tokyo 数据 "lack of viewpoint variability"，导致 RIPE "struggles to learn to cope with the strong viewpoint variations"。最佳配比 80% MegaDepth + 20% Tokyo。
- reward 基于「单一对极几何下的内点数」，理论上存在**坍缩到对极点（epipole）**的风险。补充材料 §6.3 专门讨论：MegaDepth 的对极点通常在图外；ACDC / Tokyo 常在图内，靠 (a) 描述子损失、(b) 网格化采样强制空间均匀 来避免，"we never observed it in any of our experiments"——**这是经验性论证，不是理论保证**。
- 代码库明确写 "Dense outputs are not supported"（Glue Factory 集成部分）——**RIPE 是纯稀疏方法**。
- 性能是 "competitive / on par with SOTA"，不是碾压：MegaDepth-1500 AUC@10° ≈ 68，HPatches 1px ≈ 38（README 报告值）。

### A.6 对用户课题的直接相关性（推断）

> **【推断】** RIPE 的 reward = 「MNN + RANSAC 估 F 后的内点数」。用户做**月球光学-SAR、输出仿射**，可把 F 换成**仿射/单应 + RANSAC**，reward 变成「仿射 RANSAC 内点数」，整个 RL 框架可原样搬。正负样本对在月球数据上也容易构造（同区域 vs 不同区域）。**新颖性风险**在于：这样做基本等于「RIPE 换个变换模型 + 换个数据集」，顶会审稿人会问增量在哪里。
> **【推断】** 真正的空白在 C.2 —— RIPE / DISK / RFP / DEAL **全部是稀疏关键点**，没人把这套 RL 用在稠密/半稠密匹配器上。

---

## B. 这条路线的整体版图

### B.1 医学影像（RL 配准的发源地，2017 起）

| 论文 | 任务 | RL 算法 | State / Action / Reward | 需要真值？ | 变换自由度 | 局限 |
|---|---|---|---|---|---|---|
| **Liao et al., AAAI 2017, "An Artificial Agent for Robust Image Registration"** [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/11230) | 3D/3D 刚体配准（脊柱 CT–CBCT、心脏 CT–CBCT） | **不是真 RL**：DRL 的 MDP 框架 + **greedy 监督学习（DSL）** 训练 Q 网络 | State = 当前变换 `T_t`，观测 = 差值图 `d_t = I_r − T_t∘I_f`；Action = 12 个离散动作（±1mm 平移 / ±1° 旋转）；Reward = `r = D(T_g,T_t) − D(T_g, a_t∘T_t)`，即**到真值变换的距离改善量**，命中容差 ε=0.5 时 bonus R=10 | **是，强依赖 `T_g`** | 刚体 6-DoF，探索范围 ±30mm / ±30° | 作者明说用 DSL 取代 DRL 探索是为了效率（图 4：同样 1 天训练 DSL 显著优于 DRL）；无收敛的理论保证；真值靠 ICP + 专家手工编辑 |
| **Krebs et al., MICCAI 2017, "Robust Non-rigid Registration Through Agent-Based Action Learning"** [[Springer]](https://link.springer.com/chapter/10.1007/978-3-319-66182-7_40) | 前列腺 MR 非刚体配准 | agent-based action learning（Q 学习族） | Agent 在**统计形变模型（SDM）的低维参数空间**中探索，动作 = 增减 SDM 系数 | 是（需先用训练数据建 SDM，动作监督来自已知形变） | 低维非刚体（SDM 主成分） | 受限于 SDM 表达能力；器官特定（ROI-specific）。**注：HAL 有反爬，仅读到摘要级信息** |
| **Sun et al., ACCV 2018, "Robust Multimodal Image Registration using Deep Recurrent RL"** [[arXiv:2002.03733]](https://arxiv.org/abs/2002.03733) | 多模态（MR–CT）配准 | 异步 RL（policy + value 网络）+ lookahead inference | State = 图像对特征（含循环结构）；Action = 变换调整；Reward = 自定义 | 摘要未明说；**【推断】** 与其 MedIA 姊妹篇一致，靠 landmark 误差 | 摘要未明说 | 摘要未列 |
| **Hu, Luo, Wang, Sun, … Wu, Medical Image Analysis 2021（online 2020）, "End-to-end multimodal image registration via reinforcement learning"** [[DOI]](https://doi.org/10.1016/j.media.2020.101878) | 鼻咽癌 CT–MR 多模态配准 | **异步 RL**（actor-critic 族），ConvLSTM 提时空特征、**隐式学相似性度量**；测试期 Monte-Carlo rollout 前瞻 | Action = 变换动作序列；**Reward = landmark error 驱动的自定义函数** | **是，依赖 landmark 真值** | 序列式刚体/仿射动作 | 依赖标注 landmark；仅单一解剖部位验证 |
| **Zhang et al., IJCARS 2026, "Warm-started RL for iterative 3D/2D liver registration"** [[DOI]](https://doi.org/10.1007/s11548-026-03653-9) [[arXiv:2604.10245]](https://arxiv.org/abs/2604.10245) | 术前 CT ↔ 腹腔镜视频 3D/2D 配准 | 离散动作 RL，**特征编码器从有监督位姿估计网络 warm-start** | State = CT 渲染 + 腹腔镜帧的共享编码特征；Action = 6-DoF 刚体离散动作 + **停止动作** | 是（warm-start 来自监督位姿网络） | 刚体 6-DoF | TRE 15.70 ± 8.18（仍较大） |
| **Choi et al., Pattern Recognition 2025, "Deep RL for efficient registration between intraoral-scan meshes and CT images"** [[DOI]](https://doi.org/10.1016/j.patcog.2025.111502) | 口扫网格 ↔ CT | DRL | 未详读 | 未详读 | 刚体 | 未详读 |
| **MorphSeek, arXiv:2511.17392 (2025)** [[arXiv]](https://arxiv.org/abs/2511.17392) | 3D 形变配准（OASIS / LiTS / Abdomen MR-CT） | **GRPO**（Group Relative Policy Optimization），多轨迹采样稳定训练 | **在 latent 特征空间**加 **stochastic Gaussian policy head** 建模 latent 分布；无监督 warm-up + 弱监督 GRPO 微调 | **弱监督**（不需要体素级标注，但需弱标签如分割/Dice） | 稠密形变场（经 latent 参数化） | 摘要未列显式局限；作者称先前 RL 方法受限于 "coarse, low-dimensional representations" |

**医学线的共同模式（读到的事实）**：几乎全部是「**agent 迭代调整变换参数**」范式——action 是变换增量，reward 是「离真值更近了吗」。**因此绝大多数依赖 ground-truth 变换或 landmark**。

### B.2 遥感 / SAR

| 论文 | 任务 | RL 算法 | State / Action / Reward | 需要真值？ | 变换自由度 | 局限 |
|---|---|---|---|---|---|---|
| **Zhang R., Wang G., Zhang Z., Xu H., Remote Sensing 15(20):4941, 2023, "A Sub-Second Method for SAR Image Registration Based on Hierarchical Episodic Control"（即 OptionEM 那篇）** [[MDPI]](https://www.mdpi.com/2072-4292/15/20/4941) [[DOI]](https://doi.org/10.3390/rs15204941) | SAR–SAR 同模态配准，端到端直接输出配准图 + **仿射矩阵** | **OptionEM**：分层 RL（Option 框架）+ **Episodic Memory**；网络 = Transformer 特征层 + correlation 层 | **State = 参考图与待配准图的灰度图对**；**Action = 16 个离散仿射动作**（左/右/上/下平移 1px、10px；顺/逆时针旋转 1°、10°；缩放 0.1、0.01 放大/缩小）**+ 1 个 trigger 停止动作**，分粗细两个精度尺度；**Reward = 变换后 DoG 显著点与参考显著点集之间欧氏距离 D 的改善量** | **是。** 原文：显著点参考集 `P_G` "derived from the **ground truth** of the sensed/reference image"，每个 episode 用真值变换的逆矩阵生成畸变显著点集。等价于 Liao 2017 的 landmark reward，只是用 DoG 点代替解剖 landmark | **仿射**（平移 + 旋转 + 缩放，无剪切） | 作者 Discussion 自报：起伏/复杂地形失败率高；训练集由 self-learning 方案生成，**当待配准图像差异超出训练数据覆盖范围就失败**，需先做几何粗配准 / DEM / GCP；扩大训练分布又会增加误配风险、降低稳定性 |
| **Liu R., Zhang H., IGARSS 2024, "Optical and SAR Image Registration with Deep Reinforcement Learning"** [[DOI]](https://doi.org/10.1109/IGARSS53475.2024.10642618) | **光学–SAR 跨模态配准**（与用户课题最接近的遥感 RL 工作） | DRL（4 页短文，摘要未点名具体算法） | **Action = 四个预定义角点在给定搜索空间内的位移方向与幅度**；由 4 个角点位移解算 **affine 或 homography 矩阵** | 摘要未明说；**【推断】** 从 "explores the possible displacement directions and magnitudes … in a specific search space" 看，reward 极可能仍需真值角点位置 | **仿射 / 单应** | 摘要自报问题："insufficient training data and mismatching in local regions"；**结果仅报告 "improvements in the visualization of image registration"——无定量 SOTA 对比** |

**遥感线目前很薄**：只找到上述两篇明确以 RL 为训练范式的配准工作。其余 RL + SAR 的工作（如 [Jiang et al., JSTARS 2024, Azimuth-Aware DRL for Active SAR ATR](https://doi.org/10.1109/JSTARS.2024.3363915)）属于目标识别/主动感知，与配准无关。

### B.3 通用视觉：稀疏特征点的 RL 训练（RIPE 的直系家族）

| 论文 | RL 算法 | Reward | 需要真值？ | 局限 |
|---|---|---|---|---|
| **Bhowmik, Gumhold, Rother, Brachmann, CVPR 2020, "Reinforced Feature Points"** [[arXiv:1912.00623]](https://arxiv.org/abs/1912.00623) [[CVF]](https://openaccess.thecvf.com/content_CVPR_2020/html/Bhowmik_Reinforced_Feature_Points_Optimizing_Feature_Detection_and_Description_for_a_CVPR_2020_paper.html) | REINFORCE，把**整条视觉流水线（匹配 + RANSAC + 位姿估计）当黑箱** | 下游任务指标（相对位姿误差） | **是，需要已知位姿** | 只在已有检测器（SuperPoint）上微调；reward 方差大 |
| **Tyszkiewicz, Fua, Trulls, NeurIPS 2020, "DISK: Learning local features with policy gradient"** [[arXiv:2006.13566]](https://arxiv.org/abs/2006.13566) | policy gradient（概率化检测 + 概率化匹配） | **正确匹配的数量**（正确性由已知深度/相对位姿判定） | **是，需要 depth** | 依赖 SfM 重建的深度，训练数据受限于 MegaDepth 类数据集 |
| **Potje et al., "DEAL"** | 沿用 DISK 的 RL 框架 + Warp Module | 同 DISK | **需要人工形变增强** | 非刚体鲁棒性提升但仍依赖合成变形（该条转引自 RIPE 的 Related Work） |
| **RIPE, ICCV 2025**（见 A 节） | REINFORCE | MNN + RANSAC 估 F 后的内点数，正负对对称 | **否，只需二值场景标签** | 纯稀疏；大视点变化时受训练数据分布限制 |

> 这个谱系的演进方向非常清楚：**reward 所需监督逐级变弱** —— 位姿（RFP 2020）→ 深度（DISK 2020）→ 人工形变（DEAL）→ **只要二值场景标签（RIPE 2025）**。

### B.4 半稠密 / detector-free + 策略梯度：只有一篇

| 论文 | 说明 |
|---|---|
| **Di, Liao, Zhou, Zhu, Zhang, Duan, Liu, Lu, Applied Intelligence 53, 2023, "FeMIP: detector-free feature matching for multimodal images with policy gradient"** [[DOI]](https://doi.org/10.1007/s10489-023-04659-5) | **detector-free**（LoFTR 式 coarse-to-fine：coarse matching module + fine regression module），多模态图像匹配。出版商摘要原句："uses the **principle of reinforcement learning to design a policy gradient method to improve the solution to the problem of discreteness in matching**"；"The coarse-to-fine module **automatically generates pixel-level labels** on the original image, enabling FeMIP to perform pixel-level matching on data with **only image-level labels**"。**注意**：Springer 全文付费墙、Semantic Scholar 摘要被出版商屏蔽，**我没有读到它的 state/action/reward 具体公式**。 |

---

## C. 三个决策性问题

### C.1 无 ground truth 时，现有工作的 reward 实际都用什么？

**读到的事实，按 reward 类型归类：**

| reward 类型 | 代表工作 | 是否需要几何真值 |
|---|---|---|
| **到真值变换 / landmark 的距离改善量** | Liao AAAI 2017、Hu et al. MedIA 2021、RS 2023 SAR-RL（DoG 显著点版）、Warm-started RL IJCARS 2026 | **需要** |
| **正确匹配数**（正确性由已知 depth/pose 判定） | DISK NeurIPS 2020 | **需要** |
| **下游任务指标**（位姿误差，RANSAC 当黑箱） | Reinforced Feature Points CVPR 2020 | **需要** |
| **几何一致性内点数**（MNN + RANSAC 估 F，只要二值场景标签） | **RIPE ICCV 2025** | **不需要** |
| **弱监督分割/Dice 信号** | MorphSeek 2025 (GRPO) | 弱标签 |

**关键发现（对用户最重要的一条）：**

> **在图像配准/匹配的 RL 文献里，我没有找到任何一篇用 MI / NCC / 相位一致性 / CFOG / 判别器分数 作为 RL reward 的工作。**

原因是结构性的，不是偶然：**MI、NCC、CFOG、判别器分数全都是可微（或可做成可微）的**。既然可微，直接当 loss 反传即可——这正是 MU-Net、VoxelMorph 系、[CoLReg](https://arxiv.org/abs/2505.22000)（明确用 NCC 作相似性度量）等无监督配准网络的做法，**根本不需要 RL**。RL 在这里没有优势，只会引入梯度方差。

> **【推断】** 因此若用户打算「无标签 + MI/NCC/CFOG 作 reward + RL」，审稿人会直接问：**为什么不直接把它当 loss 梯度下降？** 这个问题必须在方法设计阶段就答掉。

**有没有工作用不可微 reward 并以此论证 RL 的必要性？—— 有，而且这是这条线的主流论证：**

1. **RIPE (ICCV 2025)**：两条不可微性——(a) 关键点选择（采样/top-k）不可微；(b) reward 链路里的 **MNN + RANSAC 估基础矩阵**不可微。原文："which, despite being non-differentiable, are used here solely for the reward calculation, and thus do not require gradient computation"。
2. **Reinforced Feature Points (CVPR 2020)**：把**整条流水线（匹配 + RANSAC + 位姿求解）当黑箱**，"We overcome the discrete nature of key point selection and descriptor matching using principles from reinforcement learning"。
3. **DISK (NeurIPS 2020)**："Local feature frameworks are difficult to learn in an end-to-end fashion, **due to the discreteness inherent to the selection and matching of sparse keypoints**"。
4. **FeMIP (2023)**：policy gradient 是为了 "improve the solution to the problem of **discreteness in matching**"。

> **【推断】** RL 在配准/匹配领域**唯一站得住脚的论证是「不可微」，不是「无标签」**。无标签本身用无监督 loss（NCC/MI/CFOG）就能解决。用户若走 RL，必须把**不可微环节**做成故事核心——例如「RANSAC 内点数」「离散 top-k 匹配选择」「置信度阈值化」这类真的没有梯度的东西。

### C.2 有没有把 RL 用在稠密 / 半稠密匹配网络上的工作？

**基本是空白，只有一篇边缘案例。**

- **【读到的】唯一命中：FeMIP（Applied Intelligence 2023）**——detector-free（半稠密，LoFTR 式 coarse-to-fine），用 policy gradient 处理匹配的离散性。**但**：(a) 2023 年，不算近期；(b) 期刊为 Applied Intelligence，非顶会；(c) 我只读到摘要，**无法确认它的 policy gradient 是作用在稠密匹配置信矩阵上，还是只作用在某个离散选择步骤上**；(d) 做的是通用多模态图像匹配，不是光学-SAR 配准。
- **【读到的】MorphSeek (2025)** 用 GRPO 做**稠密形变配准**，但 policy 加在 **latent 特征空间**（Gaussian policy head），不是稠密匹配网络的匹配层，且是医学 3D 形变场，不是 2D 跨模态匹配。
- **【读到的】主流稀疏 RL 工作明确排斥稠密**：RIPE 代码库写明 "Dense outputs are not supported"；DISK / Reinforced Feature Points / DEAL 全是稀疏关键点。RIPE 的 Related Work 把 RoMa / Mast3r / DUSt3R（稠密）与 S2DNet / LoFTR / Efficient LoFTR（半稠密）单列一类，并指出它们**同样依赖预生成 3D 模型的深度信息**——**没有一篇用 RL 训练**。
- **【未找到】** 没有找到任何把 REINFORCE / PPO / GRPO 用在 **LoFTR / Efficient LoFTR / RoMa / DKM 这类稠密或半稠密匹配器**训练上的论文；也没有找到把 RL 用于**遥感稠密跨模态匹配**的工作。

> **→ 这确实是一个空白点。** **【推断】** 但要小心：空白可能因为「难做」而非「没人想到」——稠密匹配的 action 空间是 H×W×H×W 量级，REINFORCE 方差会爆炸。RIPE 的网格化采样（每 cell 一个点）正是在压 action 空间。用户若走这条路，**方差控制（learned baseline / GRPO 的 group-relative advantage / 分块采样）本身就该是一项技术贡献**，否则做不出来。

### C.3 跨模态（光学-SAR）配准里的「传统方法生成粗匹配 → 置信度筛选 → 迭代自训练」

**结论：这个精确形式的文献我没有找到；但周边有若干「伪标签自训练」工作。用户「没找到文献」的判断需要修正为「没有完全对应的，但有很近的邻居」。**

**【读到的】找到的具体文献：**

1. **Gai & Li, arXiv:2508.07812 (2025), "Semi-supervised Multiscale Matching for SAR-Optical Image"** [[arXiv]](https://arxiv.org/abs/2508.07812)
   —— **最接近用户设想的一篇**。对**无标注的光学-SAR 图像对生成伪真值相似度热图**，方式是 "combining both **deep and shallow level matching results**"（深层 + 浅层匹配结果融合），再把有标注与伪标注热图**一起训练**；另有无需真值的 cross-modality mutual independence loss。
   **差异**：伪标签来自网络自身的多尺度结果，不是来自传统手工方法（CFOG/HOPC/SAR-SIFT）；摘要**未描述显式的置信度筛选与多轮迭代自训练循环**。

2. **CoLReg — "Collaborative Learning for Unsupervised Multimodal Remote Sensing Image Registration"**, arXiv:2505.22000 / Information Fusion 2025 [[arXiv]](https://arxiv.org/abs/2505.22000) [[ScienceDirect]](https://www.sciencedirect.com/science/article/abs/pii/S1566253525008759)
   —— 三网络交替训练：(1) MIM 引导的条件扩散模型做跨模态图像翻译，生成模态一致的图像对；(2) 自监督中间配准网络，在翻译后的同模态对上用**精确位移标签**学变换；(3) **蒸馏的跨模态配准网络，用中间网络预测的伪标签监督**。三者交替优化、互相提升，"progressively reducing modality discrepancies, **enhancing the quality of pseudo-labels**"。
   **这就是「伪标签 + 迭代自训练」**，只是粗匹配来源是**扩散翻译 + 自监督网络**，不是传统方法。**注意**：文中 "mutual reinforcement" 意为「互相促进」，**不是强化学习**。

3. **ACAMatch**（Remote Sensing 17(14):2501, 2025, "Robust Optical and SAR Image Matching via Attention-Guided Structural Encoding and Confidence-Aware Filtering"）[[DOI]](https://doi.org/10.3390/rs17142501)
   —— 含 **confidence-aware filtering** 与 "automatically generated **pseudo-labels from transformed coordinates**" 的自监督策略。伪标签来自已知合成变换坐标（等价于人工增强），**不是从传统方法的粗匹配来的**。

4. **同模态 SAR / 多模态的自监督伪标签先例**（RS 2023 综述段落明确称前两篇为 "pseudo-label-generation method, eliminating the need for additional annotations"）：
   - Zou, Li, Zhang, **TGRS 2022**, "Self-Supervised SAR Image Registration With SAR-SuperPoint and Transformation Aggregation" [[DOI]](https://doi.org/10.1109/TGRS.2022.3210185)
   - Zhao, Zhang, Ding, **Int. J. Remote Sensing 43:915–931, 2022**, "Heterogeneous self-supervised interest point matching for **multi-modal** remote sensing image registration" [[DOI]](https://doi.org/10.1080/01431161.2021.2022240) ← **标题即多模态，最值得精读**
   - Mao et al., **TGRS 2023**, "Adaptive Self-Supervised SAR Image Registration with Modifications of Alignment Transformation" [[DOI]](https://doi.org/10.1109/TGRS.2023.3246964)

**【未找到】** 没有找到明确写成「用 CFOG / HOPC / 相位一致性 / SAR-SIFT 等**传统手工方法**先跑粗匹配 → 按置信度筛选 → 训练网络 → 用网络重新生成更好的匹配 → **多轮迭代**」的光学-SAR 配准论文。

> **【推断】为什么这个空位存在**：遥感光学-SAR 数据通常自带地理编码 / RPC / DEM，粗配准几乎免费，所以社区更倾向「合成变换生成精确伪标签」（ACAMatch 路线）或「跨模态翻译后自监督」（CoLReg 路线），而不是「从传统匹配器 bootstrap」。**对月球数据这个假设不成立**——月球影像定位精度差得多，这恰恰是用户课题可以成立的差异化理由，但需要在论文里**显式论证这个前提差异**。

---

## D. 关键判断

1. **A 篇已锁定：RIPE, ICCV 2025（arXiv:2507.04839）。** 它是目前「用 RL 训练稀疏匹配网络、且 reward 完全不需要几何真值」的最强、最新代表作。用户的直觉是对的——这条路子是通的，而且刚被顶会认可。

2. **RL 在这个领域的合法性来自「不可微」，不是来自「无标签」。** 这是全部文献最一致的信号（RIPE / DISK / RFP / FeMIP 四篇都这么论证）。**【推断】** 若把 RL 的卖点写成「因为没有标签所以用 RL」，会被一句「NCC/MI/CFOG 可微，直接当 loss 不行吗」打回。正确写法是：**我的 reward 里有 RANSAC 内点数 / 离散 top-k 选择 / 阈值化置信度，这些真的没有梯度。**

3. **「仿射 + RANSAC 内点数作 reward」是 RIPE 的直接可迁移版本。** **【推断】** 技术上几乎零风险（F 换成仿射即可），但**创新性风险高**——等于换变换模型 + 换数据集。要上顶会需要额外的实质贡献。

4. **真正的空白在「RL + 稠密/半稠密匹配」。** 【读到的】除 FeMIP（2023, Applied Intelligence, 仅摘要级证据）外未找到任何工作；RIPE 代码明确不支持稠密输出。**【推断】** 空白伴随真实技术障碍（action 空间爆炸 → REINFORCE 方差），所以**方差控制方案本身应成为贡献**（GRPO 的 group-relative advantage、分块/网格采样、learned baseline 都是候选）。这是我认为最值得投入的方向。

5. **遥感 RL 配准的现有工作质量普遍不高。** 【读到的】RS 2023 那篇 reward 依赖 DoG 显著点的真值、且自报复杂地形失败；IGARSS 2024 光学-SAR 那篇只有 4 页、只报告「视觉上有改善」。**这意味着遥感侧 baseline 很弱——好消息是好超越，坏消息是没有强 baseline 可对标，用户可能需要自己搭一个 RIPE 的遥感版本作对照组。**

6. **伪标签自训练在跨模态配准里不是空白，只是形式不同。** 【读到的】最该精读的三篇：`arXiv:2508.07812`（SAR-光学半监督伪标签热图）、`arXiv:2505.22000` CoLReg（伪标签 + 交替自训练）、Zhao et al. IJRS 2022（多模态自监督兴趣点匹配）。**【推断】** 用户可主张的差异点是「**月球影像缺乏可靠先验地理编码**，因此必须从传统匹配器 bootstrap」——但这个前提必须写清楚并用实验支撑。

7. **一条措辞陷阱**：CoLReg 论文中的 "mutual reinforcement learning process" 指三个网络互相促进，**不是强化学习**。检索时容易误命中。

---

## 附：检索覆盖与未解决项

- **被拦截的路径**：IEEE Xplore 直连（AWS WAF，HTTP 202 + JS 挑战）、DuckDuckGo HTML（CAPTCHA）、MDPI 直连（403）、Inria HAL（Anubis 反爬）。改用 r.jina.ai 阅读器代理成功读取 IEEE 标题页与 MDPI 全文。
- **未能读到全文的**：FeMIP（Springer 付费墙，S2 摘要被出版商屏蔽，仅有出版商页面摘要片段）；Krebs 2017（仅摘要级）；IGARSS 2024（仅 Semantic Scholar 摘要）。这三条的 state/action/reward 细节已标注为「摘要未明说」。
- **建议用户自行补做的检索**：Zhao et al., IJRS 2022（多模态自监督兴趣点匹配）全文；FeMIP 全文（确认其 policy gradient 具体加在哪一层）。
