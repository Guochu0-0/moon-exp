# 出处（#119）

检索和阅读日期：2026-10-06。除特别注明外，都读的是 arXiv PDF（`https://arxiv.org/pdf/<id>`，经 pdftotext 抽文本），节号、表号、图号按 PDF。原报告（`research/offline-pseudo-label`）已读过的 22 篇不再重读，引用其结论时以原报告的 sources.md 为准。

## 稠密匹配器的无监督 / 自监督训练

| 文献 | 链接 | 读了什么 |
|---|---|---|
| WarpC: Warp Consistency for Unsupervised Learning of Dense Correspondences（Truong et al., ICCV 2021） | [arXiv:2104.03308](https://arxiv.org/abs/2104.03308) | §3.3–3.6（一致性图、三种 bipath 约束的退化与偏置分析、W-bipath 损失式 (7)–(9)、warp 监督与自适应权重、W 的采样）；§4.1 表 1–3；§4.3 语义匹配无监督微调与表 5；附录 A.1（偏置推导） |
| PWarpC: Probabilistic Warp Consistency（Truong et al., CVPR 2022） | [arXiv:2203.04279](https://arxiv.org/abs/2203.04279) | §4.3（无匹配状态与负样本对）；§5.4 表 2（消融；最小熵、最大分数对照） |
| GLU-Net（Truong et al., CVPR 2020） | [arXiv:1912.05524](https://arxiv.org/abs/1912.05524) | §3.5 训练（单图合成仿射 / TPS / 单应 warp 的自监督） |
| PDC-Net+（Truong et al., TPAMI） | [arXiv:2109.13912](https://arxiv.org/abs/2109.13912) | §3.4（自监督不确定度的扰动）、§3.5（自监督数据生成、独立运动物体） |
| SMURF（Stone et al., CVPR 2021） | [arXiv:2105.07014](https://arxiv.org/abs/2105.07014) | §3.2.1（自监督：干净全图的最终输出监督裁剪 + 扰动输入；去掉掩码；扰动含随机擦除）；§5.2 表 6 |
| ARFlow（Liu et al., CVPR 2020） | [arXiv:2003.13045](https://arxiv.org/abs/2003.13045) | 摘要、§3（空间 / 外观 / 遮挡变换）、§4 表 4 |
| Semi-Supervised Learning of Optical Flow by Flow Supervisor（Im et al., ECCV 2022） | [arXiv:2207.10314](https://arxiv.org/abs/2207.10314) | 摘要、§3.3（参数分离、传入学生输出、推理不用监督者）、§4.2 表 1 与图 3 |
| DCFlow: Rethinking Unsupervised Cross-modal Flow Estimation | [arXiv:2509.24423](https://arxiv.org/abs/2509.24423) | 摘要、§3.3（去掉残差最大 ρ% 像素的损失）、§3.4（已知仿射下的跨模态一致性）、表 2 |
| CoLReg（Wei et al.） | [arXiv:2505.22000](https://arxiv.org/abs/2505.22000) | 只读了 Highlights 和摘要（扩散模型模态转换 + 自监督中间配准 + 蒸馏）；已在 `docs/research/label-free-reward` §2.F 讨论过 |
| iMatching（Zhan et al., ECCV 2024） | [arXiv:2312.02141](https://arxiv.org/abs/2312.02141) | §4 表 1–3（iDKM；KITTI360 上 DKM 已饱和），补充材料的 COLMAP 有监督对照 |
| RoMa（Edstedt et al., CVPR 2024） | [arXiv:2305.15404](https://arxiv.org/abs/2305.15404) | 已由 `research/roma-finetune` 读过；本文只用它的损失和代码事实 |
| RoMa v2（Edstedt et al.） | [arXiv:2511.15706](https://arxiv.org/abs/2511.15706) | §3.2（冻结 DINOv3）、§3.3（细化器、预测协方差、EMA 消除亚像素偏置，decay 0.999）、§3.5（分辨率）、表 9、表 10 |
| DKM（Edstedt et al., CVPR 2023） | [arXiv:2202.00667](https://arxiv.org/abs/2202.00667) | 只用于引用追踪 |

## 基础模型的自适应、参数高效微调、权重平均

| 文献 | 链接 | 读了什么 |
|---|---|---|
| WeSTAR: Weakly-Supervised Adaptation with Regularization（Huang et al.） | [arXiv:2511.14238](https://arxiv.org/abs/2511.14238) | 摘要、「Domain Adaptation via Self-Training」（EMA 老师、强弱扰动）、LoRA 与 Weight Regularization 两段、实现细节（EMA 0.996、LoRA r 8）、表 4、表 5、图 4 |
| Surgical Fine-Tuning（Lee et al., ICLR 2023） | [arXiv:2210.11466](https://arxiv.org/abs/2210.11466) | 摘要、§2.1 图 2 与表 1、§2.2（无监督自适应下的同一结论） |
| UniMatch V2（Yang et al.） | [arXiv:2410.10777](https://arxiv.org/abs/2410.10777) | 摘要、引言（特征层 Dropout 最有效）、§3.3.2（互补 channel dropout，0.5）、§4.4.3 表 7（微调 vs 冻结 DINOv2） |
| SEAR（Skorokhodov et al.） | [arXiv:2603.18774](https://arxiv.org/abs/2603.18774) | 摘要、§4.1（LoRA 加在 AA 模块，DINOv2 冻结）、§5.2（标签来源）、§6.8 表 2–3、§6.9 |
| ExPLoRA（Khanna et al.） | [arXiv:2406.10973](https://arxiv.org/abs/2406.10973) | 摘要、算法 1、图 2、§5、表 1–3（解冻块数、LoRA 秩、GPU 小时） |
| LoRA（Hu et al., ICLR 2022） | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | §3（adapter 的推理延迟）、§4.1（合并后无额外推理延迟） |
| Model Soups（Wortsman et al., ICML 2022） | [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) | 摘要、§2（uniform / greedy soup，Recipe 1）、§3.3.1 表 3（误差壁垒与高学习率） |
| WiSE-FT（Wortsman et al., CVPR 2022） | [arXiv:2109.01903](https://arxiv.org/abs/2109.01903) | 摘要、§3（权重插值）、§4（α = 0.5 接近最优）、附录 B 表 3 |
| AdaBN（Li et al.） | [arXiv:1603.04779](https://arxiv.org/abs/1603.04779) | 摘要、§3.2 |
| TENT（Wang et al., ICLR 2021） | [arXiv:2006.10726](https://arxiv.org/abs/2006.10726) | 摘要、§3（熵最小化 + BN 仿射参数） |
| Kendall & Gal, What Uncertainties Do We Need（NeurIPS 2017） | [arXiv:1703.04977](https://arxiv.org/abs/1703.04977) | §2.2、§3.2（异方差回归即损失衰减） |
| Debiased Self-Training（Chen et al., NeurIPS 2022） | [arXiv:2202.07136](https://arxiv.org/abs/2202.07136) | 摘要、§4.1（独立伪标签头需要有标签数据） |

## 对应、匹配的其他新工作（2025–2026）

| 文献 | 链接 | 读了什么 |
|---|---|---|
| Self-Supervised Spatial Correspondence Across Modalities（Shrivastava & Owens） | [arXiv:2506.03148](https://arxiv.org/abs/2506.03148) | 摘要、引言、§3 开头（跨模态 + 模态内对比随机游走；只用跨模态收敛差） |
| DIY-SC: Learning Semantic Correspondence from Pseudo-Labels（Dünkel et al.） | [arXiv:2506.05312](https://arxiv.org/abs/2506.05312) | 摘要、§3.2–3.3（链式伪标签、松弛循环一致性过滤、轻量 adapter） |
| Jamais Vu（Mariotti et al.） | [arXiv:2506.08220](https://arxiv.org/abs/2506.08220) | 摘要、引言 |
| Semi-Supervised Multiscale Matching for SAR-Optical Image（S²M²-SAR） | [arXiv:2508.07812](https://arxiv.org/abs/2508.07812) | 摘要（需有标注子集） |
| DistillMatch | [arXiv:2509.16017](https://arxiv.org/abs/2509.16017) | 摘要（有监督 + GAN 合成数据） |
| Self-Supervised Contrastive Embedding Adaptation for Endoscopic Image Matching | [arXiv:2512.10379](https://arxiv.org/abs/2512.10379) | 摘要与方法概要（冻结 DINOv2 + 一层 Transformer，新视角合成生成伪真值）；与本题关系远，未列入正文 |
| SemiMatch（Kim et al., CVPR 2022） | [arXiv:2203.16038](https://arxiv.org/abs/2203.16038) | 摘要（弱扰动出伪标签、强扰动学生）；与 UniMatch V2 同类，未单列 |
| CRFT、RBE-Flow、SOMA | [2604.05689](https://arxiv.org/abs/2604.05689)、[2606.30492](https://arxiv.org/abs/2606.30492)、[2511.13168](https://arxiv.org/abs/2511.13168) | 只读摘要：有监督跨模态配准网络，与无标注训练无关 |
| Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration? | [arXiv:2604.10217](https://arxiv.org/abs/2604.10217) | 已由 `research/roma-finetune` 读过：只评测，不微调 |

## 引用追踪

用 Semantic Scholar Graph API（`/paper/arXiv:<id>/citations`，取 2024 年及以后）：

| 被引文献 | 结果 |
|---|---|
| SCENES（2401.10886） | 2 条（FAR 2403.03221；一篇立体校正），与无标注匹配训练无关 |
| EYOC（2403.03532） | 约 30 条，几乎都是点云配准；其中无监督点云配准（2409.07558、2411.01870、2609.15228）依赖点云几何，未读 |
| 2607.10082 | 无记录 |
| iMatching（2312.02141） | 接口返回 404 |
| PWarpC（2203.04279） | 约 20 条，筛出 DIY-SC、Jamais Vu、2506.03148 |
| WarpC、PDC-Net+、DKM、RoMa、MINIMA（2412.19412）、MatchAnything（2501.07556） | 合计约 800 条，按题目关键词（self-supervised、unsupervised、adapt、fine-tun、pseudo、LoRA、SAR、cross-modal 等）筛出约 120 条，其中与「无标注训练或微调」直接有关的只有上表几篇；其余多为有监督的多模态匹配网络 |

关键词检索（WebSearch）：「self-supervised fine-tuning dense matcher RoMa unlabeled」「LoRA DINOv2 dense correspondence」「semi-supervised optical flow fine-tuning unlabeled」「optical SAR matching self-supervised pseudo labels」「unsupervised domain adaptation feature matching 2025 2026」「self-training dense regression confirmation bias foundation model」「warp consistency remote sensing multimodal」等，补充了 WeSTAR、ExPLoRA、DCFlow、S²M²-SAR、Flow Supervisor。

## 仓库内

- `runs/M/notes.md`、`runs/M/exp.toml`（RoMa 伪标签与类 RIPE 的全部数字）。
- `runs/P/notes.md`、`runs/Q/notes.md`、`runs/C/notes.md`（LoFTR 两条路线）。
- `runs/E1/notes.md`、`runs/E2/notes.md`（有标注训练的参考水平）。
- `finetune/models/roma.py`（训练分辨率 560；可训参数为 decoder，可选 VGG）；`finetune/` 中没有权重 EMA。
- `origin/research/offline-pseudo-label:docs/research/offline-pseudo-label/{README,sources}.md`（原报告）。
- `origin/research/roma-finetune:docs/research/roma-finetune/README.md`（RoMa 损失、864 一遍未训练、显存估算、VGG 学习率为解码器 1/20）。
- `docs/research/label-free-reward/README.md`（§2.D 变换一致性对平移偏差没有信号，§2.F 模态转换类方法）。
- 按票面要求，`runs/E3`–`runs/E6` 未作为依据。
