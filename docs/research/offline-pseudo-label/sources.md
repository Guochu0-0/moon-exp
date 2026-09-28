# 出处（#52）

检索和阅读日期：2026-09-28。除特别注明外，都读的是 arXiv PDF（`https://arxiv.org/pdf/<id>`，经 pdftotext 抽文本），节号、表号、图号按 PDF。

## 起点

| 文献 | 链接 | 读了什么 |
|---|---|---|
| Li, Xu, Yang, Yu, Xia. Learning Cross-view Visual Geo-localization without Ground Truth | [arXiv:2403.12702](https://arxiv.org/abs/2403.12702)（只有 v1，2024-03-19） | **LaTeX 源**（`https://arxiv.org/e-print/2403.12702`）全文：§3.4 EMPL（式 (4) argmax 伪标签、0.1 阈值、InfoNCE M-step）；§3.5 AIC（reverter + L2 重建）；§3.6 算法 1；§4.2 实现（线性 adapter，Adam 1e-3，T = 60 / 10，每次迭代的采样数）；表 1、表 6（EMPL / Residual / AIC / 有监督 adapter 消融）、表 7；§4.6 图 13（AIC 分析）；§6.2、§6.3 局限。`main.bbl` 中的相关引用：Xu et al. 2022 Bayesian Pseudo Labels、Neal & Hinton 1998、SwAV、CLIP-Adapter、SVL-Adapter |

## 引用 2403.12702 的工作（经 Semantic Scholar API `/paper/arXiv:2403.12702/citations` 查得，约 37 条，筛出与伪标签自训练有关的）

| 文献 | 链接 | 读了什么 |
|---|---|---|
| STEAM: Stable Self-Training with Elastic Matching and Adaptive Purification | [arXiv:2607.09057](https://arxiv.org/abs/2607.09057) | 方法全节（Elastic Matching 的 Top-K 与 margin 动态阈值，式 (8)–(12)；Adaptive Purification 的式 (13)–(16)）；实现细节（K = 50、阈值 0.05 余弦衰减、M_e = 1、100 epoch） |
| DMNIL: Without Paired Labeled Data: End-to-End Self-Supervised Learning for Drone-View Geo-Localization | [arXiv:2502.11381](https://arxiv.org/abs/2502.11381) | 只读了引言和方法概要（聚类伪标签、记忆库动量 0.1 / 0.3、Top-k 邻域一致性） |
| UniABG | [arXiv:2511.12054](https://arxiv.org/abs/2511.12054) | 只读了方法概要（两阶段：DBSCAN 视图内伪标签 → 异构图过滤的跨视图关联） |
| 其余引用（SGMS-Fusion、GLEAM、Video2BEV、BGG、InfoGeo 等） | — | 只看了标题，与伪标签自训练无关，未读 |

## 匹配 / 配准领域的自训练

| 文献 | 链接 | 读了什么 |
|---|---|---|
| SCENES（Kloepfer, Henriques, Campbell） | [arXiv:2401.10886](https://arxiv.org/abs/2401.10886) | §4.2 bootstrap 实现（100 万对离线估 F、≥ 100 匹配 / ≥ 20 内点、1:1 混 MegaDepth）；§4.3 消融（位姿扰动 1–2°）；§5 局限（「entrenches any biases」）与未来工作（迭代重估 F）；附录（Aachen、EuRoC 上的 bootstrap；基础模型失败的对帮助最弱） |
| SGP: Self-supervised Geometric Perception（Yang, Dong, Carlone, Koltun, CVPR 2021） | [arXiv:2103.03114](https://arxiv.org/abs/2103.03114) | 算法 1（teacher / student / verifier，retrain 与 finetune）；Remark 1；§5.1–5.3（verifier 阈值、T = 10、每轮步数、retrain 与 finetune 对比、关掉 verifier 的结果）；补充材料里非鲁棒 teacher 的实验 |
| EYOC: Extend Your Own Correspondences（Liu et al., CVPR 2024） | [arXiv:2403.03532](https://arxiv.org/abs/2403.03532) | §4.1 渐进距离扩展（B：1 → 30）；§4.2 EMA 式 (1)；§4.3 空间过滤；§4.4 推测配准；表 2 消融（λ、d_thresh、s_thresh，去掉 PD 就失败）；§5.4 续训与从头训对比；附录中的上限与误差累积讨论 |
| Label-Free Target-Domain Adaptation for Unconstrained Event-Image Feature Matching via Dual-Stage Distillation（MM'26） | [arXiv:2607.10082](https://arxiv.org/abs/2607.10082) | §3.4 极线引导自蒸馏（EMA teacher、单应增强、双阈值一致性、极线几何置信度）；§4.2 实现（1.5 px / 5 px，EMA 0.999 / 0.9999）；§4.4 消融（w/o Conf.、Desc. Sim. Conf.、Full） |

## 域自适应 / 半监督里的伪标签自训练

| 文献 | 链接 | 读了什么 |
|---|---|---|
| CBST（Zou et al., ECCV 2018） | [arXiv:1810.07911](https://arxiv.org/abs/1810.07911) | §3.2、§4.1（轮的定义）；§4.2–4.3（比例 p 从 20% 起每轮 +5%、上限 50%，类平衡 k_c）；§4.4 空间先验；附录（每轮 2 epoch，第 3 轮） |
| CRST（Zou et al., ICCV 2019） | [arXiv:1908.09822](https://arxiv.org/abs/1908.09822) | 引言、§4（LR / MR 正则）、实现（3 轮，每轮 2 epoch） |
| ProDA（Zhang et al., CVPR 2021） | [arXiv:2101.10979](https://arxiv.org/abs/2101.10979) | §4.1 原型去噪（式 (3)–(7)，动量 0.9999）；§4.2 结构学习；§4.3 蒸馏；§5.3 表 3–5、图 3–4（固定 p_t,0 与动态标签的对比） |
| DAFormer（Hoyer et al., CVPR 2022） | [arXiv:2111.14887](https://arxiv.org/abs/2111.14887) | §3.1 自训练（式 (2)–(5)，质量估计 τ = 0.968）；§3.3 FD 与 warmup；实现细节（α = 0.99 → 0.999，40k 步）；§4 中关于伪标签漂移的讨论 |
| Mean Teacher（Tarvainen & Valpola, NeurIPS 2017） | [arXiv:1703.01780](https://arxiv.org/abs/1703.01780) | EMA 衰减的取值与 ramp-up（正文与附录） |
| Noisy Student（Xie et al., CVPR 2020） | [arXiv:1911.04252](https://arxiv.org/abs/1911.04252) | 算法 1；过滤与平衡（§3，以及附录 A.2 Study #5：置信度 > 0.3）；§4.1 表 6（噪声消融）；§4.2 表 7（3 轮迭代）；Finding #1–#8；附录 A.2 |
| Curriculum Labeling（Cascante-Bonilla et al., AAAI 2021） | [arXiv:2001.06001](https://arxiv.org/abs/2001.06001) | 算法 1（百分位阈值、每轮从头训）；§5.4 表 4–7 |
| ST++（Yang et al., CVPR 2022） | [arXiv:2106.05095](https://arxiv.org/abs/2106.05095) | §3.2–3.4（耦合问题、强增强、ckpt 稳定性打分式 (4)、先取 top 50%）；实现与消融（可靠图像的比例） |
| Soft Teacher（Xu et al., ICCV 2021） | [arXiv:2106.09018](https://arxiv.org/abs/2106.09018) | §3.2–3.3（前景阈值 0.9、box jittering 方差式 (7)–(9)）；实现（N_jitter = 10、阈值 0.02） |
| Unbiased Teacher（Liu et al., ICLR 2021） | [arXiv:2102.09480](https://arxiv.org/abs/2102.09480) | §3.1–3.3（burn-in、置信度 0.7、不对框回归用伪标签） |
| Arazo et al., Pseudo-Labeling and Confirmation Bias（IJCNN 2020） | [arXiv:1908.02983](https://arxiv.org/abs/1908.02983) | §III（软标签按 epoch 更新、mixup、每 batch 最少有标签样本数 k）；§IV 表（k = 16 等） |
| Bayesian Pseudo Labels（Xu et al., MICCAI 2022）/ Expectation Maximization Pseudo Labels | [arXiv:2208.04435](https://arxiv.org/abs/2208.04435)、[arXiv:2305.01747](https://arxiv.org/abs/2305.01747) | 2305.01747 的 §3（E-step / M-step 对应）、§4（变分学阈值）、图 4–5（学到的阈值约 0.8） |

## 噪声标签学习

| 文献 | 链接 | 读了什么 |
|---|---|---|
| Co-teaching（Han et al., NeurIPS 2018） | [arXiv:1804.06872](https://arxiv.org/abs/1804.06872) | 算法 1（R(T)、T_k）；§3 Q1–Q2（小损失、为什么要两个网络） |
| DivideMix（Li, Socher, Hoi, ICLR 2020） | [arXiv:2002.07394](https://arxiv.org/abs/2002.07394) | §3.1 co-divide（GMM、阈值 τ）；warmup 与置信惩罚；实现与附录（τ = 0.5、warmup epoch 数） |
| ELR（Liu et al., NeurIPS 2020） | [arXiv:2007.00151](https://arxiv.org/abs/2007.00151) | 引言、§3（早期学习与记忆）、§4.3（目标用时间滑动平均，式 (9)） |

## 仓库内

- `origin/worktree-s1-recipe-50:runs/R/notes.md:47-52`（S1c、S1lr3、S1r2、S3a 的数值；S1r2 为「用 S1 重新打标后续训」）。
- `origin/research/online-selfsup-collapse:docs/research/online-selfsup-collapse/README.md`（S2 设置、位置编码捷径、负样本对建议）。
- `docs/research/label-free-reward/README.md`（§1 系统偏移与随机噪声，§2.D 变换一致性对平移偏差没有信号，§2.G 合成 warp 监督，§2.H 跨模型一致）。
- `docs/research/matcher-rl-feasibility/README.md`（§3 AnyMatch 的 SGVC 用 RoMa 质检）。
- Issue #52 票面（本文按票面背景和 #23 的诊断结论写，未另查 issue 评论）。
