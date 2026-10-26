# 来源清单（self-ref-collapse）

读取日期：2026-10-26。除非另注，论文正文都是从 arXiv 或作者/课程托管的 PDF 下载后，用 `pdftotext` 抽取文本读的。「读过的部分」列出引用处所在的章节。作者、年份、会议与 arXiv 编号用 arXiv API 的元数据（comment / journal_ref 字段）核对过。

## 论文

### 方向 1：已知扰动的几何、等变、循环一致

| 论文 | 链接 | 读过的部分 / 用于 |
|---|---|---|
| SuperPoint：DeTone, Malisiewicz, Rabinovich, CVPR 2018 Workshop（Deep Learning for Visual SLAM） | [arXiv:1712.07629](https://arxiv.org/abs/1712.07629) | §5.1 协变公式与式 10；§5.2 单应采样；§5.3 迭代；附录 C（N_h=10/100/1000，可重复性 0.67、+21%、+22%） |
| Zhou, Krähenbühl, Aubry, Huang, Efros, CVPR 2016 | [arXiv:1604.05383](https://arxiv.org/abs/1604.05383) | 摘要；§3（合成–合成边按构造为真值，以防恒等流这类平凡解；截断阈值 T=15 px） |
| Wang, Jabri, Efros, CVPR 2019 | [arXiv:1903.07593](https://arxiv.org/abs/1903.07593) | §1（「不动的跟踪天然循环一致」、捷径）；§3.1（skip cycle、特征相似度项 λ=0.1） |
| CRW：Jabri, Owens, Efros, NeurIPS 2020 | [arXiv:2006.14613](https://arxiv.org/abs/2006.14613) | §2.2 edge dropout；§2.3 patch 空间抖动；附录 C（利用边界伪影编码位置的捷径，四种对照都没消掉） |
| ICON：Greer, Kwitt, Vialard, Niethammer, ICCV 2021 | [arXiv:2105.04459](https://arxiv.org/abs/2105.04459) | 摘要；§1 贡献；§2.3 近似逆一致与随机离格评估（只用于说明「逆一致作为正则、与相似度同用」） |

### 方向 2：参照惯性、信赖域、噪声

| 论文 | 链接 | 读过的部分 / 用于 |
|---|---|---|
| Mean Teacher：Tarvainen & Valpola, NeurIPS 2017 | [arXiv:1703.01780](https://arxiv.org/abs/1703.01780) | §3（EMA teacher）；评测设置（ramp-up 期 α=0.99，之后 0.999）；附录 B.1.2（SVHN：前 40000 步 0.99，之后 0.999，使 250 标签收敛稳定） |
| BYOL：Grill et al., NeurIPS 2020 | [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) | §3.1（τ_base=0.996，余弦升到 1）；§3.2；§5「Bootstrapping」与表 5a（18.8 / 69.8 / 72.5 / 68.4 / 0.3）；表 5b（去掉预测器或目标网络则坍缩）；相关工作（Mean Teacher 去掉分类损失会坍缩）；附录 F.4 表 19（提升来自目标稳定、陈旧） |
| DINO：Caron et al., ICCV 2021 | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | §3.1（teacher 的 EMA λ 0.996→1；centering 式 4）；§3.2（τ_s=0.1，τ_t 0.04→0.07）；§5.2（学生副本、上一次迭代作 teacher 不收敛，上一 epoch 66.6；Polyak–Ruppert 解释）；§5.3 坍缩研究（m=0/0.9/0.99/0.999 → 69.1/69.7/69.4/0.1；teacher 温度需 < 0.06） |
| DQN：Mnih et al., Nature 518:529–533, 2015 | [DeepMind 托管 PDF](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)，DOI 10.1038/nature14236 | 正文与 Methods（目标网络每 C 步克隆、防振荡/发散的理由）；Extended Data Table 1（C=10000，从渲染的页面图像读出，因为表格是图片）；Extended Data Table 3 只读了表题，没读数值 |
| DDPG：Lillicrap et al., ICLR 2016 | [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) | 实验细节（软目标更新 τ=0.001） |
| PPO：Schulman et al., 2017 | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | §3（clip，ε=0.2）；§4（自适应 KL）；§5 算法（K 个 epoch）；§6.1 表 1（不 clip −0.39，clip 0.2 为 0.82） |
| TRPO：Schulman et al., ICML 2015 | [arXiv:1502.05477](https://arxiv.org/abs/1502.05477) | 实验部分（δ=0.01） |
| DeepSeekMath（GRPO）：Shao et al., 2024 | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | §4.1 GRPO 目标与 KL 无偏估计器；训练设置（β=0.04，每个探索阶段只更新一次，每题 64 个样本） |
| DAPO：Yu et al.（ByteDance Seed 与清华 AIR）, 2025 | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) | §2.3 去掉 KL；§3.1 Clip-Higher（ε_low 0.2，ε_high 0.28，上裁剪 token 概率 < 0.2）；§3.2 动态采样（零优势组）；§4 设置（小批量 512，每个 rollout 步 16 次更新） |
| Cui et al., 2025，「The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models」 | [arXiv:2505.22617](https://arxiv.org/abs/2505.22617) | §2（R = −a·exp(H) + b；前 200 步的熵与性能占比）；§3 引理 1、命题 1、定理 1–2；§4.1（熵损失系数 0.0001–0.01；对参照的 KL）；§4.2 Clip-Cov、KL-Cov 与设置（每个 rollout 步 8 次更新，r=2×10⁻⁴，界 1 到 5，k=2×10⁻³ 或 2×10⁻⁴） |
| Mei, Dai, Xiao, Szepesvári, Schuurmans, NeurIPS 2021 | [arXiv:2110.15572](https://arxiv.org/abs/2110.15572) | 摘要；§1–2（单状态老虎机，定理 3：随机 NPG；定理 4：随机 GNPG，以正概率 π(a)→1）；§3 承诺率；关于 TRPO/PPO「需多试种子」的段落。证明部分没读 |
| Spurious Rewards：Shao, Li, Xin et al., 2025 | [arXiv:2506.10947](https://arxiv.org/abs/2506.10947) | 摘要；§4（clipping bias）；附录 B.2（去掉 clip 后期望梯度为 0 但极端随机；加大小批量后稳定） |
| SRT「Can Large Reasoning Models Self-Train?」：Shafayat, Tajwar, Salakhutdinov, Schneider, Zanette, 2025 | [arXiv:2505.21444](https://arxiv.org/abs/2505.21444) | 摘要；§1；§4.1（自演化标签优于固定 teacher）；§5（奖励钻空子、模板答案、四条消融：RLOO/GRPO、KL 系数、学习率、生成数）；附录中 KL 加在奖励里的实现说明。附录 I 只读到标题列表 |
| Malladi, Lyu, Panigrahi, Arora, NeurIPS 2022 | [arXiv:2205.10287](https://arxiv.org/abs/2205.10287) | 摘要、§1（RMSprop/Adam 的 SDE、平方根缩放规则）；只读了噪声主导这一前提的表述，没读推导 |

### 方向 3：图像相似度

| 论文 | 链接 | 读过的部分 / 用于 |
|---|---|---|
| Viola & Wells, IJCV 24(2):137–154, 1997 | [课程托管 PDF](https://www.khoury.northeastern.edu/home/jaa/CSG195.08F/Topics/Papers/ViolaWe97.pdf)，DOI 10.1023/A:1007958904918 | 摘要；EMMA（Parzen 窗加随机近似）；局部极小的讨论。只略读 |
| Maes, Collignon, Vandermeulen, Marchal, Suetens, IEEE TMI 16(2):187–198, 1997 | [课程托管 PDF](https://cs.uwaterloo.ca/~jhoey/teaching/cs793/papers/MaesTMI97.pdf) | 摘要；§I（稀疏特征相关在远处不敏感）；§II（部分重叠下 MI 的解释）；§IV 设置（256×256 联合直方图，Powell）；§VI-C 部分重叠（表 IV 只读了表题）；§VII 讨论 |
| MIND：Heinrich, Jenkinson, Bhushan, Matin, Gleeson, Brady, Schnabel, Medical Image Analysis 16(7):1423–1435, 2012 | [第三方托管 PDF](https://blog.mbedded.ninja/misc/modality-independent-neighbourhood-descriptor-mind/2012-01-16-mind-modality-independent-neighbourhood-descriptor-article.pdf)（来源未核实，页眉与期刊版一致），DOI 10.1016/j.media.2012.05.008 | 结论段（不依赖全局强度关系、不是旋转不变、搜索区的讨论）。正文方法与耗时没细读 |
| HOPC：Ye, Shan, Bruzzone, Shen, IEEE TGRS 2017 | [arXiv:2103.16871](https://arxiv.org/abs/2103.16871)（作者上传的录用版），DOI 10.1109/TGRS.2017.2656380 | §I（MI 计算贵、对窗口大小敏感）；§II；§IV 运行时间比较（图 20 的文字说明）；§VI 结论与局限（对缩放、旋转不具不变性） |
| CFOG：Ye, Bruzzone, Shan, Bovolo, Zhu, IEEE TGRS 2019 | [arXiv:1808.06194](https://arxiv.org/abs/1808.06194)（录用版），DOI 10.1109/TGRS.2019.2924684 | §I（MI 的缺点）；§II–III（CFOG 构造、3D FFT）；结论（优于 NCC、MI，比 NCC 快；不能处理大旋转和大缩放） |

### 方向 4：防坍缩正则与仿射约束

| 论文 | 链接 | 读过的部分 / 用于 |
|---|---|---|
| IIC：Ji, Henriques, Vedaldi, ICCV 2019 | [arXiv:1807.06653](https://arxiv.org/abs/1807.06653) | 摘要；§3「Why degenerate solutions are avoided」「Meaning of mutual information」；过聚类辅助头 |
| VTN：Zhao, Lau, Luo, Chang, Xu, IEEE JBHI 24(5), 2020 | [arXiv:1902.05020](https://arxiv.org/abs/1902.05020) | §III 正交损失（式 9）、行列式损失（式 10）；表 I（相对权重 1 / 0.1 / 0.1） |
| Recursive Cascaded Networks：Zhao, Dong, Chang, Xu, ICCV 2019 | [arXiv:1907.12353](https://arxiv.org/abs/1907.12353) | §3（仿射级沿用 VTN 的正则损失）。只用于说明沿用关系 |
| C2FViT：Mok & Chung, CVPR 2022 | [arXiv:2203.15216](https://arxiv.org/abs/2203.15216) | §3 几何参数约束（旋转、剪切 ±π，平移 ±50%，缩放 0.5–1.5）。pdftotext 丢掉了 π 符号，按上下文补回 |
| 伪标签与 confirmation bias：Arazo, Ortego, Albert, O'Connor, McGuinness, IJCNN 2020 | [arXiv:1908.02983](https://arxiv.org/abs/1908.02983) | 摘要；§1（confirmation bias 的定义）；§3（每个小批量的最少有标签样本数） |

## 本库实验（只引用结论）

`runs/E13`、`E15`、`E16`、`E25`、`E26`、`E27`、`E28` 的 `notes.md`；实现见 `finetune/parts/rloo.py`（`in_frame`、`ncc`、`frozen_copy`、A_ref 的拟合位置，以及 `st.batch["T"]`）。

## 只引用、不重复的已有调研

- WarpC 的一致性约束退化分析、AltO 的平凡解、MU-Net 的 CFOG+NCC、SSHNet、warp-supervision、月面 Mini-RF 的难点：[`../label-free-reward/README.md`](../label-free-reward/README.md)。

## 检索过但没有采用

- SimSiam（arXiv 2011.10566）、Tian et al. 2021（arXiv 2102.06810）：下载了，但它们讨论的是表示坍缩中停梯度与预测器的作用。我们的 A_ref 本来就不求梯度，与本文问题关系较远，没有引用。
- DQN 的 arXiv 版（1312.5602）没有独立的目标网络，只用于确认「目标网络是 Nature 版加的」。
