# 来源清单（dense-rl-finetune）

读取日期：2026-10-19。除另注外，读的是 arXiv HTML 全文。

| 来源 | 链接 | 用于 |
|---|---|---|
| Pinto et al., Tuning computer vision models with task rewards, ICML 2023 | [arXiv:2302.08242](https://arxiv.org/abs/2302.08242) | 两步法（MLE 预训练 + REINFORCE）；2 个样本、一个作 baseline；不加正则；检测 recall reward；上色 reward |
| DDPO：Black et al., 2023 | [arXiv:2305.13301](https://arxiv.org/abs/2305.13301) | 去噪 MDP 与高斯策略；DDPO_SF / DDPO_IS（clip 1e-4）；按提示词归一化；不用 KL（附录 C）；过优化例子（附录 A） |
| DPOK：Fan et al., 2023 | [arXiv:2305.16381](https://arxiv.org/abs/2305.16381) | 仅摘要：KL 正则、优于监督微调 |
| Flow-GRPO：Liu et al., 2025 | [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) | ODE→SDE；G=24，G=12/6 塌缩；KL β 与「不等于早停」；训练 10 步 vs 推理 40 步 |
| SGA-Flow-GRPO：Yang et al., 2026-09 | [arXiv:2609.32340](https://arxiv.org/abs/2609.32340) | 逐 token 优势、reward 梯度 credit 图 |
| MorphSeek，2025（v3 2026-05） | [arXiv:2511.17392](https://arxiv.org/abs/2511.17392) | 潜空间高斯头；6 条轨迹 / 3 步；Dice + NJD reward；warm-up、截断 log σ、KL 到 N(0,I)；λ_KL=0 塌缩；33%→79% |
| DeepSeekMath（GRPO）：Shao et al., 2024 | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | 去 critic、组均值 baseline、KL 直接加入损失 |
| RLOO：Ahmadian et al., 2024 | [arXiv:2402.14740](https://arxiv.org/abs/2402.14740) | 留一 baseline；强初始化下 PPO 部件不必要；裁剪 <5% |
| Dr. GRPO：Liu et al., 2025 | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) | 标准差归一化带来难度偏差；去掉 |
| DAPO：Yu et al., 2025 | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) | Clip-Higher、Dynamic Sampling、去 KL |
| PPO：Schulman et al., 2017 | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | 仅作定义引用 |
| Cui et al., The Entropy Mechanism of RL for Reasoning LMs, 2025 | [arXiv:2505.22617](https://arxiv.org/abs/2505.22617) | 熵塌缩、协方差解释、Clip-Cov / KL-Cov（摘要） |
| Mohamed et al., Monte Carlo Gradient Estimation in ML, JMLR 2020 | [arXiv:1906.10652](https://arxiv.org/abs/1906.10652) | PDF 全文 §4.3（score-function 方差随维数增长，图 4）；pathwise 方差；Rao-Blackwell 化 |
| DISK：Tyszkiewicz et al., NeurIPS 2020 | [arXiv:2006.13566](https://arxiv.org/abs/2006.13566) | reward（+1 / −0.25 / λ_kp=−0.001）；网格采样；匹配概率精确计算；前 5 epoch 退火负分 |
| DSAC：Brachmann et al., CVPR 2017 | [arXiv:1611.05705](https://arxiv.org/abs/1611.05705) | 仅摘要：概率化假设选择 |
| NG-RANSAC：Brachmann & Rother, ICCV 2019 | [arXiv:1905.04132](https://arxiv.org/abs/1905.04132) | 内点数自监督；score-function + 平均损失 baseline；K、M；NG-DSAC++ 忽略天空/草地 |
| RIPE++：Künzel et al., 2026 | [arXiv:2608.19693](https://arxiv.org/abs/2608.19693) | ρ_in=1.0、ρ_out=−0.1、λ=−1e-7（表 5）；闭式期望；不可匹配退化；熵正则 1e-5 / 1e-4；内点升位姿不升；分级 reward |
| WarpC：Truong et al., ICCV 2021 | [arXiv:2104.03308](https://arxiv.org/abs/2104.03308) | 各 bipath 约束的平凡解、偏置不敏感、W→0 无信号；forward-backward 零流平凡解 |
| AltO：Song et al., NeurIPS 2024 | [arXiv:2411.13036](https://arxiv.org/abs/2411.13036) | 平凡解（常数编码 + 恒等）；交替优化；GAP |
| RoMa v2：Edstedt et al., 2025 | [arXiv:2511.15706](https://arxiv.org/abs/2511.15706) | refiner 输出逐像素精度矩阵，高斯 NLL |
| SRT：Shafayat et al., Can Large Reasoning Models Self-Train?, 2025 | [arXiv:2505.21444](https://arxiv.org/abs/2505.21444) | 摘要：多数投票自奖励长期训练后突然崩溃 |
| Spurious Rewards：2025 | [arXiv:2506.10947](https://arxiv.org/abs/2506.10947) | 摘要：随机 reward 在 Qwen2.5-Math 上 +21.4；裁剪偏差解释；跨模型族不成立 |

## 本库线索（非一手文献）

- [#66](https://github.com/Guochu0-0/moon-exp/issues/66) 评论：RoMa 上类 RIPE 闭式期望的各变体，certainty 塌到 0、M7 与 zero-shot 持平、随机对照。
- `docs/research/matcher-rl-feasibility/README.md`：RoMa 代码中的随机性位置（GM 4096 锚点、certainty、multinomial + KDE 采样）、refiner 为确定性回归。
- `docs/research/label-free-reward/README.md`、`docs/research/rl-registration-survey.md`：reward 候选的四维评估、SSHNet / MU-Net 事实。

## 检索过但没有采用

- ReinFlow、DPPO、FPO、π_RL：机器人 / VLA 的流匹配策略 RL，形式与 Flow-GRPO 同类，没有带来新的信息。
- 「Dense prediction rewards collapse GRPO-trained LLM agents」（ResearchGate）：来源可信度低，未采用。
- 没找到对光流、立体、深度网络做策略梯度微调的工作。
