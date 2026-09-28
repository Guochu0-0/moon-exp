# 来源清单（label-free-reward）

读取日期：2026-09-27。论文正文是从 arXiv PDF 下载后用 `pdftotext` 抽取文本读的（除非另注）；「读过的部分」列出引用处所在的章节。

## 论文

| 论文 | 链接 | 读过的部分 / 用于 |
|---|---|---|
| NG-RANSAC：Brachmann & Rother, ICCV 2019 | [arXiv:1905.04132](https://arxiv.org/abs/1905.04132) | 全文。§3 期望任务损失与 baseline；§4.1「Self-supervised Learning」（内点数当训练信号、稳定、自监督略差于有监督）；K=4 池 × M=16 假设；NG-DSAC++ 忽略天空/草地（§4.3） |
| RIPE++：Künzel, Eisert, Hilsmann, 2026 | [arXiv:2608.19693](https://arxiv.org/abs/2608.19693) | 全文。§3 仅正样本 reward 与不可匹配惩罚；附录分级 reward（截断二次核，Sampson 距离）；消融表（内点数升、位姿不升；正则权重 1e-5/1e-4）；§8 去掉负样本的影响 |
| SCENES：Kloepfer et al., 3DV 2024 | [arXiv:2401.10886](https://arxiv.org/abs/2401.10886) | 全文。§3.4 bootstrap；§4.2 实现（OpenCV RANSAC、≥100 匹配 / ≥20 内点过滤、MatchFormer-lite / ASpanFormer）；消融（位姿扰动 1–2°）；结论的未来工作；补充材料定性分析（提升来自一致性、基础模型失败的对提升最弱） |
| WarpC：Truong, Danelljan, Yu, Van Gool, ICCV 2021 | [arXiv:2104.03308](https://arxiv.org/abs/2104.03308) | §1 引言（光度假设失效、warp-supervision 泛化差）；§3.3–3.4 各 bipath 约束的退化解 / 偏置不敏感 / W→0 无信号；§4 消融；补充 A（forward-backward 权重把网络推向零） |
| AltO：Song et al., NeurIPS 2024 | [arXiv:2411.13036](https://arxiv.org/abs/2411.13036)，代码 [songsang7/AltO](https://github.com/songsang7/AltO)（HEAD `8d7211e`，未读代码） | §3.2 Trivial Solution Problem；§5 表 1 说明（MACE > 23 px 即失败）；§8 局限 |
| SSHNet：Yu et al., 2024 | [arXiv:2409.17993](https://arxiv.org/abs/2409.17993) | §3（同时训练两个子网络不收敛）；§5 数据集（含 OPT-SAR）；消融（无重构时灰度监督不收敛）；表 5（PDF 抽取错位，只确认了量级） |
| DCFlow：2025 | [arXiv:2509.24423](https://arxiv.org/abs/2509.24423) | §3.4 跨模态一致性约束（两侧施加随机仿射）；§4 消融（只用光度损失效果差） |
| MU-Net：Ye et al., ISPRS Archives XLIII-B3-2022；期刊版 TGRS 2022 | [ISPRS PDF](https://isprs-archives.copernicus.org/articles/XLIII-B3-2022/537/2022/isprs-archives-XLIII-B3-2022-537-2022.pdf)，[IEEE 9758703](https://ieeexplore.ieee.org/document/9758703/)（期刊版未读），代码 [yeyuanxin110/MU-Net](https://github.com/yeyuanxin110/MU-Net)（HEAD `98be918`，未读代码） | 会议版全文：损失公式（双向、exp）、CFOG + NCC、Sentinel-1/2 光学–SAR 数据、对比方法讨论（CFOG 对旋转敏感、端到端回归的通病） |
| 事件–图像无标签目标域自蒸馏：2026 | [arXiv:2607.10082](https://arxiv.org/abs/2607.10082) | §3.4 Epipolar-Guided Self-Distillation（EMA teacher、一致性验证、RANSAC 极线置信度、confirmation bias）；附录 8.1（SuperPoint 底座） |
| Self-Improving Visual Odometry：DeTone, Malisiewicz, Rabinovich, 2018 | [arXiv:1812.03245](https://arxiv.org/abs/1812.03245) | 摘要、§1–2、§5.1（用重投影误差标注稳定性） |
| ∇-RANSAC：Wei, Patel, Shekhovtsov, Matas, Barath, ICCV 2023 | [arXiv:2212.13185](https://arxiv.org/abs/2212.13185)，代码 [weitong8591/differentiable_ransac](https://github.com/weitong8591/differentiable_ransac)（HEAD `d128128`，未读代码） | 仅摘要（经 arXiv API） |
| Mini-RF/Mini-SAR DEM 辅助配准：Remote Sensing 17(4):613, 2025 | [DOI 10.3390/rs17040613](https://doi.org/10.3390/rs17040613)，[PolyU 仓储 PDF](https://ira.lib.polyu.edu.hk/bitstream/10397/114990/1/remotesensing-17-00613.pdf) | §1 引言（灰度方法适用条件、月面 SAR 缺角点 / 边缘模糊 / 低 SNR、NAC 因 PSR 与轨道误差不能当参考）；§2 方法概述（DEM 模拟 SAR + NCC） |

NG-RANSAC 代码 [vislearn/ngransac](https://github.com/vislearn/ngransac)（HEAD `866530a`）、WarpC 代码 [PruneTruong/DenseMatching](https://github.com/PruneTruong/DenseMatching)（HEAD `b054fe9`）只记录了版本，没有读代码。

## 只引用、不重复的已有调研

- RIPE、DISK、RFP、FeMIP、CoLReg、arXiv:2508.07812、ACAMatch：[`docs/research/rl-registration-survey.md`](../rl-registration-survey.md)（main）。
- LoFTR/RoMa 的不可微点、RIPE++ 闭式期望、AnyMatch SGVC、MatchAnything 伪标签：`research/matcher-rl-feasibility` 分支的 `docs/research/matcher-rl-feasibility/README.md`。

## 检索过但没有采用

- Suri & Reinartz, TGRS 2010（MI 配准 TerraSAR-X 与 Ikonos，[IEEE 5340570](https://ieeexplore.ieee.org/document/5340570/)）：只看到检索摘要，没读到正文，没有引用具体结论。
