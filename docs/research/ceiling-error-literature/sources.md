# 来源清单（#67）

核对日期 2026-09-30。「元数据已核」指题名、作者、刊物、卷期页、年份、DOI 都经 Crossref API 对过。「内容已核」指读过原文或官方全文的相关段落；没读到原文的只标元数据已核，引用时写到章节级，不写式号。

## 配准误差理论（FLE / FRE / TRE）

| 编号 | 文献 | 元数据 | 内容 | 用在哪 |
|---|---|---|---|---|
| F98 | J. M. Fitzpatrick, J. B. West, C. R. Maurer Jr., "Predicting error in rigid-body point-based registration", *IEEE TMI* 17(5):694–702, 1998. doi:10.1109/42.736021 | 已核 | 〈FRE²〉≈(1−2/N)〈FLE²〉 经二手来源（UCL MPHY0026 教程、Vanderbilt 讲义）印证；式号未核 | 残差反推定位误差的先例；TRE 预测 |
| FW01 | J. M. Fitzpatrick, J. B. West, "The distribution of target registration error in rigid-body point-based registration", *IEEE TMI* 20(9):917–927, 2001. doi:10.1109/42.952729 | 已核 | 未读全文 | TRE 分布 |
| F09 | J. M. Fitzpatrick, "Fiducial registration error and target registration error are uncorrelated", *Proc. SPIE* 7261, Medical Imaging 2009, 726102. doi:10.1117/12.813601 | 已核 | 摘要已核：一阶近似下，FLE 为正态时各种拟合优度量都和 TRE 统计独立 | r 与 Hε 独立的配准版本 |
| W08 | A. D. Wiles, A. Likholyot, D. D. Frantz, T. M. Peters, "A statistical model for point-based target registration error with anisotropic fiducial localizer error", *IEEE TMI* 27(3):378–390, 2008. doi:10.1109/TMI.2007.908124 | 已核 | 摘要级 | 各向异性标注误差 |
| DF11 | A. Danilchenko, J. M. Fitzpatrick, "General approach to first-order error prediction in rigid point registration", *IEEE TMI* 30(3):679–693, 2011. doi:10.1109/TMI.2010.2091513（勘误 30(11):2012, doi:10.1109/TMI.2011.2173637） | 已核 | 摘要级：异方差 FLE、加权最小二乘 | 异方差标注误差 |
| MA09 | M. H. Moghari, P. Abolmaesumi, "Distribution of fiducial registration error in rigid-body point-based registration", *IEEE TMI* 28(11):1791–1801, 2009. doi:10.1109/TMI.2009.2024208 | 已核 | 未读全文 | FRE 分布（备查） |

## 带标注误差的配准评测数据集

| 编号 | 文献 | 元数据 | 内容 | 用在哪 |
|---|---|---|---|---|
| C09 | R. Castillo et al., "A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets", *Phys. Med. Biol.* 54(7):1849–1870, 2009. doi:10.1088/0031-9155/54/7/001 | 已核 | 摘要已核：用重复标注估计观察者间 / 内变异 | 用重复标注直接测 σ |
| M11 | K. Murphy et al., "Evaluation of registration methods on thoracic CT: the EMPIRE10 challenge", *IEEE TMI* 30(11):1901–1920, 2011. doi:10.1109/TMI.2011.2158349 | 已核 | 二手来源：每个地标至少 3 名观察者独立标注，差异 ≥3 mm 的再复核 | 多人标注 |
| H23 | A. Hering et al., "Learn2Reg: comprehensive multi-task medical image registration challenge, dataset and evaluation in the era of deep learning", *IEEE TMI* 42(3):697–712, 2023. doi:10.1109/TMI.2022.3213983 | 已核 | 未读全文 | 仅作背景，README 没有引它的具体做法 |
| FIRE | C. Hernandez-Matas et al., "FIRE: Fundus Image Registration dataset", *Modeling and Artificial Intelligence in Ophthalmology* 1(4):16–28, 2017. doi:10.35119/maio.v1i4.42 | 已核 | 未核它对真值误差的处理 | AUC 式指标的来源背景 |
| HP | V. Balntas, K. Lenc, A. Vedaldi, K. Mikolajczyk, "HPatches: a benchmark and evaluation of handcrafted and learned local descriptors", CVPR 2017, pp. 3852–3861. doi:10.1109/CVPR.2017.410 | 已核 | 未核 | 背景；没找到它对真值噪声上限的讨论 |

## 测绘精度标准

| 编号 | 文献 | 元数据 | 内容 | 用在哪 |
|---|---|---|---|---|
| ASPRS23 | ASPRS, *ASPRS Positional Accuracy Standards for Digital Geospatial Data*, Edition 2, Version 1.0.0, Feb 2023（正文 PDF：aagsmo.org 镜像 `ASPRS_PosAcc_Edition2_MainBody.pdf`；公告见 *PE&RS* 89(10):589–592, doi:10.14358/PERS.89.10.589） | 已核 | **已读原文**：§7.11 两个误差分量（拟合检查点的误差 + 检查点测量误差）按误差传播平方和合成，表 7.4 示例 1.0 与 2.0 cm 合成为 2.24 cm；§7.12 要求检查点至少比被测产品精度高 2 倍（第 1 版是 3 倍，见 Foreword「Summary of Changes」）；§7.12 之后要求至少 30 个检查点 | 检查点误差如何进入报告精度 |
| ASPRS24 | ASPRS Positional Accuracy Standards, Edition 2 Version 2, 2024. doi:10.14358/ASPRS.PAS.2024 | 已核 | 未读，不知道相对 v1.0 改了什么 | 引用时注明版本 |
| A20 | Q. Abdullah, "Rethinking error estimations in geospatial data: the correct way to determine product accuracy", *PE&RS* 86(7):397–403, 2020. doi:10.14358/PERS.86.7.397 | 已核 | 未读；ASPRS23 §7.11.2 脚注引用了它 | ASPRS 合成公式的出处 |

## 线性回归诊断与预测误差

| 编号 | 文献 | 元数据 | 内容 | 用在哪 |
|---|---|---|---|---|
| SL03 | G. A. F. Seber, A. J. Lee, *Linear Regression Analysis*, 2nd ed., Wiley, 2003. doi:10.1002/9780471722199 | 已核 | 教科书内容（帽子矩阵性质、正态下 β̂ 与残差独立、欠拟合偏差）；具体节号未核 | 第 1、2 项的一般理论 |
| HW78 | D. C. Hoaglin, R. E. Welsch, "The hat matrix in regression and ANOVA", *The American Statistician* 32(1):17–22, 1978. doi:10.1080/00031305.1978.10479237 | 已核 | 标准内容（杠杆值、留一残差 r_i/(1−h_ii)） | 杠杆值 |
| A74 | D. M. Allen, "The relationship between variable selection and data augmentation and a method for prediction", *Technometrics* 16(1):125–127, 1974. doi:10.1080/00401706.1974.10489157 | 已核 | PRESS 的原始出处 | 留一法 |
| C77 | R. D. Cook, "Detection of influential observation in linear regression", *Technometrics* 19(1):15–18, 1977. doi:10.1080/00401706.1977.10489493 | 已核 | 标准内容 | 学生化残差、影响点 |
| M73 | C. L. Mallows, "Some comments on Cp", *Technometrics* 15(4):661–675, 1973. doi:10.2307/1267380 | 已核 | 标准内容：E[RSS_p]=(n−p)σ²+偏差平方和 | 第 2 项的矩估计 |
| E86 | B. Efron, "How biased is the apparent error rate of a prediction rule?", *JASA* 81(394):461–470, 1986. doi:10.1080/01621459.1986.10478291 | 已核 | 标准内容：训练残差对预测误差的乐观偏差 | 第 1 项的 r 偏乐观 |
| E04 | B. Efron, "The estimation of prediction error: covariance penalties and cross-validation", *JASA* 99(467):619–632, 2004. doi:10.1198/016214504000000692 | 已核 | 标准内容 | 同上，另有交叉验证的偏差 |
| ET93 | B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993（第 17 章 "Cross-validation and other estimates of prediction error"，doi:10.1007/978-1-4899-4541-9_17） | 已核 | 标准内容 | 按 pair 自助法求区间 |
| W80 | H. White, "Using least squares to approximate unknown regression functions", *International Economic Review* 21(1):149–170, 1980. doi:10.2307/2526245 | 已核 | 标准内容：模型设定错误时，最小二乘估的是总体最优线性近似 | 「整片最优仿射」的定义 |
| B19 | A. Buja et al., "Models as approximations I: consequences illustrated with linear regression", *Statistical Science* 34(4):523–544, 2019. doi:10.1214/18-STS693 | 已核（页码据记忆，Crossref 没返回） | 标准内容 | 同上，另讲设计点随机时的额外方差 |

## 空间相关误差场

| 编号 | 文献 | 元数据 | 内容 | 用在哪 |
|---|---|---|---|---|
| RW06 | C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006. doi:10.7551/mitpress/3206.001.0001 | 已核 | 标准内容：§4.2 平方指数核，式 (4.9)；§5.4.1 边际似然，§5.4.2 留一交叉验证选超参 | 第 3 项 |
| S99 | M. L. Stein, *Interpolation of Spatial Data: Some Theory for Kriging*, Springer, 1999. doi:10.1007/978-1-4612-1494-6 | 已核 | 标准内容：批评平方指数（「Gaussian」）协方差过于光滑，推荐 Matérn | 第 3 项的核选择 |
| C93 | N. Cressie, *Statistics for Spatial Data*, rev. ed., Wiley, 1993. doi:10.1002/9781119115151 | 已核 | 标准内容：变差函数估计与拟合 | 第 3 项 |
| C85 | N. Cressie, "Fitting variogram models by weighted least squares", *Mathematical Geology* 17(5):563–586, 1985. doi:10.1007/BF01032109 | 已核 | 标准内容 | 第 3 项 |
| M63 | G. Matheron, "Principles of geostatistics", *Economic Geology* 58(8):1246–1266, 1963. doi:10.2113/gsecongeo.58.8.1246 | 已核 | 标准内容 | 克里金、变差函数的源头 |

## 「噪声上限」概念

| 编号 | 文献 | 元数据 | 内容 | 用在哪 |
|---|---|---|---|---|
| N14 | H. Nili et al., "A toolbox for representational similarity analysis", *PLoS Comput. Biol.* 10(4):e1003553, 2014. doi:10.1371/journal.pcbi.1003553 | 已核 | 标准内容：「noise ceiling」，用受试者间一致性给出上下界 | 第 4 项 |
| LC19 | A. Lage-Castellanos, G. Valente, E. Formisano, F. De Martino, "Methods for computing the maximum performance of computational models of fMRI responses", *PLoS Comput. Biol.* 15(3):e1006397, 2019. doi:10.1371/journal.pcbi.1006397 | 已核 | 按题名和摘要：比较了几种算模型最大可达性能的方法 | 第 4 项 |

## 未采用 / 未核实
- Förstner & Wrobel, *Photogrammetric Computer Vision*, Springer 2016（doi:10.1007/978-3-319-11550-4）：元数据已核。印象里有用精度有限的参考值做评估的讨论，但**章节没有核实**，README 没有引用。
- FIRE、HPatches、MegaDepth 有没有显式讨论「真值误差带来的上限」：**没找到**。
