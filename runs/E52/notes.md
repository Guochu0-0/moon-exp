# 月球光学–SAR 上的 SuperPoint 检测可重复性

复用 [E51 月球关键点叠图](../E51/extra/moon_detections.png) 的 4 对 Val patch 及其原版 SuperPoint 检测坐标。图中 3 对有人工评测真值点，`ROI_080/patch_13` 缺少评测真值点。E51 的检测配置为长边 1024、NMS 半径 4、分数阈值 0.0005，与论文 SuperPoint+SuperGlue 基线的检测配置有差异。

## 评测口径

沿用 SuperPoint 附录 A 的点级可重复率。用每对 patch 的评测真值点最小二乘拟合光学到 SAR 的参考仿射。检测点两侧均加 0.5，与人工点统一为像素角点坐标。先按参考仿射及其逆变换筛出共同视野中的检测点，再在两侧分别按分数取前 300 点，并统一在 SAR 的 512 × 512 patch 坐标中计算欧氏距离。一个点在另一侧任意点的容差内即记为命中，两侧命中数之和除以两侧点数之和。主读数为 3 px，同时报告 1 px 和 5 px。 [SuperPoint 原文附录 A](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf) · [HPatches 复现代码](https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/detector_evaluation.py)

## 试跑结果

| Val patch 对 | 1 px | 3 px | 5 px | 参考仿射留一平均误差 |
| --- | ---: | ---: | ---: | ---: |
| `ROI_035/patch_26` | 0.0% | 4.7% | 13.5% | 7.30 px |
| `ROI_070/patch_21` | 1.0% | 4.0% | 13.5% | 2.06 px |
| `ROI_071/patch_5` | 0.7% | 5.2% | 12.0% | 3.92 px |
| 三对平均 | 0.6% | 4.6% | 13.0% | |

3 px 下每侧 300 点中，光学有 13–17 个点命中，SAR 有 11–14 个点命中。这些是点级命中数，同一个点可以命中另一侧的多个点。参考仿射留一误差是逐个留出人工点评估所得，包含标注误差和仿射近似误差。三对的留一平均误差为 2.06–7.30 px，其中两对超过 3 px 容差。因此这些分数反映检测点在当前参考几何下的位置重合程度，单凭这次试跑还无法确定检测器本身的可重复性或原因。

从 `moon-exp` 根目录复现 `python -m runs.E52.code.repeatability ../moon-dataset/data/ground-truth`。 [逐对结果与留一误差](extra/repeatability.json) · [计算脚本](code/repeatability.py)
