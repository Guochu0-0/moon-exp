# 月球光学-SAR 配准

在无标注训练集上微调多模态匹配器，估计月球光学与 SAR patch 对之间的仿射变换。

## Language

### 数据

**patch 对**：
一张光学 patch 和一张 SAR patch，在数据集构建时已经粗对齐到大致同一块区域；算法要估计的是二者之间的仿射。
_Avoid_: 图像对、样本

### 方法

**底座**：
被微调的预训练多模态匹配器（候选为 AnyMatch-LoFTR、AnyMatch-RoMa）。
_Avoid_: backbone、基线模型

**逐匹配 reward**：
给每个匹配单独打的分，例如它相对 RANSAC 仿射是否为内点、残差有多大。
_Avoid_: 单独打分、per-match reward（口头可用，文档统一用本词）

**整对 reward**：
只能对一对 patch 的整组匹配给出的一个总分，例如按估出的仿射 warp 后的跨模态结构相似度。
_Avoid_: 整组打分、全局 reward

**自洽信号**：
只用这组匹配自己估出的几何来判断好坏的信号；它看不见整体偏移。
_Avoid_: 内部一致性

**整体偏移**：
一对 patch 的全部匹配朝同一方向偏离真值，偏移之后彼此仍然一致。
_Avoid_: 系统误差、bias
