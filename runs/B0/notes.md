## 假设

现有多模态匹配器（含数据引擎论文在多模态数据上微调的权重）zero-shot 在月球光学-SAR 上效果不够好，但能提供初始起点。

## 改动

不训练。统一输入映射（光学 /255，SAR S1 dB 逐 patch p2–p98），官方默认分辨率，统一仿射 RANSAC 3 px。
runner 见 baselines/README.md；配置见 configs/baselines/。

## 附件

AnyMatch-LoFTR、AnyMatch-RoMa 在 Val 上逐 pair 的误差分解（自洽误差、真值误差、共同偏移），见[偏移诊断](https://github.com/Guochu0-0/moon-exp/issues/23)。

![](extra/offset/offset_val.png)

[pairs_val.csv](extra/offset/pairs_val.csv) · [summary_val.json](extra/offset/summary_val.json)
