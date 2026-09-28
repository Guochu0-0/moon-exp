状态：探索性实验（从 S1 出发的细级共享平移 RL，偏离设计文档 5.1），不属于伪标签路线或类 RIPE 路线，保留为记录，见 #27。

## 假设

保留离线伪标签项作锚点、训练全部模块时，修正后的整对 RL 仍有增益。

## 改动

从 S1 step 6000 出发：S1 的伪标签损失（labels_b0）+ 整对 RL（σ_i 0，σ_g 0.25，K 4，梯度幅值 NCC）。全部模块，AdamW lr 1e-5，4000 步。
方法：main = 整对权重 4、梯度累积 16（R28 step 4000）；placebo = 同 main、reward 换随机数（R28p）；
noaccum = 整对权重 1、不累积（R29）；noaccum_sigg05 = 同 noaccum、σ_g 0.5（R31）；noaccum_K8（R32）；noaccum_cfog（R33）；
single_stage = 从 zero-shot 出发、伪标签与 RL 同时训 8000 步、累积 16（R43）。每个方法取 Val AUC@5 最好的 ckpt。

## 结论

main：Val AUC@5 0.283–0.287（各 ckpt），Test 0.254；与 S1 配对 Test ΔAUC@5 +0.021 [+0.015, +0.027]。placebo 0.264–0.272（= S1 平台）。
不累积梯度的变体波动更大：noaccum 0.275–0.283，noaccum_sigg05 0.269–0.280，noaccum_K8 0.267–0.286，noaccum_cfog 0.274–0.285。
single_stage 最好 0.277，低于「先 S1 再 RL」的两阶段。
与 R27（纯 RL、只训细级）效果相当。
