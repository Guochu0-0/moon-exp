状态：探索性实验（从 S1 出发的细级共享平移 RL，偏离设计文档 5.1），不属于伪标签路线或类 RIPE 路线，保留为记录，见 #27。

## 假设

R26 的做法在全量 Train 上同样有效，且增益来自 reward（安慰剂对照）、对种子与超参稳定。

## 改动

从 S1 step 6000 出发，纯 RL（无伪标签项）：细级高斯策略只探索全体匹配共享的平移 z ~ N(0, σ_g²)（σ_g 0.25 = 1 输入 px，σ_i 0），
log π = −‖mean_m(a − μ)‖² / 2σ_g²；K 4，组内标准化 advantage；整对 reward = 采样匹配经仿射 RANSAC 后 warp 的梯度幅值 NCC。
只训 fine_preprocess + loftr_fine；AdamW lr 5e-5，梯度累积 16，4000 步。推理与底座相同（用 μ），零额外开销。
方法（除注明外其余同 main）：main = R27 step 2000；seed1 / seed2 = 换种子（R27s1 / R27s2）；placebo = reward 换随机数（R27p）；
lr1e-4（R37）；steps8000（R38）；K8（R39）；sigg05 = σ_g 0.5（R40）；cfog = reward 用 CFOG（R41）；
match = 再加逐匹配残差 reward（R44）；from_b0 = 从 zero-shot 出发、不经 S1（R42）；
on_S1s1 / on_S1s2 = 从 S1 换种子的 ckpt 出发（S1/seed1 step 4000、S1/seed2 step 5000；R27b1 / R27b2）。每个方法取 Val AUC@5 最好的 ckpt。

## 结论

Test AUC@5（相对 S1 0.233 的配对 ΔAUC@5，95% CI）：main 0.253（+0.020 [+0.015, +0.025]）、seed1 0.251（+0.018 [+0.013, +0.023]）、
seed2 0.251（+0.018 [+0.013, +0.023]）、cfog 0.256（+0.023 [+0.017, +0.028]）。main 与 seed1 之差 −0.001 [−0.005, +0.002]。
Val AUC@5 各 ckpt：main 0.284–0.287、seed1 0.285–0.291、seed2 0.287–0.289；placebo 0.269–0.271（= S1 平台）。
从 S1 其他种子出发同样有效（Test，相对各自起点的配对 ΔAUC@5）：on_S1s1 0.233 → 0.248（+0.014 [+0.010, +0.019]），
on_S1s2 0.229 → 0.250（+0.021 [+0.016, +0.027]）。汇总见 extra/paired_test.md。
lr1e-4、steps8000、K8、sigg05、cfog、match 均在 0.283–0.290；增益在 1000–2000 步内出现，8000 步不再上升也不退化。
from_b0 只到 0.226（zero-shot 0.219）：只训细级、只探索共享平移时，从 zero-shot 出发几乎没有增益。
Test SR@10 0.871–0.877，略低于 S1 的 0.885：增益来自 5–10 px 的对进入 5 px 以内，大错没有减少。
策略均值处的整对 reward（固定 Val pair）：S1 0.1002 → main 0.1041；placebo 0.0999。

## 下一步

探索性实验，偏离了设计文档 5.1（只训细级、粗级冻结，从 S1 出发两阶段），结论不直接进入主方案；按 5.1 的首轮实验另开票。解读时注意：
- 与 S1 的比较跨了优化配方（R27 lr 5e-5、梯度累积 16；S1 lr 1e-5、不累积）；reward 的作用由同配方的 placebo 控制。
- 测试时把 S1 的预测整体平移到梯度 NCC 峰，Val AUC@5 即达 0.284（baselines/diagnose_reward.py）；R26 只用 64 对也拿到几乎全部增益。R27 的增益很可能主要是学到了整体偏移修正。
- 夜间还跑过若干未登记的 run，见 runs/R/notes.md。
