## 假设

用 zero-shot 匹配估出的仿射当几何伪真值，按 LoFTR 原始监督形式微调，能提高精度（SCENES，arXiv:2401.10886）。
作为对比方法之一（「复现 SCENES 式伪标签微调 baseline」#26）。

## 改动

底座 AnyMatch-LoFTR。离线伪标签：zero-shot 在 Train 上推理一次，仿射 RANSAC 3 px；匹配 < 100 或内点 < 20 的对不用（7907 对中保留 7880 对）。
损失：粗级 = 光学格经伪仿射落到的 SAR 格为正样本，sparse focal；细级 = 窗口内归一化偏移，l2_with_std。
AdamW lr 1e-5，batch 1，8000 步，模块保持 eval。每 1000 步存 ckpt，按 Val AUC@5 选 step 6000。
代码：finetune/{pseudo,label,train}.py，scripts/finetune/scenes.sh。

## 结论

Val AUC@5 0.220 → 0.272（CI 0.255–0.290），Test 0.188 → 0.233；SR@10 Val 0.85 → 0.94，Test 0.73 → 0.89。
增益在 1000 步内出现，之后平台期（extra/val_by_step.json）。从 step 6000 继续训练（S1c）或 lr 3e-5（S1lr3）均停在 0.26–0.27。
换种子复现（同配置 8000 步，按 Val 选模）：seed1 Val 0.273 / Test 0.233（step 4000），seed2 Val 0.270 / Test 0.229（step 5000）。

## 下一步

配方未调（lr 1e-5 恒定、batch 1，无 warmup / 衰减 / 梯度裁剪），各 ckpt 的 Val 在 0.259–0.272 间波动，main 取的是 Val 峰值；调参另开票。
