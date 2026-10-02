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

## 配方调参（「把 S1 调好」#50）

每次只改一个因素，lr 1e-5、seed 0、8000 步、每 1000 步在 Val 上评一次。后段 = step ≥ 2000 的 7 个 ckpt。
汇总：`scripts/finetune/curves.py`；原始结果在 gpfs `results/finetune/<name>/sweep/S`。

| run | 改动 | Val AUC@5 峰值 | 后段均值 | 后段 std |
|---|---|---|---|---|
| S1 / S1s1 / S1s2 | 对照（恒定 lr、bs 1、不裁剪） | 0.272 / 0.274 / 0.270 | 0.264 / 0.263 / 0.264 | 0.0045 / 0.0049 / 0.0040 |
| S1wc | warmup 500 + cosine | 0.266 | 0.263 | 0.0024 |
| S1clip | 梯度裁剪 0.5（上游 LoFTR） | 0.265 | 0.253 | 0.0080 |
| S1acc8 | 梯度累积 8（lr 不放大） | 0.267 | 0.261 | 0.0040 |
| S1all | 以上三者 | 0.266 | 0.262 | 0.0018 |

- 没有一个因素抬高平台期。warmup + cosine 只让 ckpt 间波动减半。
- 裁剪 0.5 有害：裁剪前梯度范数约 13–53，每一步都被裁剪，相当于逐步归一化梯度。
- 累积 8 时 lr 没放大，更新次数只有 1/8，所以不算公平比较；这个配置没有补跑。
- lr 网格按用户决定没有跑。平台期与此前 S1c、S1lr3、S1r2 一致，更像是伪标签本身的上限，不是优化配方的问题。

## 记录补记（#77，2026-10-02）

- 训练产物从 `YGC/results/finetune/{S1,S1s1,S1s2}/` 迁入本目录的 `ckpt/`、`sweep/`（方法 main、seed1、seed2）；ckpt 只留 Val 峰值与最后一个，其余在 `YGC/_archive/2026-10/ckpt/S1/`。其余 S1 变体（S1wc、S1acc8 等）没有单独的方法记录，整目录归档到 `YGC/_archive/2026-10/`。
- `code/`：`launch26.sh`（S1、S2 的启动）、`s1test.sh`、`testeval.sh`（Test 评测）、`wave1.sh`（S1 变体一波）。原在 `YGC/tmp/`，未进过 git。
