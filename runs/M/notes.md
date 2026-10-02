RoMa 首轮（#65 伪标签、#66 类 RIPE）。底座 AnyMatch-RoMa + minmax，只训 decoder（标 VGG 的另训 VGG）。分析见两张票的进展评论，本页只记方法与读数。

## 记录来源（事后重建，#77）

本实验在 #76 管理规范之前跑完，记录是 2026-10-02 从 `YGC/results/finetune/<m>/` 迁来的：

- 代码：用 `git archive` 副本 `YGC/moon-exp-roma` 跑，副本被就地覆盖过多次，当时用的确切代码已无法复原。preds meta 里的 commit 按领取时间推断：M1–M6 为 8457109，M7 及以后（含 M3s1）为 b7246eb（150 时钟偏移按 13h07m 校正；M8 用了 b7246eb 才有的 `--cert-out`）。
- 任务清单：`scripts/finetune/jobs/roma-r1.txt`（6541106 版），驱动 `scripts/finetune/roma.sh` + `queue.sh`。没有 M18。
- 一次性脚本在 `code/`：`smoke_roma.sh`（冒烟）、`roma_train.sh`（RoMa zero-shot 对 Train 匹配，产出 `labels_roma0`）、`roma_126a.sh`（M1、M3 的 Test，M3@2000 对 Train 重新打标）、`rm1_label.sh`（拼出 `labels_rm1`）、`roma_test2.sh` / `roma_test3.sh`（M4、M11n / M13、M14、M4s1 的 Test）。
- 峰值：按 Val AUC@5（不含 step 0）从 `sweep/<m>/S/metrics.json` 选出，写进 `sweep/<m>/peak.json`；Test 只在下表有值的方法上评过（只在峰值 step 上），其余只有 Val。
- ckpt 只留峰值与最后一个，其余在 `YGC/_archive/2026-10/ckpt/M/`。

## 方法

标签：`p2` = LoFTR 第三轮标签（P 系），`roma0` = RoMa zero-shot 打标，`rm1` = M3@2000 重新打标。未注明的默认：lr 1e-5、8000 步、标签前 50%。

| 方法 | 路线 | 改动 | 峰值 step | Val AUC@5 | Test AUC@5 |
|---|---|---|---|---|---|
| M1 | 伪标签 | p2，geo+photo 扰动 | 5000 | 0.306 | 0.283 |
| M2 | 伪标签 | p2，扰动，lr 3e-6 | 8000 | 0.300 | |
| M3 | 伪标签 | p2 | 2000 | 0.309 | 0.278 |
| M3s1 | 伪标签 | M3 seed 1 | 8000 | 0.306 | |
| M4 | 伪标签 | roma0 | 2000 | 0.306 | 0.276 |
| M4s1 | 伪标签 | M4 seed 1 | 4000 | 0.309 | 0.279 |
| M10 | 伪标签 | p2，标签前 100% | 4000 | 0.307 | |
| M11 | 伪标签 | rm1，扰动 | 6000 | 0.308 | |
| M11n | 伪标签 | rm1 | 6000 | 0.310 | 0.286 |
| M13 | 伪标签 | roma0，训 VGG | 8000 | 0.313 | 0.281 |
| M13s1 | 伪标签 | M13 seed 1 | 6000 | 0.314 | |
| M14 | 伪标签 | roma0，lr 3e-5 | 8000 | 0.312 | 0.284 |
| M14s1 | 伪标签 | M14 seed 1 | 7000 | 0.312 | |
| M15 | 伪标签 | roma0，训 VGG，lr 3e-5 | 7000 | 0.316 | |
| M15s1 | 伪标签 | M15 seed 1 | 8000 | 0.316 | |
| M15s2 | 伪标签 | M15 seed 2 | 4000 | 0.319 | |
| M16 | 伪标签 | roma0，训 VGG，16000 步 | 16000 | 0.317 | |
| M17 | 伪标签 | M15，16000 步 | 14000 | 0.316 | |
| M19 | 伪标签 | M15，标签前 100% | 7000 | 0.310 | |
| M20 | 伪标签 | roma0，lr 3e-5，16000 步 | 14000 | 0.315 | |
| M21 | 伪标签 | M15，lr 1e-4 | 7000 | 0.317 | |
| M22 | 伪标签 | M15 换 p2 标签 | 8000 | 0.312 | |
| M5 | 类 RIPE | Q4 配方（外点 −0.25、负样本对），4000 步 | 1000 | 0.000 | |
| M5p | 类 RIPE | M5 随机 reward 对照 | 1000 | 0.000 | |
| M6 | 类 RIPE | M5，取舍项权重 0.1 | 1000 | 0.063 | |
| M7 | 类 RIPE | 只留粗级锚点项（不训 certainty） | 1000 | 0.267 | |
| M7p | 类 RIPE | M7 随机 reward 对照 | 1000 | 0.258 | |
| M8 | 类 RIPE | 取舍项外点给 0 | 3000 | 0.256 | |
| M9 | 类 RIPE | M5，lr 1e-6 | 1000 | 0.222 | |
| M12 | 类 RIPE | 只给正分（外点 0、无负样本对） | 1000 | 0.208 | |
| M12p | 类 RIPE | M12 随机 reward 对照 | 1000 | 0.217 | |
