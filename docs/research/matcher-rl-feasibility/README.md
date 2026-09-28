# LoFTR 系 vs RoMa 系：监督方式与 RL（policy gradient）微调可行性

> 调研日期：2026-09-27。回答 [#14](https://github.com/Guochu0-0/moon-exp/issues/14)，服务于 #5（网络结构选型）。
> 前提：只在现成多模态权重（AnyMatch-LoFTR / AnyMatch-RoMa / MatchAnything-RoMa / MINIMA-RoMa）上微调，不重新预训练；训练集无标注；RL 卖点立在「不可微」上。
> 标注约定：**【读到】** = 直接读到论文原文或代码（出处与 commit 见 [sources.md](sources.md)）；**【推断】** = 我的判断。
> 不重复 [`rl-registration-survey.md`](../rl-registration-survey.md) 已有内容（RIPE/DISK/FeMIP 等），只补本题需要的新事实。

## 对照表

| 维度 | LoFTR 系（AnyMatch-LoFTR） | RoMa 系（AnyMatch-/MINIMA-/MatchAnything-RoMa） |
|---|---|---|
| 匹配表示 | 半稠密：1/8 粗网格上的**匹配概率矩阵** + 1/2 分辨率 5×5 窗口内细化 | 稠密：每像素一个 **warp（回归坐标）+ certainty** |
| 网络原生的概率分布 | 粗级 dual-softmax 置信矩阵 `P(i,j)`（行×列两个 softmax 相乘）；细级 5×5 窗口 softmax 热图 | 粗级 GM 分类器：每个 1/14 格子对 64×64=4096 个锚点的 softmax；certainty（sigmoid）；**refiner 输出的位移是确定性回归，没有分布** |
| 推理时的离散选择 | 置信阈值 `thr`（默认 0.2，本库 baseline 用 0.1）→ 去边 → **MNN** → 细级 soft-argmax（可微）→ 下游 RANSAC | 取 GM 分类的 argmax 锚点再局部加权 → certainty 阈值（>0.05 置 1）→ **multinomial 采样 4×num** → KDE 平衡 → **再 multinomial 采 num 个** → 下游 RANSAC |
| 离散选择的维度（本库输入：512² patch → 长边 640 / RoMa 560+864） | 粗网格 80×80=6400 格/图，置信矩阵 6400×6400≈4.1×10⁷；细级每个匹配 25 选 1 | 粗级 40×40 格 × 4096 类；最终稠密 warp 864×864≈7.5×10⁵ 像素/方向（symmetric 共约 1.5×10⁶ 候选），采样 40 000 → 10 000（MatchAnything 为 5 000） |
| 原始损失 | 粗级：置信矩阵 focal loss（官方代码默认监督整张矩阵）；细级：带 std 权重的 L2 | 每尺度：GM 锚点交叉熵 + certainty BCE；refiner 广义 Charbonnier（α=0.5）+ certainty BCE；尺度间 `detach` |
| GT 来源 | MegaDepth 深度 + 位姿投影（`warp_kpts`） | 同：深度 + 位姿投影（`get_gt_warp`） |
| 多模态版训练代码 | AnyMatch 仓库带 `third_party/LoFTR_AnyMatch/train.py`，但数据加载器是 MINIMA 的 MegaDepth 多模态版，**没有 Any-syn 加载器**；README 的 “Training Code” 仍未勾选 | MINIMA：训练脚本已公开（2025-04）；AnyMatch：同 MINIMA 代码（仅一行不同），Any-syn 训练未公开；MatchAnything：**训练代码未公开**（README “will be available later”） |
| 在权重上继续微调的难度 | 模型 + 损失 + Lightning 训练循环都在，ckpt 为 Lightning 格式，可直接接 | 模型 + 损失 + 训练循环都在（romatch）；MatchAnything-RoMa 是另一份分叉，有 Lightning 包装但缺 train 入口与数据集 |
| 已公开的多模态微调成本 | MINIMA-LoFTR：4×RTX 3090，总 bs 8，30 epoch，长边 640；AnyMatch-LoFTR：4×RTX 4090，bs 4，10 epoch，512² | MINIMA-RoMa：4×RTX 3090，总 bs 12，4 epoch，560²；AnyMatch-RoMa：4×RTX 4090，bs 2，4 epoch，512² |

## 1. 流水线与离散/不可微选择

**LoFTR（AnyMatch-LoFTR，推理代码与上游 LoFTR `df7ca80` 相同）【读到】**
- ResNet-FPN 出 1/8 与 1/2 特征 → LoFTR transformer → 粗级 `sim/temperature`，`conf = softmax(sim,1)*softmax(sim,2)`（`coarse_matching.py:113-119`）。
- `get_coarse_match` 整个在 `@torch.no_grad()` 下：`conf > thr` → `mask_border` → 行/列都取最大（MNN）→ 得到匹配索引（`:150-196`）。**这是第一个不可微点，维度就是 6400×6400 的置信矩阵。**
- 细级：对每个粗匹配，在 5×5 窗口做 softmax 热图，取期望（`dsnt.spatial_expectation2d`）作为亚像素位置，同时算 std（`fine_matching.py:43-57`），这一步**可微**。
- 训练时粗匹配不够就随机补 GT 匹配（`coarse_matching.py:198-236`），细级损失依赖 GT。

**RoMa（MINIMA / AnyMatch 用 `RoMa_minima`，MatchAnything 用自己的分叉）【读到】**
- 冻结的 DINOv2-L（粗）+ VGG19（细）→ 1/14 格子上 GP 回归 + transformer 解码器，输出 64×64+1 通道：4096 个锚点的分类 logit + certainty（`train_roma_outdoor.py:26-37`）。
- `cls_to_flow_refine`：softmax 后**取 argmax 锚点**，再用它和 4 个邻居的概率加权平均出坐标（`utils.py:302-323`），argmax 不可微，加权部分可微。
- ConvRefiner 在 16/8/4/2/1 各尺度回归位移 + certainty 增量；`Decoder(detach=True)`，尺度之间的 flow/certainty 被 `detach`（`matcher.py:418-420`）。推理时先在 560 跑，再在 864 上 upsample 细化一遍。
- `match()` 整个在 `torch.no_grad()` 里，输出稠密 warp + sigmoid certainty（`matcher.py:642-748`）。
- `sample()`：certainty>0.05 的一律置 1，`torch.multinomial` 无放回抽 4×num，KDE 估密度后按 1/(density+1) **再抽一次** num 个（`matcher.py:468-495`）。**这是不可微且本身就是随机的选择**；之后 RANSAC 同样不可微。

**【推断】对 RL 的含义**
- LoFTR 的粗级置信矩阵本身就是一个「在候选对应上的概率分布」，与 RIPE++ 给 LightGlue 用的 dual-softmax `P(i,j)` 形式相同（见第 5 点）。policy 可以直接定义在它上面，不用加头。细级 25 类 softmax 也能当 categorical policy，直接覆盖亚像素定位。
- RoMa 真正决定精度（3 px 以内）的是 refiner，而 refiner 是**确定性回归**，没有可采样的分布。要让「匹配位置」成为 RL 动作，得自己加随机性（比如给 warp 加高斯策略头）；现成的随机性只在「选哪些像素」（certainty 采样）和粗级 4096 类 GM 分类上。粗级锚点间距是 560/64≈8.75 px（560 网格），比 3 px 的瓶颈粗。
- RoMa 的采样只用一次前向的输出，能以很低成本在同一张图上多次采样（便于 GRPO 式组内比较）；LoFTR 从置信矩阵采样也一样。两者的前向都不需要为每次采样重跑。

## 2. 原始监督信号与损失【读到】

- **LoFTR**：`LoFTRLoss` = 粗级 focal loss（官方代码对整张置信矩阵做稠密监督，`_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS=False`；LoFTR `docs/TRAINING.md` 说明这与论文不同，但效果更好）+ 细级 `l2_with_std`（按 1/std 加权的 L2，只算落在窗口内的 GT，`FINE_CORRECT_THR=1.0`）。GT：`spvs_coarse` 用 `warp_kpts`（深度 + K + T）投影网格点得到 `conf_matrix_gt` 和细级偏移。
- **RoMa**：`RobustLosses`：粗级 GM 锚点交叉熵（离 GT 最近的锚点为目标）+ `ce_weight=0.01` 的 certainty BCE；各 refiner 尺度广义 Charbonnier（`alpha=0.5, c=1e-4`，按 scale 缩放）+ certainty BCE；较细尺度只监督上一尺度误差小于 `local_dist` 的像素（`robust_loss.py:138-141`）。GT：`get_gt_warp` 用深度 + 位姿投影，`prob` 为共视/深度一致掩码。RoMa 论文：DINOv2 全程冻结；K=64×64 锚点；Charbonnier α=0.5。
- 两者的 **GT 都来自深度 + 位姿**。我们的训练集两者都没有，所以原始损失不能直接用。

## 3. 多模态版本怎么训的【读到】

| 权重 | 数据引擎 | 初始化 / 损失 | 训练代码 |
|---|---|---|---|
| MINIMA-LoFTR / -RoMa（CVPR 2025） | MD-syn：MegaDepth RGB 经生成模型转成红外（StyleBooth+LoRA）、深度（DepthAnything V2）、事件（物理仿真）、法线（DSINE）、素描/油画；GT 直接继承 MegaDepth 的深度 + 位姿；约 4.8 亿对 | 论文：用官方 RGB 权重初始化后微调，损失沿用原版；配置 `pre_trained_path: ./weights/roma_outdoor.pth` / `outdoor_ds.ckpt`，`lr_scale 0.1`，训练时每对随机抽一种模态 | **公开**：`train_orders/minima_{loftr,roma}.sh` + `LoFTR_minima` / `RoMa_minima` 子模块 |
| AnyMatch-LoFTR / -RoMa（ECCV 2026） | Any-syn：GLDv2 单视图图像 → 单目深度 → 重投影出新视角 + 扩散补洞 → 转成 IR/深度/法线/事件；SGVC 用 RoMa 算 PCK 做质检（τ=5，η=0.6）；训练集 50 万对，512² | 论文：用官方 LoFTR / RoMa 预训练权重初始化；RoMa 编码器/解码器学习率 6.25e-8 / 1.25e-6，4 epoch 收敛；LoFTR 学习率 8e-4，10 epoch | **只公开了一部分**：数据引擎已发布；`third_party/*_AnyMatch` 带训练代码，但加载器是 MINIMA 的 MegaDepth 多模态版（与 RoMa_minima 相比 RoMa 只差 1 行）；README TODO 里 “Training Code” 未勾选 |
| MatchAnything-RoMa（TPAMI 2026） | 多源：MegaDepth/ScanNet++/BlendedMVS 深度投影；DL3DV 视频用 RoMa 跟踪做伪标签；GoogleLandmark/SA-1B 单图 + 随机单应；CycleGAN 可见光→红外、昼→夜，DepthAnything 出深度；约 8 亿对；**SAR 不在训练集中** | 论文：「用官方实现，保持相同超参与损失」，16×A100-80G、bs 64、RoMa 约 6 天；**论文没写是从头训还是从官方权重初始化** | **未公开**：GitHub README（`8cd8c11`，2026-09-15）仍写 “training code will be available later”；HF Space 的代码里有 `forward_train_framework` 和 Lightning `training_step`，但没有 train 入口和训练数据集 |

**【推断】** 按「直接接着微调」的门槛排：MINIMA-RoMa ≈ AnyMatch-RoMa（同一套 romatch 训练代码，加载 ckpt 即可）≈ AnyMatch-LoFTR（LoFTR Lightning 训练代码完整）＞ MatchAnything-RoMa（模型可加载，训练循环要自己补）。无论哪一个，RL 微调都要自己写：数据加载器（无 GT 的月球 patch 对）、reward、采样与 policy 对数概率。

## 4. 微调成本

**【读到】**
- MINIMA：全部在 4×RTX 3090 上训练；LoFTR 总 bs 8、30 epoch；RoMa 总 bs 12（配置为每卡 3）、4 epoch；长边 640 / RoMa 560²。
- AnyMatch：4×RTX 4090；LoFTR 总 bs 4、RoMa 总 bs 2；512²；RoMa 4 epoch。
- LoFTR 官方：复现论文需 MegaDepth 8/16 张 ≥24 GB 的卡；发布的脚本用 4 卡、640²、每卡 bs 1。
- RoMa 官方训练脚本：`N = 32*250000`（25 万步 × bs 32），DINOv2 冻结，AMP；RoMa 论文报告 560² 每对约 199 ms（RTX6000，bs 8）。
- MatchAnything（大规模预训练，不是微调）：16×A100-80G，RoMa 约 6 天。

**【推断】**
- 两类都能在 24 GB 单卡上以小 batch 微调（上面几家都是 24 GB 卡、每卡 1–3 对）。RoMa 前向更重（冻结的 ViT-L + 两遍 560/864），但冻结 DINOv2 后可训练参数只在解码器/refiner。**显存、吞吐的具体数字我没有实测**，需要在服务器上跑一次才算数。
- RL 的额外成本主要是 reward：每次采样都要跑一次 RANSAC（CPU）。两类都能「一次前向、多次采样」，所以组内采样的开销主要在 RANSAC，不在网络。

## 5. 先例：RL 或无 GT 的 detector-free / 稠密匹配器微调

- **RIPE++**（arXiv:2608.19693，2026-08，RIPE 同组）【读到】：把 RIPE 的 RL 从关键点检测扩展到**匹配器 LightGlue**。损失 `L_match = −E[R]`，梯度 `Σ_(i,j) P(i,j|A,B)·r(i,j)·∇Γ_ij`，其中 `P(i,j)` 就是 dual-softmax 分配概率；RANSAC 内点 r=+1、外点 −1；作者说「匹配概率可以闭式算出，不需要采样」。只需要同场景正样本对，不需要位姿或深度。MegaDepth-1500 AUC@5 从 56.58 提到 59.65。**它仍是稀疏匹配**，只在相关工作里提到 LoFTR/RoMa，没有训练它们。
- **FeMIP**（2023）：已在已有调研中，不重复。
- **无 GT 的稠密匹配自监督**：WarpC（ICCV 2021，arXiv:2104.03308）用随机合成 warp 构成三元组的 warp 一致性，无监督训练稠密对应网络【读到摘要】；它不是 RL。MatchAnything 的视频伪标签和 AnyMatch 的 SGVC 都是「用 RoMa 生成或筛选标签」，不是无 GT 训练。arXiv:2607.10082（事件-图像）做无标签的目标域自蒸馏，用对极一致性筛匹配；摘要没说用哪个匹配器，也没有 RL【读到摘要】。
- **【未找到】** 对 LoFTR / ELoFTR / RoMa / DKM 用 REINFORCE / PPO / GRPO 做微调的公开工作（检索截至 2026-09-27）。「RL × 稠密/半稠密匹配」这个空白在 RIPE++ 之后仍然存在，但它离空白更近了一步：dual-softmax 匹配器已经有人用 RL 做了。

## 对选型的含义（事实层面，不做决定）

1. **policy 放在哪**：LoFTR 系有一个网络原生、已受训的「候选对应上的分布」（粗级 dual-softmax + 细级 25 类 softmax），和 RIPE++ 的 LightGlue 目标形式相同，可以直接套用。RoMa 系原生的随机性只在「选哪些像素」（certainty + multinomial）和 8.75 px 间距的粗级分类上，决定 3 px 以内精度的 refiner 是确定性回归，要让位置成为动作得加策略头。
2. **「不可微」卖点两边都成立**：LoFTR 的阈值 + MNN、RoMa 的 multinomial 采样 + KDE 平衡，后面都接 RANSAC，都在 `no_grad` 下，可以作为 RL 的正当理由。
3. **动作空间**：LoFTR 粗级约 4×10⁷ 个配对，但每行是 6400 类的 categorical，可以按行分解；RoMa 约 1.5×10⁶ 个稠密候选，采样 1 万个。两者都远大于 RIPE 的网格化动作空间，方差控制仍然是要做的工作（已有调研的结论不变）。RIPE++ 的闭式期望能绕开 LoFTR 粗级的采样方差，但前提是 reward 能按每个 (i,j) 分配。
4. **能不能接着训**：MINIMA-RoMa、AnyMatch-RoMa、AnyMatch-LoFTR 都有可用的训练代码骨架；MatchAnything-RoMa 没有，要自己补训练循环。
5. **成本**：公开的多模态微调都在 4 张 24 GB 卡上用几个 epoch 完成，两类都不贵。RoMa 前向更重，RL 的开销主要在每次采样后的 RANSAC。
6. **与 baseline 的衔接**：#3 中四个可用模型的瓶颈都在 3 px 以内。LoFTR 系的细级 softmax 正好覆盖这个尺度；RoMa 系对应的是 refiner。这一点影响「RL 能不能直接作用于瓶颈」，属于推断，需要 #5 结合实验来判断。

## 未能核实

- MatchAnything-RoMa 是从头训还是从官方 RoMa 初始化：论文没写。
- MINIMA-LoFTR 使用的 `LoFTR_minima` 子模块没有逐行读，只读了 AnyMatch 的 `LoFTR_AnyMatch`（后者带 MINIMA 式多模态加载器）。
- 两类模型在本任务分辨率下的实际显存和吞吐没有实测。
- FeMIP 的 RL 具体公式（付费墙，沿用已有调研的结论）。
- AnyMatch README 的 arXiv 徽章写 2605.04730，但那是另一篇论文；正确编号是 2606.31077。
