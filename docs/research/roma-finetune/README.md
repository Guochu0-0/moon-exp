# RoMa 系微调：训练损失、可用代码、伪标签与类 RIPE 两条路线的接法

> 调研日期：2026-09-30。回答 [#63](https://github.com/Guochu0-0/moon-exp/issues/63)，为 #64「RoMa 微调代码接入与实测」提供建议。
> 前提：只在现成多模态权重上微调；训练集无标注；LoFTR 上已有两条路线（伪标签 P8、类 RIPE Q4，Test AUC@5 约 0.265），
> 对比对象是 AnyMatch-RoMa zero-shot（Test 0.231）。
> 标注约定：**【读到】** = 直接读到代码或论文原文（commit 与出处见文末「来源」）；**【实测】** = 本次在 126 上用 CPU 读 ckpt 得到；**【推断】** = 我的判断，#64 需实测。
> 不重复 [`matcher-rl-feasibility`](../matcher-rl-feasibility/README.md)（#14）已有内容，只补充和更正。

## 结论速览

1. **三个 RoMa 系权重是同一套网络，都能装进 romatch 接着训。** MINIMA-RoMa 与 AnyMatch-RoMa 的 state_dict 键完全相同；MatchAnything-RoMa 的 ckpt 去掉 `matcher.model.` 前缀后，键也完全相同（603 个）【实测】。公开训练代码的只有 MINIMA（romatch 分叉 + `train_orders/minima_roma.sh`）。AnyMatch 的 RoMa ckpt 里带优化器和调度器状态，可以原样续训。MatchAnything 没有公开 RoMa 的损失。
2. **损失**：粗级 64×64 锚点交叉熵 + 0.01·certainty BCE；5 个 refiner 尺度各有广义 Charbonnier（α=0.5，c=1e-4·scale，归一化坐标）+ 0.01·BCE；从 8 往下只监督「上一尺度误差 < local_dist」的像素。GT 只有两个张量：归一化 warp `x2` (B,h,w,2) 和 0/1 的 `prob` (B,h,w)。
3. **仿射伪标签转成 RoMa GT 有闭式解**：原网格角点约定的 A=[L|t] 换成归一化坐标后是 A_n=[L | (L−I)·1 + t/256]，与分辨率无关，每个尺度直接用；`prob` = 落在 SAR 图内（外加有效像素掩码）。MatchAnything 自己的单应 GT 就是这么做的（`homo_warp_kpts_glue`）。
4. **类 RIPE**：RoMa 里只有两处是分布，一是粗级每格对 4096 个锚点的 softmax，二是 certainty（它决定哪些像素会被采样）。LoFTR 上「外点被压掉」靠的是 dual-softmax 低于阈值，**在 RoMa 里对应的是 certainty，不是锚点概率**。所以 ±1 / −0.25 要同时挂在两处：锚点概率 P_i(k*) 管位置，σ(certainty) 管取舍。refiner 是确定性回归；LoFTR 第二轮三种细级信号都没带来增益，最小实现里 refiner 不接 RL。
5. **显存**：DINOv2 不在 `parameters()` 里，始终冻结（no_grad、fp16）。MINIMA 在 24 GB 的 3090 上以 560²、每卡 bs 3 训练过全部解码器和 VGG【读到】。据此推断，560 下 bs 1–2 在 24 GB 卡上宽裕。864 在训练代码里从没出现过：原训练只跑 560 一遍，864 是推理时的第二遍 upsample。若训 864 这一遍，bs 2 在 24 GB 上大概放不下【推断】。
6. **前作**：iMatching（ECCV 2024，arXiv:2312.02141）用 BA 重投影误差无标注微调了 DKM（RoMa 的前身，同为「GP 粗匹配 + 确定性 refiner」）。这是离本题最近的先例。对 RoMa 本身做无标注或自训练微调的公开工作【未找到】。
7. **给 #64 的建议**：底座用 AnyMatch-RoMa；冻结 DINOv2 和 VGG，只训 decoder（约 1.01 亿参数）；训练分辨率 560，评测照旧 560→864；损失自己重写（romatch 的 `RobustLosses` 里写死了 `wandb.log`）；模块用 `train()` 但所有 BatchNorm 保持 `eval()`。伪标签路线可以**直接复用 LoFTR 老师的仿射标签**（如 `labels_p2`），因为标签是 patch 级仿射，与学生网络结构无关。

---

## 1. 训练代码、能否接着训、损失与超参

### 1.1 三个权重的来源与可续训性

| 权重 | 网络代码 | ckpt 内容【实测】 | 训练代码 | 从 ckpt 续训 |
|---|---|---|---|---|
| MINIMA-RoMa | `LSXI7/RoMa_minima@0d3fd22`（本库 submodule） | 纯 state_dict，603 键，445 MB | **公开**：`MINIMA/train_orders/minima_roma.sh` + `minima_roma_train_config.yaml`（lr_scale 0.1，560²，4 卡 × bs 3，`pre_trained_path: ./weights/roma_outdoor.pth`）【读到】 | 可以：`train_roma_outdoor.py --ckpt_path` 做 strict 加载（`train_roma_outdoor.py:197-200`） |
| AnyMatch-RoMa | `AnyMatch/third_party/RoMa_AnyMatch`，romatch 目录与 RoMa_minima 相比只差 `roma_models.py:169` 一行注释 | 训练 ckpt `{model, n, optimizer, lr_scheduler}`，1.34 GB；`n=200000`，调度器 `last_epoch=50000`（每步 4 个样本），`base_lrs=[2.5e-7, 5e-6]`（encoder / decoder），milestone 225000 未到 | README TODO「Training Code」未勾（`README.md:245`）；数据加载器只有 MINIMA 的 MegaDepth 版 | 可以：`model` 键直接 strict 加载；优化器状态也在，可原样续 |
| MatchAnything-RoMa | HF Space 的 `third_party/ROMA` 分叉；`get_model` 与 romatch 同构（`experiments/roma_outdoor.py:25-180`） | Lightning `{state_dict}`，键前缀 `matcher.model.`，去前缀后与 romatch **完全相同**；445 MB | **未公开**：README「Data generation and training code」「Finetune code」未勾；Lightning 的 `_trainval_inference` 里 RoMa 分支只算 GT（`compute_roma_supervision`），`"Compute losses"` 是 `pass`（`src/lightning/lightning_loftr.py:140-161`） | 能装进 romatch 的 `roma_model` 接着训，但输入口径不同：**不做 ImageNet 归一化**（`NORMALIZE_IMG=False`）、拉伸 resize（`configs/models/roma_model.py:2-3`），训练时必须照这个口径 |

补充【实测】：
- AnyMatch-RoMa 与 MINIMA-RoMa 逐张量相对差 ‖Δ‖/‖W‖ 的中位数是 0.027，两者很接近。MatchAnything-RoMa 与它们的中位相对差约 1.0，是独立训出来的。没有官方 `roma_outdoor.pth` 做对照，所以判断不了 AnyMatch 是从 MINIMA 还是从官方权重起步。
- AnyMatch ckpt 里的实际 lr 是 encoder 2.5e-7、decoder 5e-6。按 `STEP_SIZE·5e-6·lr_scale/8` 反推，每步样本数 4、lr_scale 0.1，与 MINIMA 的配置一致。#14 援引的论文数字（6.25e-8 / 1.25e-6）与 ckpt 对不上，**以 ckpt 为准**。
- 可训练参数共 1.114 亿：`encoder.cnn`（VGG19-BN 前 40 层）1060 万、`decoder.embedding_decoder`（5 层 transformer + to_out）6720 万、`decoder.conv_refiner` 3270 万（16/8/4/2/1 分别为 1745 万 / 1196 万 / 307 万 / 23 万 / 1.2 万）、`decoder.proj` 87 万、`decoder.gps` 1536。DINOv2-L（3 亿）不在 state_dict 里。

romatch 训练脚本里两个要避开的坑【读到】：
- `CheckPoint.save(self, model, optimizer, lr_scheduler, n, epoch)` 要 5 个参数，`train_roma_outdoor.py:268` 只传了 4 个，RoMa_minima 和 RoMa_AnyMatch 在第一次存盘时会 TypeError。上游 RoMa 的 `save` 没有 `epoch`（`RoMa/romatch/checkpointing/checkpoint.py:16-22`）。我们自己写训练循环，碰不到这个坑。
- `RobustLosses` 和 `train_step` 每步都调 `wandb.log`（`robust_loss.py:60,79,86,99`；`train/train.py:20-21,33`），不 `wandb.init(mode="disabled")` 就会报错。上游 RoMa 另外对训练有警告：「Current version of romatch is not tested for training」（`RoMa/romatch/models/matcher.py:295`）。

### 1.2 前向结构（训练态）【读到】

`RegressionMatcher.forward(batch, batched=True)`（`matcher.py:497-514`）把 im_A、im_B 拼成一个 batch 过 encoder，再由 `Decoder.forward`（`matcher.py:333-422`）按 `scales=["16","8","4","2","1"]` 依次处理。以 560² 输入为例：

| 尺度键 | 特征来源 | 网格（560 输入） | 输出 |
|---|---|---|---|
| 16 | DINOv2-L patch14（`encoders.py:114-121`，`torch.no_grad()`、fp16） | 40×40 | GP（`gps["16"]`）→ transformer 解码器 → `gm_cls` (B,4096,40,40) + `gm_certainty`；`cls_to_flow_refine` 取 argmax 锚点及其 4 邻域，按概率加权得到初始 flow（`utils.py:301-323`，**带 `@torch.no_grad()`**）；随后 refiner 16 输出位移与 certainty 增量 |
| 8 / 4 / 2 / 1 | VGG19 的 1/8、1/4、1/2、1 特征 | 70 / 140 / 280 / 560 | ConvRefiner：`flow += scale·Δ/(4·w)`，`certainty += Δc`（`matcher.py:391-406`） |

- 尺度之间 `flow`、`certainty` 都被 `detach`（`Decoder(detach=True)`，`matcher.py:418-420`），所以每个尺度只受自己那一项损失的梯度。
- `gm_cls`、`gm_certainty`、`flow_pre_delta` **只在 `self.training` 为真时才放进输出**（`matcher.py:386,392,396`）。因此不能照搬 LoFTR 底座「全部 eval() 只开梯度」的做法，要先 `model.train()`，再把所有 BatchNorm 设回 `eval()`。`CNNandDinov2.train` 只切换 `self.cnn`（`encoders.py:107-108`）。解码器里的 BN 在 `proj*` 和各 refiner 中，`bn_momentum=0.01`。
- 推理 `match()` 先在 560 上跑一遍 `forward_symmetric`，再把图拉伸到 864 跑一遍 `upsample=True`。第二遍只用尺度 8/4/2/1，不跑 DINOv2、GP 和 transformer，从第一遍的 scale-1 flow 起步（`matcher.py:666-700`）。**训练脚本只跑 560 一遍**，864 这一遍从来没有被训过。

### 1.3 损失 `RobustLosses`（`romatch/losses/robust_loss.py`）【读到】

`train_roma_outdoor.py:230-236` 的配置：`ce_weight=0.01, local_dist={1:4, 2:4, 4:8, 8:8}, local_largest_scale=8, alpha=0.5, c=1e-4`。对每个尺度 s，先由 GT 在该尺度的 (h,w) 上生成 `x2`（归一化 warp）和 `prob`：

- **粗级锚点分类**（s=16，`gm_cls_loss`，`:43-61`）：锚点 G = 64×64 个归一化格点中心（`linspace(-1+1/64, 1-1/64)`），目标类别 = 离 x2 最近的锚点。
  `L_gm = mean_{prob>0.99} CE(gm_cls, argmin_k ‖G_k − x2‖) + 0.01 · BCEwithLogits(gm_certainty, prob)`。
  锚点间距是 2/64，折合 512 原网格 8 px。
- **refiner 回归**（每个 s，`regression_loss`，`:82-100`）：epe = ‖flow − x2‖（归一化坐标），cs = c·s，
  `L_reg = mean_{prob>0.99} cs^α · ((epe/cs)² + 1)^{α/2} + 0.01 · BCEwithLogits(certainty, prob)`。
  α=0.5 时，epe 远大于 cs 时这一项约等于 √(cs·epe)，近似 L^{1/2}。s=1 时 cs=1e-4（归一化），相当于 512 网格上约 0.026 px。
- **局部掩码**（`:138-141`）：s ≤ 8 时 `prob *= (prev_epe < (2/512)·local_dist[s]·s)`，prev_epe 是上一尺度的误差。这里写死了 512，**我们的 patch 恰好是 512，阈值可以直接读成原网格像素**：s=8 为 64 px，s=4 为 32 px，s=2 为 8 px，s=1 为 4 px。
- 总损失：`Σ_s L_reg(s) + L_gm(16)`，各尺度权重都为 1（`:106`）。
- 优化（`train_roma_outdoor.py:237-251`，`train/train.py:23-37`）：AdamW(wd 0.01)；encoder lr = `STEP·5e-6·lr_scale/8`，decoder lr = `STEP·1e-4·lr_scale/8`；MultiStepLR 在 90% 处 ×0.1；fp16 GradScaler，scale 不低于 1；**`grad_clip_norm = 0.01`**（非常小）；总量 `N = 32·250000` 个样本。
- 论文（arXiv:2305.15404 §4.2）：560² 训练，canonical lr（bs 8）decoder 1e-4、encoder 5e-6，DINOv2 全程冻结，refiner 之间 detach；式 13–19 给出上述损失（论文的 c 写作 0.03，按像素计，与代码的归一化 1e-4 口径不同，以代码为准）。

MatchAnything 分叉里的 `robust_loss.py` 与此相同，只差 `meshgrid` 的 indexing 和一处语句顺序（逐行 diff 过）。

---

## 2. 伪标签路线：由 patch 级仿射生成稠密 warp GT 与 certainty 目标

### 2.1 坐标换算（闭式）

本库约定（`finetune/pseudo.py`）：仿射 A=[L|t]（2×3）作用在原 512 网格的**角点约定**坐标上，方向为 光学 → SAR，即 v = L·u + t，其中像素 i 的中心在 i+0.5。

RoMa 的归一化坐标：n 个像素的中心取 `linspace(-1+1/n, 1-1/n)`（`utils.py:333-342`，`matcher.py:715-723`），也就是 x_n = 2u/W − 1，u 为角点约定坐标。拉伸 resize 只是对角点坐标整体缩放，**所以同一个 x_n 在 512、560、864 以及每个尺度的网格上都代表同一个物理位置**。代入 W=512：

```
u = 256·(x_n + 1),  v = L·u + t,  y_n = v/256 − 1
⇒ y_n = L·x_n + [ (L − I)·(1,1)ᵀ + t/256 ]        即 A_n = [ L | (L−I)·1 + t/256 ]
```

核对：A=[I|0] 时 A_n=[I|0]；平移 t=(8,0) px 时，归一化平移为 8/256=2/64，正好一个锚点间距（8 px）。

每个尺度的 GT（替换 `get_gt_warp`）：

```python
def affine_gt_warp(A_n, h, w):            # A_n: (B,2,3)，归一化坐标，光学→SAR
    ys = torch.linspace(-1 + 1/h, 1 - 1/h, h); xs = torch.linspace(-1 + 1/w, 1 - 1/w, w)
    g = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)          # (h,w,2)，(x,y) 顺序
    x2 = torch.einsum("bij,hwj->bhwi", A_n[:, :, :2], g) + A_n[:, None, None, :, 2]
    prob = (x2.abs() < 1).all(-1).float()                               # 落在 SAR 图内
    return x2, prob
```

MatchAnything 对单应 GT 的处理完全一样：`homo_warp_kpts_glue` 对网格点做单应映射，出界的点 valid=0（`src/loftr/utils/geometry.py:256-266`），再在 `get_gt_flow` 里换成归一化坐标（`supervision.py:168-240`）。这说明「几何变换 → 稠密 warp + 0/1 prob」是 RoMa 系训练里现成的 GT 形式【读到】。

### 2.2 重叠区、边界、certainty 目标

- **重叠区**：`prob=1` 当且仅当 y_n 落在 (−1,1)² 内。patch 对的真值偏移是几十 px，重叠率大多在 85–95%【推断】。
- **边界**：可以再收紧成 |y_n| < 1 − 2m/512，m 取几个 px。RoMa 原训练不设边距，这是可选项。
- **无效像素**：如果 SAR 或光学有 nodata（补 0 的区域，包括 P8 式几何增强补出来的 0），把对应的有效掩码按 grid_sample 采到 y_n（SAR 侧）和 x_n（光学侧），与 `prob` 相乘。
- **certainty 目标**：RoMa 的 certainty 学的是「共视」，不是「好不好匹配」。深度 GT 下，无纹理区只要共视，prob 也是 1。仿射标签给出的 prob 语义相同，不需要另造目标。风险在于**标签本身错了**：BCE 会教模型在错误位置上自信。缓解手段有两层：一是沿用 LoFTR 路线的 SCENES 筛选（匹配 ≥ 100、内点 ≥ 20）和课程子集；二是 local_dist 掩码自带一层保护，模型与标签在 s=2 上差 8 px 以上的像素，在 s=2 和 s=1 都不再监督。
- **与原 GT 格式对齐**：只需要 `x2 (B,h,w,2)` 的 (x,y) 顺序归一化坐标和 `prob (B,h,w)` 两个张量，可以原样喂给 `gm_cls_loss` 和 `regression_loss`（只看 `prob>0.99` 和 BCE 目标）。
- **双向**：推理是 symmetric 的（同时用 A→B 和 B→A），训练前向只有 A→B。建议每对以 0.5 概率交换光学和 SAR，交换后用 A⁻¹（同样按 2.1 换算），让 B→A 方向也受监督【推断】。
- **学生侧扰动（P8 配方）**：SAR 叠加已知仿射 T 后，标签变为 T∘A（3×3 相乘），再按 2.1 换算，不需要其他改动。
- **负样本对**（可选）：`prob≡0`，只有 certainty BCE 起作用（CE 和回归都按 `prob>0.99` 取像素，自然为空）。形式上等价于 PWarpC（arXiv:2203.04279）用不同类别的图对监督「无匹配」状态。
- **标签来源**：标签是 patch 级仿射，与网络结构无关。**LoFTR 路线已有的老师标签（如 `labels_p2`，P8 用的第三轮标签）可以直接给 RoMa 学生用**，不必先用 RoMa 自己打一遍。RoMa zero-shot 自己打的标签（Test 0.231）不比 LoFTR 老师（P8 Test 约 0.26）好，可以作为对照【推断】。

---

## 3. 类 RIPE 路线：闭式期望挂在哪里

### 3.1 RoMa 里哪些输出是分布

| 输出 | 形式 | 在推理里起什么作用【读到】 | 梯度能到哪【读到】 |
|---|---|---|---|
| 粗级 `gm_cls` | 每个 40×40 格对 4096 个锚点的 softmax P_i(k) | `cls_to_flow_refine` 取 argmax 锚点 k*，与 4 邻域按概率加权得到初始位置（`utils.py:316-323`）。决定**位置**，而且只是初值，后面还有 5 级 refiner | `cls_to_flow_refine` 是 no_grad，所以只有直接作用在 logits 上的损失才有梯度，能到 embedding_decoder / GP / proj16 |
| certainty | 每像素一个 logit，sigmoid 后是 [0,1]；各尺度的增量相加，尺度之间 detach | `sample()`：先把 σ > 0.05 的**一律置 1**，再多项式无放回抽 4·num，然后用 KDE 平衡再抽 num（`matcher.py:468-495`）；另有 `attenuate_cert` 按粗级 certainty 压低（`:674-680,703`）。决定**取哪些像素**，不改位置 | 每个尺度只能到本尺度的 refiner（及 s=16 的解码器），靠各尺度各自的损失 |
| refiner 位移 | 确定性回归 | 决定 3 px 以内的精度 | 可微，但没有分布 |

### 3.2 LoFTR Q4 在 RoMa 上的对应物

LoFTR 的 dual-softmax P(i,j) 同时管两件事：有没有这个匹配（P 低于阈值就被丢掉）和匹配落在哪里。Q4 里外点的 −0.25 之所以有效，主要是把外点的 P 压到阈值以下，匹配就消失了（参见 runs/C：起步阶段直接塌到「无匹配」）。**RoMa 把这两件事拆开了**：

- 在锚点概率上给外点负分，只会让 P_i 变平，argmax 往往不变，输出几乎不动。它主要的作用是正分那一侧：让内点的 argmax 更尖锐，并稳定初值。
- 真正决定外点会不会被采到的是 certainty。

因此建议的最小形式（RIPE++ 式闭式期望 −E[R]，不采样）【推断】：

```
设 当前步：dense warp（560 上 scale 1，或推理同款的 864）→ sample 出 num 个匹配 → 仿射 RANSAC（3 px，原网格）→ A_t
逐像素 reward r_p：‖warp(p) − A_t(p)‖ < 3 px → +1，否则 −0.25（与 Q4 同）；RANSAC 失败的对全部 −0.25；
                   负样本对：内点 −1，外点 0
(a) 位置项   L_cls  = − Σ_{i∈40×40, σ(c_i)>0.05} P_i(k*_i) · r̄_i / N_c       r̄_i = 该格内像素 reward 的均值（或格中心的值）
(b) 取舍项   L_cert = − Σ_s Σ_p σ(c_p^s) · r_p^s / N_s                      r^s = 把 r 最近邻下采样到尺度 s
L_ripe = L_cls + λ_cert · L_cert
```

- (a) 与 LoFTR Q4「只给实际输出的匹配打分」一致。k* 就是 `cls_to_flow_refine` 里的 mode，用同一次前向的 logits 算。
- (b) 是 certainty 这个 Bernoulli「选取」策略的闭式期望。它比 LoFTR 更方便：dense warp 让**每个像素**都能直接算 reward，不只是被采到的那 1 万个，也不用重跑 RANSAC。注意 0.05 的阈值：只有 σ 降到 0.05 以下，外点才真正采不到。σ 在 0.05 以上时，这一项只影响 `attenuate_cert` 和排序。
- (b) 在形式上接近「以内点掩码为目标的 certainty BCE」（即在线伪标签的 certainty 项）。区别在于它只用本步自洽的 RANSAC 结果，不引入外部仿射。

### 3.3 refiner 怎么办

- #14 的结论不变：refiner 没有分布，要当 RL 动作就得自己加高斯策略头，做法与 `finetune/rl.py` 相同，refiner 的 flow 就是均值 μ【读到 #14】。
- 但 LoFTR 第二轮里，三种细级信号（Q1 闭式在线 l2、Q2 逐匹配高斯 RL、Q3 整对 NCC RL）**全部没有带来增益**（Q1 还降到 Test 0.225，`runs/Q/notes.md`@round2-51-53）。所以最小实现里，**类 RIPE 路线不给 refiner 的位移接任何 RL 项**：只训粗级解码器，以及各尺度 certainty 头的取舍项；也可以干脆冻结 refiner 16…1 的位移输出，作为一个因素比较。
- RoMa v2 的 refiner 会逐像素预测 2×2 精度矩阵 Σ⁻¹（Cholesky 参数化，`romav2/refiner.py:190-222`；论文 §3.3 用残差 NLL 训练），**天然是一个高斯位置分布**。3 px 盘内的 reward 期望在高斯下有近似闭式解。不过 v2 没有公开训练代码（仓库里没有任何 loss 或 optimizer，见第 4 节），这只是记下的可能性。

### 3.4 塌缩风险

RoMa 的 warp 是回归出来的，「恒等 warp + 高 certainty」同样是一条与内容无关的捷径。#49 的结论是负样本对能压住往恒等的漂移。RoMa 上负样本对的作用点更明确：负样本对上的 certainty 应该趋近 0，所以 (b) 的负样本项也可以直接写成 BCE(certainty, 0)。建议照搬 C2/Q4 的监控：负样本对上的恒等匹配数、估出的仿射相对 [I|0] 的位移【推断】。

---

## 4. 显存与速度

### 4.1 能确定的事实【读到 / 实测】

- DINOv2-L 以 `self.dinov2_vitl14 = [model]` 的方式藏在 list 里，不注册为参数（`encoders.py:104`），前向在 `torch.no_grad()` 下（`:115-121`）并转成 fp16。**DINOv2 在所有 romatch 训练里都是冻结的**，RoMa 论文 §4.2 也这么说。MatchAnything 分叉相同（`third_party/ROMA/roma/models/encoders.py:105,125`）。RoMa v2 冻结的是 DINOv3（`romav2/features.py:59-64`）。
- 公开的训练规模：MINIMA-RoMa 4×3090（24 GB），**560²，每卡 bs 3**，训练 decoder 和 VGG（`minima_roma_train_config.yaml`）；AnyMatch-RoMa 每步 4 个样本（ckpt 反推），论文写 4×4090；RoMa 原版 560²、bs 8/卡，推理约 199 ms/对（RTX6000，论文 §4.5）。
- 本库服务器（本次核对）：**126 是 4×V100-SXM2-32GB**（票里写的「126 TITAN RTX」应为 154，见 `docs/agents/servers.md`），不支持 bf16，romatch 默认用 fp16 AMP，不受影响。126 上已有 `dinov2_vitl14_pretrain.pth`、`vgg19_bn` 的 hub 缓存，以及三个 RoMa 系权重。

### 4.2 估算【推断，#64 须实测】

静态部分：DINOv2 fp16 0.6 GB，romatch 参数 fp32 0.45 GB；只训 decoder 时 AdamW 状态加梯度约 1.2 GB；CUDA 上下文约 0.5 GB，合计约 3 GB。激活按各尺度 ConvRefiner（8 个 hidden block，通道数：s16 1377@40²，s8 1137@70²，s4 569@140²，s2 144@280²，s1 24@560²）和 VGG 粗算：

| 配置 | 估计峰值 | 24 GB 卡 |
|---|---|---|
| 560，bs 1，VGG 冻结，decoder 训练 | 6–8 GB | 可以 |
| 560，bs 2，同上 | 9–13 GB | 可以 |
| 560，bs 3，训 VGG（MINIMA 实际配置） | 已知能在 24 GB 上跑 | 可以【读到】 |
| 560 一遍加 864 upsample 一遍都训，bs 1 | 13–17 GB | 大概可以 |
| 同上，bs 2 | 24–30 GB | 大概放不下；V100 32 GB / A6000 可以 |

- 864 在 DINOv2 的 patch14 上不能整除（864/14 不是整数），粗级不能在 864 上跑。864 只可能作为 refiner 8…1 的 upsample 一遍来训，而且要从 560 一遍 detach 后的 flow 起步（`matcher.py:696-700` 的做法）。
- 速度：560 一步（前向加反向）在 V100 / TITAN RTX 上估计 0.4–0.8 s，A6000 更快。类 RIPE 路线另外要做一次 1 万点仿射 RANSAC（CPU），与 LoFTR 路线同量级。可以采 5000 点（MatchAnything 的默认值）来减负。

---

## 5. 前作：对 RoMa 类稠密匹配器做无标注或自训练微调

| 工作 | 匹配器 | 无标注信号 | 与本题的关系 |
|---|---|---|---|
| **iMatching**，ECCV 2024，[arXiv:2312.02141](https://arxiv.org/abs/2312.02141) | CAPS、Patch2Pix、ASpanFormer、**DKM**（「We experiment with CAPS, Patch2Pix, ASpanFormer, and DKM」） | BA 重投影误差当监督（双层优化，隐式梯度），视频无位姿、无深度；iDKM 全部参数更新 | **最近的先例**：DKM 与 RoMa 同为「GP 粗匹配 + 确定性 refiner」。信号是几何一致性，相当于「本步几何模型 → 对应点目标」，与我们在线仿射目标同类；TartanAir 上 iDKM 位姿 AUC@5° 从 58.2 升到 76.7 |
| SCENES，[arXiv:2401.10886](https://arxiv.org/abs/2401.10886) | LoFTR 等 | 极线伪监督 + 自训练 | LoFTR 路线 S1 的出处，不涉及 RoMa |
| WarpC（ICCV 2021，[arXiv:2104.03308](https://arxiv.org/abs/2104.03308)）/ PWarpC（CVPR 2022，[arXiv:2203.04279](https://arxiv.org/abs/2203.04279)） | GLU-Net / PDC-Net 等稠密匹配器 | 已知随机 warp 构成三元组的 warp 一致性；PWarpC 另用不同类别的图对监督「无匹配」 | 相当于我们「已知几何增强 + 一致性」和「负样本对 → certainty 0」这两招的来源 |
| MatchAnything（arXiv:2501.07556） | RoMa / ELoFTR 预训练 | DL3DV 视频用 RoMa 跟踪生成伪标签 | 伪标签用于**大规模预训练**，不是目标域无标注微调 |
| AnyMatch（arXiv:2606.31077） | RoMa 做 SGVC 质检 | RoMa 算 PCK 筛合成数据 | RoMa 当筛子用，不是被微调的对象 |
| Yi et al.，[arXiv:2607.10082](https://arxiv.org/abs/2607.10082) | **检测器式**（事件网络 + 冻结 SuperPoint），不是 RoMa | 师生 + 已知单应 + 基础矩阵 RANSAC 置信度加权 | 读了正文（§3.4、§5）：明确说是 detector-based，#14 里「没说用哪个匹配器」这一条可以更正 |
| RoMa-Ω（arXiv:2609.09507） | RoMa v2 换 VGGT-Ω 骨干 | 有监督重训 | 无关 |
| 光学–SAR 上的 RoMa 微调（如 MDPI Remote Sensing 18(10):1662，2026；rsim，arXiv:2604.10217） | RoMa | 前者用有标注的 SAR patch 集微调（正文 403，只读到摘要级信息）；后者只评测 | 不是无标注 |

**【未找到】** 对 RoMa（v1 或 v2）做无标注、自训练或 RL 微调的公开工作（检索截至 2026-09-30）。

---

## 6. 给 #64 的建议（最小方案）

**底座**：AnyMatch-RoMa（`anymatch/RoMa_AnyMatch.pth`，取 `['model']`；代码 `third_party/AnyMatch/third_party/RoMa_AnyMatch`，与 baseline 适配器同一份）。理由：zero-shot 最好（Val 0.236 / Test 0.231），训练代码骨架现成，lr 口径可以从 ckpt 读出。MatchAnything-RoMa（Test 0.226）作为第二底座：同一份 romatch 代码，去掉 `matcher.model.` 前缀即可加载，但**输入不做 ImageNet 归一化**。MINIMA-RoMa（Test 0.214）不作优先。
另外注意：`runs/B0m` 里 AnyMatch-RoMa 换成 minmax 输入映射后，Val 从 0.236 升到 0.262。先定好输入映射再微调，否则微调增益会和映射增益混在一起。

**封装**（照 `finetune/model.py` 的思路，新增 RoMa 版 Base）：
1. 构造与权重加载直接复用 `RomatchAdapter`（`roma_outdoor(weights=sd)`），保证训练的网络就是评测的网络。
2. `model.train()`，然后把全部 `nn.BatchNorm2d` 设回 `eval()`（原因见 1.2：训练态才输出 `gm_cls`，但 bs 1–2 不能更新 BN 统计量）。
3. 冻结 `encoder.cnn`（VGG），只训 `decoder`（1.01 亿参数）。这是最小改动：原训练里 VGG 的 lr 本来就只有 decoder 的 1/20，冻结后也省显存。VGG 解冻作为后续因素。
4. 输入：512 patch 用张量双线性拉伸到 560²，灰度复制成 3 通道，按底座决定是否做 ImageNet 归一化。训练前向用 `model.forward({"im_A","im_B"}, batched=True)`（非 symmetric），以 0.5 概率交换两图方向。
5. 损失：**自己重写** `RobustLosses`，约 60 行，照抄 1.3 的公式和默认值，不 import wandb；GT 用 2.1 的 `affine_gt_warp`。
6. 优化：AdamW，wd 0.01；decoder lr 从 AnyMatch 末态 5e-6 起扫（1e-5 与本库 LoFTR 配方同量级），warmup 500 + cosine；fp16 GradScaler。梯度裁剪：romatch 原配置是 0.01，建议 #64 同时记录裁剪前的梯度范数，再决定用 0.01 还是本库的「不裁」。
7. ckpt 存成 `{'model': state_dict}`，`RomatchAdapter(weights_key="model")` 直接能读。评测走原链路（560→864、symmetric、采 10000 点）。**lr=0 验收**：存出来的 ckpt 经适配器推理，应与 B0 的 `anymatch_roma` 逐点一致。随机采样要重置种子，这一点 runner 已经做了。

**分辨率**：训练 560（与原训练相同，覆盖粗级和 5 级 refiner）；推理保持 560→864。第二阶段可选：no_grad 跑 560 一遍，只训 864 这一遍的 refiner 8…1，作为一个因素，需要 bs 1 或 32 GB 以上的卡。

**两条路线的最小实现**：
- 伪标签（P8 对应物）：标签直接用 `labels_p2` 前 50%，加 geo + photo 学生扰动（标签变为 T∘A），全 decoder，RoMa 原损失。对照组：用 RoMa zero-shot 自己打的标签。
- 类 RIPE（Q4 对应物）：3.2 的 L_cls + λ·L_cert，外点 −0.25，加负样本对，只训 decoder（refiner 的位移可冻结，作为因素）。对照组：随机 reward（同 C1p），因为「无匹配 / 全 certainty 为 0」同样可能是吸收态。

**#64 要实测的数字**：560 下 bs 1 / 2 的峰值显存与每步耗时（126 V100、154 TITAN RTX、A6000）；上表中 864 那两行；每步 RANSAC 耗时。

---

## 未能核实

- AnyMatch-RoMa 是从 MINIMA-RoMa 还是从官方 RoMa 起步（缺官方 `roma_outdoor.pth` 做三方对照）；论文写的 lr 与 ckpt 不一致的原因。
- MatchAnything-RoMa 的训练损失（代码里没有）；它是否用了与 romatch 相同的 `RobustLosses` 配置。
- 显存与速度全部是估算，没在 GPU 上跑（本次按约定不跑 GPU 任务）。
- MDPI 那篇 SAR 上的 RoMa 微调只读到摘要级信息（正文 403）。

## 来源

代码（本次 `git clone --depth 1` 后按下列 commit 读取，行号对应这些 commit）：

| 仓库 | commit | 读过的文件 |
|---|---|---|
| [LSXI7/RoMa_minima](https://github.com/LSXI7/RoMa_minima) | `0d3fd22`（本库 submodule） | `train_roma_outdoor.py`、`romatch/losses/robust_loss.py`、`romatch/models/{matcher,encoders}.py`、`romatch/models/model_zoo/{__init__,roma_models}.py`、`romatch/utils/utils.py`、`romatch/train/train.py`、`romatch/checkpointing/checkpoint.py`、`romatch/__init__.py`、`romatch/models/transformer/__init__.py` |
| [MnYangs/AnyMatch](https://github.com/MnYangs/AnyMatch) | `259ad34` | `README.md`；`third_party/RoMa_AnyMatch/romatch/` 与 RoMa_minima 整目录 diff |
| [LSXI7/MINIMA](https://github.com/LSXI7/MINIMA) | `796e772` | `README.md`、`train_orders/minima_roma.sh`、`train_orders/minima_roma_train_config.yaml` |
| [Parskatt/RoMa](https://github.com/Parskatt/RoMa) | `77f8d68` | `romatch/checkpointing/checkpoint.py`、`romatch/models/matcher.py:295`、`romatch/losses/robust_loss.py`（与 minima 相同） |
| [HF Space LittleFrog/MatchAnything](https://huggingface.co/spaces/LittleFrog/MatchAnything) | `6a7bcb5` | `imcui/third_party/MatchAnything/`：`README.md`、`configs/models/roma_model.py`、`src/config/default.py`、`src/lightning/lightning_loftr.py`、`src/loftr/utils/{supervision,geometry}.py`、`third_party/ROMA/roma/{matchanything_roma_model.py,losses/robust_loss.py,models/encoders.py,models/matcher.py}`、`third_party/ROMA/experiments/roma_outdoor.py` |
| [Parskatt/RoMaV2](https://github.com/Parskatt/RoMaV2) | `95c9968` | `README.md`、`src/romav2/{romav2,refiner,features,matcher}.py`（全仓无 loss / optimizer） |

ckpt【实测】：126 上 `YGC/weights/{anymatch/RoMa_AnyMatch.pth, minima/minima_roma.pth, matchanything/matchanything_roma.ckpt}`，CPU 读取键、参数量、优化器与调度器状态、逐张量相对差。

本库：`finetune/{train,model,pseudo,rl,label,data}.py`（main `0a8b9e6`）；`finetune/coarse.py`、`runs/{C,P,Q}/notes.md`（分支 `round2-51-53@23d7c18`、`ripe-coarse-49@adc704a`）；`runs/B0/metrics.json`、`runs/B0m/metrics.json`；`baselines/adapters/{romatch,matchanything}.py`；`configs/baselines/{anymatch,ma,minima}_roma.json`。

论文（arXiv HTML 版）：
- RoMa：Edstedt et al., CVPR 2024，[arXiv:2305.15404](https://arxiv.org/abs/2305.15404)（§4.2 训练设置，式 13–19 损失，§4.5 耗时）。
- RoMa v2：Edstedt et al., [arXiv:2511.15706](https://arxiv.org/abs/2511.15706)（§3.2–3.5 损失与训练、预测精度矩阵、冻结 DINOv3）。
- iMatching：Zhan et al., ECCV 2024，[arXiv:2312.02141](https://arxiv.org/abs/2312.02141)（实验节：CAPS / Patch2Pix / ASpanFormer / DKM）。
- PWarpC：Truong et al., CVPR 2022，[arXiv:2203.04279](https://arxiv.org/abs/2203.04279)（摘要）。
- Yi et al., [arXiv:2607.10082](https://arxiv.org/abs/2607.10082)（§3.4、§5）。
- RoMa-Ω，[arXiv:2609.09507](https://arxiv.org/abs/2609.09507)（摘要）。
- 另见 #14 的 sources.md（MINIMA arXiv:2412.19412、AnyMatch arXiv:2606.31077、MatchAnything arXiv:2501.07556、RIPE++ arXiv:2608.19693、SCENES arXiv:2401.10886）。
