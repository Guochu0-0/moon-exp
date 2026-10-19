# RoMa 代码中可插入随机性的位置与采样开销

> 调研日期：2026-10-19。回答 [#163](https://github.com/Guochu0-0/moon-exp/issues/163)（子地图 #161）。
> 一手来源：本仓库 `finetune/`、`baselines/adapters/romatch.py`、`moonlib/ransac.py`；romatch 代码取 `third_party/RoMa_minima`（0d3fd22，AnyMatch 的 `RoMa_AnyMatch` 与它只差一行注释，见 `baselines/adapters/romatch.py:1-2`）。下文 `matcher.py` 指 `third_party/RoMa_minima/romatch/models/matcher.py`，`roma_models.py`、`utils.py` 同理在 `romatch/` 下。
> 标注：**【读到】** = 读代码确认；**【实测】** = 本地 RTX 4060 Laptop（8 GB，另有 4.4 GB 被占用）上用**随机权重**、同结构的模型量的；**【推断】** = 我的判断。

## 0. 结论速览

1. **粗级 `gm_cls` 是现成的分布**，最适合当第一个策略：每个 1/14 格子上 4096 类（64×64 锚点）的 logits，已用交叉熵训过。romatch 自带随机版 `cls_to_flow(deterministic_sampling=False)`。推理照旧取 mode，零开销。锚点间距在原网格是 8 px，所以它只管「落在哪个锚点附近」，3 px 以内的精度由下游 refiner 决定。
2. **refiner 的位移是确定性回归，没有分布**。不过仓库里已有一个**只在训练时挂的不确定度头** `unc_head`（#127），输出两分量拉普拉斯混合的参数，正好能当高斯/拉普拉斯策略头。它不存进 `'model'` 权重，推理结构不变。
3. **匹配点采样（certainty → multinomial）是网络输出上现成的分布**，但推理时把 σ(c) > 0.05 的像素都截成 1，所以 certainty 只决定「过不过 0.05」。按无放回多项采样精确算 log π 很麻烦，要改成逐像素伯努利或有放回采样来近似。
4. **仿射拟合（cv2 RANSAC）的随机性不由网络参数化**，本身不是策略，只能当环境或 reward 的一部分。评测口径固定，按 #161 也不在范围内。
5. **复用 backbone**：编码器（VGG + DINOv2）每对只算一次，解码器沿 batch 维 `expand` 成 N 份即可，本地已跑通。**带梯度的解码器显存大致随 N 线性增长**（每份约 5 GB【实测】）。省显存的办法：只对产生 log π 的那一段留计算图，rollout 的其余部分放在 `no_grad` 里跑（§2）。
6. 每对图本来就有 5000～10000 个点，每个点是一个动作。**逐点 reward 一次 rollout 就有几千个样本**；只有整对标量 reward（如合成扰动的恢复程度）才需要 N 份 rollout 做组内基线。

## 1. 前向结构与张量形状（训练态，560 一遍，bs B）

【读到】`finetune/models/roma.py:215-217` 调 `self.model(...)`，即 `RegressionMatcher.forward`（`matcher.py:497-514`，**非 symmetric**）→ `extract_backbone_features`（`matcher.py:458-466`，两图 cat 后过编码器）→ `Decoder.forward`（`matcher.py:333-422`）。尺度 16 用 DINOv2 特征，8/4/2/1 用 VGG 特征。各尺度的输出形状如下【实测】，与代码一致：

| 尺度 | 网格 | corresps 内容 |
|---|---|---|
| 16 | 40×40 | `gm_cls` (B,4096,40,40)、`gm_certainty` (B,1,40,40)、`flow_pre_delta`、`delta_flow` (B,2,40,40)、`certainty`、`flow` |
| 8 / 4 / 2 / 1 | 70 / 140 / 280 / 560 | `flow_pre_delta`、`delta_flow` (B,2,h,w)、`certainty` (B,1,h,w，logit)、`flow` (B,2,h,w) |

几个关键事实【读到】：
- `flow` 是 B 图上的归一化坐标，取值 (−1,1)，像素中心约定见 `finetune/models/roma.py:11`。原网格 512 px（`finetune/geom.py:10`）。
- `detach=True`（`roma_models.py:149`）：各尺度之间传递的 `flow`、`certainty` 都被 detach（`matcher.py:418-420`），而且粗级 `cls_to_flow_refine` 带 `@torch.no_grad`（`utils.py:301-323`）。**每一级的输出只通过特征和本级参数接收梯度，不会穿过上一级的 flow 回传。** 所以「每级采样一次」构成一个顺序策略，log π 可以按级相加，各级之间不需要重参数化【推断，由前述代码直接推出】。
- 推理（`matcher.py:611-748`）多了三样：symmetric（A→B、B→A 拼在一起），864 上采样那一遍（拿 560 一遍尺度 1 的 flow 当初值，重跑 8…1），以及 `attenuate_cert`。训练默认只训 560 一遍（`train_res`，`finetune/models/roma.py:36`）。
- `Decoder` 构造参数里有 `warp_noise_std`、`displacement_dropout_p`、`gm_warp_dropout_p`（`matcher.py:278,296-299`），**但前向从没用过**（grep 结果只有赋值），不是现成的噪声开关。

## 2. 逐个候选位置

### A. 粗级 warp 分类 `gm_cls` 与 `gm_certainty`

- **位置**：`matcher.py:373-386`。transformer 解码器输出 4097 通道，切成 4096 类 logits 加 1 通道 certainty（`romatch/models/transformer/__init__.py:43-46`；`roma_models.py:26-33`）。由 `cls_to_flow_refine` 转成 flow：在 mode 及其 4 邻域上做 softmax 加权平均（`utils.py:316-323`）。
- **含义**：A 图每个 14 px 块（560 输入）落在 B 图 64×64 锚点中哪一个。锚点间距为 2/64（归一化）= 8 px（原网格 512）。`gm_certainty` 是该格可匹配的 logit。
- **分布**：**已有**，Categorical(softmax(gm_cls))。romatch 自带采样版 `cls_to_flow(cls, deterministic_sampling=False)`（`utils.py:285-299`，`torch.multinomial`）。log π = `log_softmax(gm_cls)[k_sampled]`，带梯度。`gm_certainty` 可以当伯努利，但它只在尺度 16 进入 certainty 累加。
- **现有用法**：`finetune/parts/cexp.py:171-175` 的闭式期望用的就是 P_i(k*)（mode 的概率），属于 #66 的「类 RIPE」，不算 RL。
- **采 N 份**：采样发生在 GP 和 transformer 之后，**粗级只需算一次**。N 份锚点场各自往下跑 refiner 16…1，得到 N 份 warp。log π 只依赖 `gm_cls`，所以带梯度的只有粗级这一段（一份），**refiner 的 N 份 rollout 可以放在 `no_grad` 里**。显存约为一次带梯度的前向加 N 次无梯度 refiner【推断，由 detach 结构推出】。实现上要把 `Decoder.forward` 拆开，或者在尺度 16 分类后插一个钩子。
- **推理**：保持 `cls_to_flow_refine`（mode）即可，**完全去掉，零开销**【读到】。
- **注意**【推断】：采到离 mode 较远的锚点时，尺度 16 refiner 的局部相关半径是 7 格（`roma_models.py:51`），能否拉回取决于 refiner。而采样偏离 mode 本身就会给 refiner 制造训练时没见过的初值。

### B. refiner 各级位移 `delta_flow`

- **位置**：`ConvRefiner.forward`（`matcher.py:105-143`）输出 3 通道 = 2 维位移 + 1 维 certainty 增量。累加在 `matcher.py:391-402`：`flow += s·delta/(4·560)`（归一化），即 560 输入上每单位 delta 对应 s/8 px。尺度 16/8/4/2/1 各一个 refiner（`roma_models.py:40-114`）。
- **分布**：**原生没有**，是确定性回归。**仓库已有可用的头**：`[model] unc_head`（`finetune/models/roma.py:47-48,136-145`）在每个 refiner 的 `out_conv` 输入上接 1×1 卷积，出 2 通道（混合权重 logit、β），放进 `corresps[s]["unc"]`。配套的 `mixlap_nll`（`finetune/parts/pseudo.py:388-398`，SEA-RAFT 2405.14793 §3.2）定义了两分量拉普拉斯混合：一个分量尺度固定 1 px，另一个尺度 e^β，β ∈ [0,10]。把它从「对目标的似然」改成「以 flow 为中心的策略」：采样 a = flow + ε，ε 服从该混合分布，log π 直接可算。缺点是这个头的初始化为零（`roma.py:140-141`，起点两分量各半、β = 0），没经过训练。另一个选择是固定 σ 的高斯，不加任何参数【推断】。
- **采 N 份**：
  - **只在尺度 1（最后一级）加噪声**：下游没有网络，一次带梯度的前向就能抽 N 份噪声，**额外开销≈0**，只多 N 次匹配采样和 RANSAC。log π 的梯度通过均值 flow 和 unc 头回传到尺度 1 的 refiner，以及经特征回传到 LoRA。
  - **在中间尺度（16/8/4/2）加噪声**：后续 refiner 的输入随样本变化，要跑 N 份。各级 log π 又依赖该样本下这一级 refiner 的输出（要带梯度），**显存随 N 线性增长**。本地实测，复用粗级、只带梯度重跑 8…1 一份约 +3.2 GB，两份 +6.2 GB【实测】。
- **推理**：去掉噪声、不跑 unc 头即可。unc 头不进 `'model'` 权重，评测适配器不读（`roma.py:246-247`）。**零开销**【读到】。
- **注意**【推断】：推理时 864 那一遍会从 560 的尺度 1 结果重新细化 8…1。如果策略只训 560 一遍，最后一级的改进未必能传到最终输出；可以考虑 `train_res = 864/both`（#122 已支持）。

### C. 匹配点采样（certainty → multinomial）

- **位置**：推理用 `RegressionMatcher.sample`（`matcher.py:468-495`）：σ(c) > 0.05 的像素截成 1，无放回多项采样 4×num 个，再按 KDE 密度倒数做第二次多项采样，取 num = 10000（适配器默认，`baselines/adapters/romatch.py:76`）。certainty 来自 `matcher.py:703,724`，已减去 `attenuate_cert` 项、取了 sigmoid，越界的落点置 0（`matcher.py:726-728`）。训练里的同口径简化版是 `cexp.sample_matches`（`finetune/parts/cexp.py:127-136`）：不做 KDE，num = 5000。
- **含义**：从 560²（推理为 864²×2，symmetric）个像素里挑哪些点送进 RANSAC。certainty 在 `matcher.py:400-402` 逐级累加，最终为 (B,1,560,560) 的 logit。
- **分布**：**已有**，但有两个问题【读到 + 推断】：(1) 截断阈值把 σ(c) > 0.05 的像素都变成等概率，截断后对 c 没有梯度；(2) 无放回多项采样的精确 log π 是 Plackett–Luce 形式，算起来麻烦。可行的近似是逐像素伯努利 π(选 i) = σ(c_i)，或用截断前的概率有放回采样。
- **采 N 份**：只依赖一份 certainty 图。抽样本身约 1 ms/次（5000 / 313600）【实测】，**几乎免费**。真正的开销在每份样本之后的 RANSAC（见 D）。
- **推理**：采样代码不变，无额外开销。但策略训练会改变 certainty 的含义，推理时仍然先截断再做 KDE，训练和推理的采样口径不一致【推断】。
- 「certainty 全 0 成为吸收态」见 `cexp.py:140-141` 的注释（#66 的 M5 实测），在这里同样适用。

### D. 仿射拟合（RANSAC）

- **位置**：`moonlib/ransac.py:16-27`，即 `cv2.estimateAffine2D`（RANSAC，阈值 3 px，confidence 0.99999，maxIters 10000，**每次调用前固定种子 0**）。评测（`baselines/fit.py:56`）与训练（`cexp.py:153`）共用。
- **分布**：**没有由网络参数化的分布**。种子固定，结果是采样点的确定函数，本身构不成策略。如果要做成策略，得自己写可学习打分的假设采样（DSAC 式，arXiv:1611.05705），但「在打标或评测时换仿射拟合方式」不在 #161 的范围内，所以它在本图里只当 reward 计算的一部分【推断】。
- **开销**【实测，本机 CPU，5000 点】：内点为主时约 6 ms，50% 外点约 9 ms，95% 外点约 98 ms。每对每份样本一次，CPU 串行。N 份 × B 对在外点多时会成为瓶颈，可以多线程并行。
- **推理**：评测本来就在跑，不变。

## 3. 复用 backbone 的实测（随机权重，560，bs 1，fp16 autocast，BN eval，只有 decoder 求梯度）

| 项 | 耗时 | 显存峰值增量 |
|---|---|---|
| 编码器（VGG+DINOv2，两图，no_grad） | 230 ms | +0.64 GB |
| 解码器全段，N=1 / 2，no_grad | 1.1 / 2.3 s | +1.40 / +1.43 GB |
| 解码器全段，N=1 / 2，带梯度 | — | **+5.1 / +9.9 GB**（N=4 OOM） |
| 复用粗级、只跑 8…1，N=1 / 2，带梯度 | — | +3.2 / +6.2 GB |
| 复用粗级、只跑 8…1，N=8，no_grad | — | +1.43 GB |

- 编码器特征按 batch 维 `expand` 给解码器，可以跑通，**能复用**【实测】。按 #161 的固定设置，DINOv2 带 LoRA，此时编码器那一段也带梯度，但只算一次，反向一次。
- **耗时只能看相对关系，绝对值不可信**：本机显存不够，带梯度的几项超出 8 GB 后溢出到共享内存，N ≥ 2 的耗时明显失真，所以表里不列。服务器上的参照值见 #64：3090、bs 1 伪标签训练 7 GB、0.43 s/步。
- **显存是可靠的结论**：带梯度的解码器约 5 GB 一份，随 N 线性增长；无梯度的 rollout 几乎不随 N 增长。所以 24 GB 卡上，「N 份都带梯度」大约只能到 N≈3，而 A（只给粗级留计算图）和 B 的尺度 1 方案不受这个限制【推断，由实测显存外推】。

## 4. 对 #165 的提示【推断】

- 开销最低、推理零改动的组合：粗级 Categorical（A），加尺度 1 的噪声头（B，复用 `unc_head`），再加逐点 RANSAC 内点 reward。只有一份带梯度的前向，N 份 rollout 都是无梯度的 refiner。
- 匹配点采样（C）可以作为「取舍」的策略，但要先解决截断和无放回采样的 log π 近似问题，还要处理与推理口径不一致。
- 中间尺度加噪声的显存线性增长，除非有理由认为瓶颈在 4/2 级，否则不建议放在第一批。

## 5. 未核实

- 用真实权重时，粗级采样偏离 mode 后 refiner 能拉回多少，需要在服务器上做一次无训练的测试才知道。
- 服务器（3090/4090）上各段的真实耗时比例。
- 本地实测用的是 `RoMa_minima` 结构加随机权重，没有加载 AnyMatch 权重，也没有挂 LoRA。
