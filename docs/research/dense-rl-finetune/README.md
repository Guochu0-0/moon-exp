# 稠密 / 连续输出上的 RL 微调：策略形式、算法、无标注 reward 与防塌缩

> 调研日期：2026-10-19。回答 [#162](https://github.com/Guochu0-0/moon-exp/issues/162)（子地图 [#161](https://github.com/Guochu0-0/moon-exp/issues/161)）。
> RL 的定义沿用子地图：从模型输出的分布里**采样**，用**不可微** reward 打分，按**策略梯度**更新。闭式期望不算。
> 标注约定：**【读到】** = 读过论文原文或摘要原句（链接见 [sources.md](sources.md)）；**【推断】** = 我的判断。
> 旧调研（[`rl-registration-survey.md`](../rl-registration-survey.md)、[`label-free-reward/`](../label-free-reward/README.md)、[`matcher-rl-feasibility/`](../matcher-rl-feasibility/README.md)）只当线索；本文引用到的结论都重新对过原文，有出入的地方在 §6 列出。

## 0. 结论速览

1. **没有找到对 RoMa / DKM / LoFTR 这类稠密匹配器做 RL 微调的论文**（检索截至 2026-10-19）。最近的先例分三类：稀疏关键点/匹配（DISK、RIPE、RIPE++）、扩散/流模型生成（DDPO、Flow-GRPO）、医学形变配准（MorphSeek）。「稠密匹配 + 严格 RL」仍是空白。
2. **连续输出变成可采样分布，文献里基本只有一种做法：给确定性输出加高斯噪声当策略，推理时取均值。** DDPO 和 Flow-GRPO 在每一步去噪上用各向同性高斯；MorphSeek 在编码器特征上加一个均值和方差头。均值和方差的参数化方式不同，核心都是「高斯 + 推理取均值」。RoMa 的 refiner 是确定性回归，套这个做法不增加推理开销。
3. **方差的关键是「一个 reward 摊到多少维」。** score-function 梯度（REINFORCE）的方差随被采样变量的维数增长（Mohamed et al., JMLR 2020）。一个整对的标量 reward 摊到上万个像素，方差会很大。文献的对策有三种：组内相对优势（GRPO / RLOO）、把 reward 拆到局部（SGA-Flow-GRPO 按空间分配 credit；DISK、RIPE++ 按单个匹配给 reward）、缩小动作空间（DISK、RIPE 每个网格只采一个点）。
4. **算法上，「组内基线 + 单次更新 + 对参考模型的 KL」是视觉 RL 微调最常见的配置。** 从强预训练模型出发时，PPO 的 critic 和多轮裁剪更新不是必需的（RLOO 论文；Pinto et al. 只用 REINFORCE 加一个样本作 baseline）。KL 有没有用，文献意见不一致：DDPO 认为它约等于早停，Flow-GRPO 报告去掉 KL 后多样性塌缩，并认为它不等于早停。对固定 σ 的高斯策略，对参考模型的 KL 可以闭式写成到 zero-shot 输出的 L2【推断，由高斯 KL 公式直接得出】。
5. **无标注 reward 都有已知的骗法。** 内点数会上升而精度不跟着上升（RIPE++ 自报）。一致性约束有平凡解，而且对常数偏置不敏感（WarpC）。跨模态相似度会塌到「编码器输出常数 + 预测恒等」（AltO）。自洽类 reward 长期优化会突然崩溃（LLM 的多数投票自奖励，SRT）。随机 reward 也可能让指标上升（Spurious Rewards）。所以随机 reward 对照应当必做。
6. **防塌缩最直接的教训来自 DISK：** 训练初期匹配质量差，带负分的 reward 会让网络「什么都不选」，停在 reward=0 的局部最优；DISK 的做法是让负分从 0 线性升到全值。本库 #66 里 RoMa 的 certainty 被压到 0，现象相同。

## 1. 先讲清楚出现的方法（各一两句）

| 方法 | 是什么 |
|---|---|
| REINFORCE（Williams 1992） | 最基本的策略梯度：采一个动作，梯度 = reward × ∇log 概率。减去一个与动作无关的 baseline 不改变期望，但能降方差。 |
| RLOO（Ahmadian et al. 2024） | 每个输入采 k 个样本，每个样本的 baseline 用其余 k−1 个的平均 reward。不需要 critic。 |
| GRPO（DeepSeekMath, Shao et al. 2024） | 每个输入采 G 个样本，优势 = (r − 组均值) / 组标准差；去掉 PPO 的价值网络；对参考模型的 KL 直接加在损失里。 |
| Dr. GRPO（Liu et al. 2025） | 指出 GRPO 除以组标准差会让「太易 / 太难」的输入权重偏大，因此去掉这一项（以及长度归一化）。 |
| PPO（Schulman et al. 2017） | 用重要性比率在同一批数据上做多步更新，并把比率裁剪在 1±ε 内，防止一步走太远。通常配 critic。 |
| DAPO（Yu et al. 2025） | GRPO 的改版：上下裁剪界分开（Clip-Higher，防熵塌缩）；丢掉组内 reward 全相同的样本（这类样本优势为 0、没有梯度）；去掉 KL。 |
| DDPO（Black et al. 2023） | 把扩散模型的每一步去噪当作一个动作，每步策略是各向同性高斯，用 REINFORCE 或 PPO 式裁剪微调；reward 按提示词归一化。 |
| DPOK（Fan et al. 2023） | 扩散模型在线 RL，加对预训练模型的 KL 正则。 |
| Flow-GRPO（Liu et al. 2025） | 流匹配模型本来是确定性 ODE，没法采样。它改写成边缘分布相同的 SDE，每步变成高斯，然后用 GRPO；训练时少步采样，推理时步数照旧。 |
| SGA-Flow-GRPO（Yang et al. 2026-09） | 在 Flow-GRPO 上把整图一个优势改成逐空间 token 的优势：用可微 reward 对图像的梯度做 credit 图，再调制每个 token 的优势。 |
| MorphSeek（2025） | 医学形变配准。在编码器顶层特征上加高斯策略头（均值 + log σ），采样潜变量送进解码器；先做无监督 warm-up，再用 GRPO（reward 是 Dice 增益）。 |
| Pinto et al., ICML 2023 | 「用任务 reward 调视觉模型」：检测、全景分割、上色都先 MLE 预训练，再用 REINFORCE 微调。输出是自回归离散 token。 |
| DISK（NeurIPS 2020） | 稀疏关键点：每个 8×8 格子里按 softmax 采一个点；匹配是双向 softmax，匹配概率可以精确算出。reward 对错匹配分别给分。 |
| DSAC / NG-RANSAC（CVPR 2017 / ICCV 2019） | 把 RANSAC 的「选假设」或「选最小样本」变成概率采样，用 score-function 梯度训练前面的网络。NG-RANSAC 可以用最终内点数当自监督 reward。 |
| RIPE / RIPE++ | 用户已熟悉，不再介绍。 |

## 2. 策略放在哪里

### 2.1 文献里的做法

| 形式 | 代表 | 采样什么 | 方差 / 开销【读到 + 推断】 |
|---|---|---|---|
| **每步高斯（整图）** | DDPO、Flow-GRPO | 每步去噪 x_{t−1} ~ N(μ_θ, σ_t²I)，σ_t 由采样器固定，或由噪声水平 a 控制 | 对数概率闭式可算【读到】。reward 是整图标量，一图百万维：Flow-GRPO 要 G=24，G=12 或 6 时训练不稳定，最后塌缩【读到】。开销是每个样本一次完整采样，Flow-GRPO 用训练时少步（10 步对 40 步）换来 4 倍以上加速【读到】。 |
| **逐空间位置分配优势** | SGA-Flow-GRPO | 同上，但每个 token 单独算对数概率比和优势 | 针对「整图一个优势」的信用分配问题【读到】；credit 图依赖可微 reward【读到】。我们的 reward 经过 RANSAC、不可微，这个做法不能直接照搬【推断】。 |
| **潜空间高斯头** | MorphSeek | 编码器顶层特征 z = μ + τσ⊙ε，解码出形变场 | 新增参数 <3%【读到】；每组 6 条轨迹，超过 8 条显存不够【读到】；log σ 要截断，否则后验塌缩【读到】。 |
| **离散 token（自回归）** | Pinto et al. | 坐标量化成 1000 桶，按 token 采样 | 每个输入只采 2 个样本（1 个算梯度、1 个当 baseline），能用的前提是 MLE 预训练提供了好的初始采样分布【读到】。 |
| **网格内 categorical + 接受概率** | DISK、RIPE | 每格采一个点 | 动作空间被网格压小；DISK 的匹配概率闭式可算，匹配这一步不引入采样方差【读到】。 |
| **几何假设采样** | DSAC、NG-RANSAC | 采最小样本集 → 假设；或按分数选假设 | NG-RANSAC 训练时 K=2–8 个假设池 × M=16 个假设，baseline 取每张图的平均损失【读到】。只训练「采样权重」，不改匹配位置本身【推断】。 |

**方差的一般规律【读到，Mohamed et al. JMLR 2020 §4.3】**：score-function 估计的方差随分布的维数增长，作者在高斯例子上画出了方差随维数 D 上升的曲线；pathwise 梯度不对「参数没有影响的项」求和，所以方差低得多。降方差的通用手段是 baseline（控制变量）和 Rao-Blackwell 化（只让一个变量对与它有关的那部分 reward 负责）。

### 2.2 放到 AnyMatch-RoMa 上【推断，结构事实取自 matcher-rl-feasibility 调研对 RoMa 代码的阅读】

RoMa 自带的随机性只有三处：粗级 4096 锚点的 softmax（锚点间距约 8.75 px）、certainty、`sample()` 里的 multinomial 选点。决定 3 px 以内精度的 refiner 是确定性回归。RoMa v2 的 refiner 已经输出逐像素 2×2 精度矩阵，用高斯 NLL 训练【读到】，说明「在 warp 上挂逐像素高斯」在这一族模型里是自然的；但 AnyMatch-RoMa 是 v1，没有这个头。

| 候选策略 | 动作 | 推理开销 | 方差 | 备注 |
|---|---|---|---|---|
| **S1 采样点上的高斯位移** | RoMa 原本就采约 1 万个匹配点；在每个点的细级 warp 上加 ε ~ N(0, σ²I)，σ 固定（如 0.5–2 px） | 不变（推理取均值） | 若 reward 是整对标量，方差约随点数增长；若能按点给 reward，就降到逐点量级 | 不加新参数。σ 固定时，对 zero-shot 的 KL 就等于到 zero-shot warp 的 L2（除以 2σ²） |
| **S2 粗级锚点 categorical** | 每格从 4096 类中采一个锚点 | 不变 | 中 | 网络原生分布，但分辨率粗于瓶颈；#66 的 M7（只训锚点项、闭式期望）与 zero-shot 持平【线索】 |
| **S3 选点（certainty 上 multinomial）** | 采哪些像素作匹配 | 不变 | 中 | RIPE 式；#66 显示 certainty 容易被压到 0【线索】 |
| **S4 假设采样（DSAC / NG-RANSAC 式）** | 按点权重抽 3 点最小集 → 仿射假设 | 不变（推理仍跑 RANSAC） | 低（每对只有几十个假设） | 只能训练点的权重，不能纠正点的位置【推断】 |
| **S5 潜空间高斯（MorphSeek 式）** | 在解码器输入特征上采样 | 不变（取均值） | 高（特征维度大，一个采样同时影响整张 warp） | 要加头和 warm-up；可训模块和现有自训练不同，可能越出「只换监督信号」的约束 |

## 3. 算法

**【读到】**
- **不需要 critic，组内基线就够。** GRPO 去掉价值网络，用组均值作 baseline；RLOO 用留一均值，并论证从强预训练模型出发时，PPO 为降方差引入偏差「不值得」，而且裁剪平均只在 <5% 的样本上触发，接近 on-policy。Pinto et al. 在视觉任务上只用 1 个额外样本当 baseline，不加任何正则。
- **单次更新还是 PPO 式多步更新。** DDPO_SF（纯 REINFORCE，每批数据只更新一次）和 DDPO_IS（PPO 式裁剪，clip 只有 1e-4）效果接近，后者略好。
- **标准差归一化有偏差。** Dr. GRPO：除以组标准差会放大「组内 reward 几乎一样」的输入。DAPO：组内 reward 全相同时优势为 0、没有梯度，干脆丢掉。
- **KL 到参考模型。** GRPO 把 KL 直接加进损失（无偏、非负的估计）。DPOK 加 KL，报告优于监督微调。DDPO 不加，认为 KL「经验上可能等价于早停」，改用手动挑退化前的检查点。Flow-GRPO 的 KL 系数 β 取 0.01–0.04，报告去掉 KL 后 PickScore 训练的输出「收敛到同一风格」，并称 KL「经验上不等价于早停」。DAPO 在长链推理里去掉 KL，理由是策略本来就该远离初始模型。
- **组大小。** Flow-GRPO 在 G=24 稳定，G=12 和 G=6 都塌了；MorphSeek 每组 6 条轨迹。

**对本问题【推断】**
- RoMa 一次前向后可以零成本地多次采样（S1 到 S4 都只改输出端），所以组大小主要受 RANSAC 次数限制，不受网络前向限制。G 取 8–16 是合理的起点。
- 先用 RLOO / Dr. GRPO 式优势（减组均值、不除标准差），每批更新一次，不上 PPO。只有在想复用样本、做多步更新时才需要裁剪。
- KL：S1 下闭式等于对 zero-shot warp 的 L2，几乎没有额外成本，应作为可调项保留；系数靠前几次试错定。LoRA 本身也限制了偏离程度。

## 4. 无标注 reward 候选与已知骗法

旧调研 `label-free-reward/` 的四维评估仍可作为线索；这里只保留重新核对过的事实。

| reward | 文献用法【读到】 | 已知骗法【读到】 | 对本问题【推断】 |
|---|---|---|---|
| **RANSAC 内点数 / 分级残差** | NG-RANSAC：最终内点数「是很稳定的训练信号」，自监督略差于有监督。RIPE++：内点 +1.0、外点 −0.1、未匹配点 −1e-7；附录的分级 reward 按 Sampson 距离给连续分数 | RIPE++：内点数明显上升，但位姿精度不升；网络可以把所有点预测成不可匹配来躲开惩罚。NG-DSAC++：学会忽略天空、草地这些预测不准的区域（有利的一面） | 仿射版直接可用，按点打分方差低。看不见整对一致的偏移；匹配会往纹理区集中 |
| **warp 后跨模态相似度** | MU-Net 用 CFOG 特征上的 NCC 直接当可微损失 | AltO：编码器输出常数、配准网络输出恒等，靠交替优化防住。SSHNet：OPT-SAR 上早期无监督方法不收敛或停在恒等水平（旧调研，本次未重核） | 只有接在 RANSAC 仿射后面才是不可微 reward；我们的数据已粗对齐，恒等附近本来就是高相似度的局部极值，骗法风险更大 |
| **已知合成扰动的一致性**（对 SAR 施加 T，要求 A₂ = T∘A₁） | WarpC 推导并分析了各种 bipath 约束 | WarpC：J→I 型约束对预测里的常数偏置不敏感，W→0 时「没有学习信号」；I′J 型约束可以把所有点映射到同一个常数点来满足；forward-backward 约束的平凡解是恒预测零流；只采很小的 warp 会把预测拉向零 | 正好对应「已知合成扰动的恢复程度」这一候选。平移偏差与纯平移的 T 可交换，所以对平移偏差没有信号；只适合当辅助项 |
| **正反向 / 循环一致性** | 同上（WarpC） | 平凡解：零流或常数映射（WarpC） | 同上，单独用风险高 |
| **自洽多数投票类**（LLM 无标注 RL） | SRT（Shafayat et al. 2025）用多数投票当 reward | 长时间训练后模型学会最大化伪 reward，出现「突然、完全的性能崩溃」 | 「多个采样估的仿射彼此一致」就是这一类，应预期同样的晚期崩溃 |
| **随机 reward（对照）** | Spurious Rewards（2025） | 在 Qwen2.5-Math 上随机 reward 让 MATH-500 提升 21.4 点（真实 reward 是 29.1）；作者归因于 GRPO 裁剪项放大了模型原有的偏好；在其他模型族上不成立 | 随机 reward 对照应必做；用了裁剪时尤其要做 |

## 5. 防塌缩：RIPE / RIPE++ 之外的做法

| 做法 | 出处【读到】 | 对本问题【推断】 |
|---|---|---|
| **负分退火**：错匹配和每点惩罚的权重在前 5 个 epoch 从 0 线性升到全值；否则网络停止采点，停在 reward=0 的局部最优 | DISK | 与 #66「外点负分让 certainty 全 0 成为吸收态」机制相同。先用非负 reward，或退火负分 |
| **每采一个点给小惩罚**，防止大量不可匹配的点 | DISK（λ_kp=−0.001）、RIPE++（λ=−1e-7） | 与上一条方向相反，权重要很小 |
| **网格 / 分层采样**，强制空间均匀 | DISK、RIPE | 防止匹配往少数纹理区集中；RoMa 的 KDE 平衡采样已有类似作用 |
| **KL 到参考模型** | GRPO、DPOK、Flow-GRPO（去掉 KL 后多样性塌缩） | S1 下就是到 zero-shot 的 L2 |
| **早停 / 选退化前的检查点** | DDPO | 现有判据本来就看 Val 后段均值，需要配合保留全部检查点 |
| **保住探索（熵）**：熵会单调下降，性能上限随之锁定；对高协方差 token 限制更新（Clip-Cov / KL-Cov）或放宽上裁剪界（Clip-Higher） | Cui et al. 2025、DAPO | 固定 σ 的高斯策略熵不变，不存在这个问题；只有学 σ 或用 categorical（S2/S3）时才要管 |
| **warm-up + 截断 log σ + 对先验的 KL** | MorphSeek（KL 权重为 0 时 Dice 掉到 49.63；warm-up 让稳定训练的成功率从 33% 升到 79%） | 学 σ 时照做；固定 σ 就不需要 |
| **交替优化**，不同时训练会合谋的两部分 | AltO | 对应「不训 certainty」或「certainty 与 warp 分开训」 |
| **熵正则要小心调** | RIPE++：权重 1e-5 时性能下降，1e-4 时训练崩溃 | 不建议在没有对照的情况下加 |

## 6. 与旧调研的出入

- `matcher-rl-feasibility` 写 RIPE++「内点 +1、外点 −1」。原文表 5 是 ρ_in=1.0、ρ_out=−0.1。
- `label-free-reward` 写 WarpC「forward-backward 权重调高会把网络推向零」。本次读到的原文表述是：forward-backward 约束的平凡解是恒预测零流；只采很小的 warp 会把预测偏向零。消融里没有单独给 forward-backward 的权重。
- `rl-registration-survey` 列出的 MorphSeek 是「弱监督」：reward 是 Dice 增益加负雅可比惩罚，需要分割标签，不是无标注 reward。它对我们有用的是策略形式和防塌缩做法，不是 reward。

## 7. 对本问题的候选清单【全部为推断】

约束：AnyMatch-RoMa；光学–SAR；DINOv2 qkv LoRA 等训练设置不变，只换监督信号；推理开销不增；必须满足采样、不可微 reward、策略梯度三点。

### 策略形式

| 候选 | 成本 | 风险 |
|---|---|---|
| **S1 采样点上的固定 σ 高斯位移**（首选） | 低：不加参数，一次前向采 G 组扰动 | 若只用整对 reward，方差大；若按点打分，期望梯度会接近「回归到 RANSAC 仿射」，与伪标签很像，论文上要讲清区别 |
| S4 假设采样（NG-RANSAC 式） | 低 | 只能训练点的权重，碰不到 3 px 内的位置误差；可以作为 S1 的补充 |
| S2 粗级锚点 categorical | 低 | 分辨率不够；#66 中同一作用点的闭式期望版本与 zero-shot 持平 |
| S3 certainty 选点 | 低 | certainty 塌到 0 的吸收态（#66） |
| S5 潜空间高斯 | 中：加头、warm-up | 方差高；可能越出「只换监督信号」 |

### 算法

| 候选 | 成本 | 风险 |
|---|---|---|
| **组内留一基线（RLOO / Dr. GRPO，不除标准差），每批更新一次**（首选） | 低：G 次 RANSAC | G 太小会不稳（Flow-GRPO 的 G=6、12 都塌了） |
| GRPO（除标准差） | 低 | 放大组内 reward 几乎一样的对（Dr. GRPO） |
| PPO 裁剪多步更新 | 中 | 裁剪偏差会放大原有偏好（Spurious Rewards）；收益不确定 |
| KL 到 zero-shot（S1 下等于 L2） | 几乎为 0 | 系数过大等于不训；DDPO 与 Flow-GRPO 对其作用结论相反 |

### reward（先做 CONTEXT 里定义的「reward 有效性检验」再选）

| 候选 | 成本 | 风险 |
|---|---|---|
| **R1 分级残差**：点到组内其他样本估的仿射（留一或子集）的距离，按 RIPE++ 截断二次核打分 | 中：每个样本 1 次仿射 RANSAC | 看不见整对一致的偏移；内点涨精度不涨 |
| **R2 RANSAC 仿射 warp 后的 CFOG-NCC**（整对标量） | 中：RANSAC + warp + FFT-NCC | 恒等附近的局部极值；重叠区落在均匀区时虚高；月面上未验证 |
| R3 已知合成扰动一致性（A₂ vs T∘A₁） | 高：2 次前向 + 2G 次 RANSAC | 对平移偏差没有信号；只能当辅助项 |
| R4 组内多个样本的仿射彼此一致 | 低 | 自洽类 reward 的晚期突然崩溃（SRT） |
| **随机 reward 对照** | 同主实验 | 必做；不做就无法区分「reward 有信号」和「优化动力学」（#66 与 Spurious Rewards 都有先例） |

### 防塌缩

| 候选 | 成本 | 风险 |
|---|---|---|
| **不训 certainty（或 certainty 只用原配方的自监督项）** | 0 | 位置之外的信号少了一块 |
| **非负 reward 或负分退火**（DISK） | 0 | 退火长度要调 |
| **KL/L2 到 zero-shot** | ≈0 | 系数要调 |
| 保留全部检查点 + 后段均值判据 | 存储 | 已有流程 |
| 学 σ 时：截断 log σ + warm-up（MorphSeek） | 低 | 只在 σ 可学时需要 |

## 8. 未能核实

- 没找到对 RoMa、DKM、LoFTR 做 REINFORCE / PPO / GRPO 微调的论文；检索可能有遗漏。
- DPOK 的 KL 分析细节只读到摘要。
- SRT 提出的缓解措施只看到二手摘要，正文未核实，因此表中没有引用。
- SSHNet 的数字沿用旧调研，本次未重核。
- S1 下「按点打分的期望梯度接近回归到 RANSAC 仿射」是推断，没有文献直接讨论。
- 所有成本数字都没有在本库服务器上实测。
