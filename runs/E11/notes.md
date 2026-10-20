# 笔记：SuperPoint+SuperGlue、LoFTR、DKM→RoMa 的网络结构与训练方式

本节点是回查用的笔记，不是实验报告，不按 Notes 的论文格式写。它覆盖三条方法线的思想、网络结构、输出、训练监督和推理流程，并单列一章对比各方法（含 AnyMatch）的训练数据。不涉及本仓库怎么调用和微调这些方法，也不涉及各方法在 benchmark 上的效果。

**怎么读。** 先看第 0 章的演进关系和对比表；需要细节再翻到对应章节。张量形状按本项目的输入写：patch 原图 512²，SP+SG 放大到长边 1600，LoFTR 输入长边 640，RoMa 粗匹配 560²、上采样细化 864²。官方默认值另行注明。

**出处。** 细节以官方代码为准，记作 `仓库@commit:文件:行`。论文和代码不一致的地方标 **⚠论文≠代码**。所用版本：

| 简称 | 仓库 | commit |
|---|---|---|
| SG | magicleap/SuperGluePretrainedNetwork（含 SuperPoint 推理代码） | ddcf11f |
| LoFTR | zju3dv/LoFTR | df7ca80 |
| DKM | Parskatt/DKM | ef57565 |
| RoMa | Parskatt/RoMa | 77f8d68 |
| AnyMatch | MnYangs/AnyMatch | 259ad34 |

论文：SuperPoint arXiv 1712.07629；SuperGlue arXiv 1911.11763；LoFTR arXiv 2104.00680；DKM arXiv 2202.00667；RoMa arXiv 2305.15404；AnyMatch arXiv 2606.31077。SuperPoint 和 SuperGlue **没有公开训练代码**，所以这两者的训练部分只能以论文为准。

---

## 0. 总览

### 0.1 演进关系

```
稀疏（先检测后匹配）         半稠密（无检测器）             稠密（每个像素一个 warp）
SuperPoint ──► SuperGlue ──► LoFTR ─────────────────────► DKM ──────────► RoMa
 检测+描述     注意力+OT匹配   粗网格Transformer+细化         GP全局匹配+逐级细化  冻结DINOv2+锚点分类
```

- **SuperPoint**：用一个 CNN 同时输出关键点和描述子。没有人工标注的关键点，它的做法是先在合成几何图形上训一个检测器，再用「单应适应」（Homographic Adaptation）给真实图像自动生成伪标签。
- **SuperGlue**：针对「描述子最近邻匹配不看上下文」的问题，把两张图的关键点看成一张图，用 self/cross 注意力交换信息，再用带 dustbin 的最优传输（Sinkhorn）求部分分配。不能匹配的点会被分给 dustbin。
- **LoFTR**：针对「检测器在弱纹理、重复纹理区域找不到可重复的点」的问题，去掉检测器。它让 1/8 分辨率网格上的每个格子都参与匹配：先用 Transformer 融合两图特征，再用 dual-softmax 选出粗匹配，最后在 1/2 分辨率的窗口里做亚像素细化。结果是半稠密匹配。
- **DKM**：走到全稠密，直接为图 A 的每个像素回归它在图 B 中的坐标（warp），同时输出一个 certainty。全局匹配用核回归（高斯过程，GP）实现，再由逐级的卷积细化器从 1/16 修正到原分辨率。最后按 certainty 采样出点对。
- **RoMa**：沿用 DKM 的框架，目标是提高鲁棒性。粗特征换成冻结的 DINOv2，细特征换成 VGG19；粗匹配解码器改成 Transformer，对锚点做分类；损失也改了：粗级用「以分类做回归」，细级用鲁棒回归。

### 0.2 横向对比

每行一个方法，按主题分成三张表。同样的内容另存为 `extra/data/compare.csv`。

**结构**

| 方法 | 输入与主干 | 匹配机制 | 输出 |
|---|---|---|---|
| SP+SG | 灰度；VGG 式 8 层卷积，1/8 | 9 对 self/cross 注意力 → Sinkhorn 最优传输（带 dustbin） | 稀疏点对 + 分数 |
| LoFTR | 灰度；ResNet-18 式 + FPN，1/8 与 1/2 | 4 对 self/cross 线性注意力 → dual-softmax + 互为最近邻 → 5×5 窗口期望坐标 | 半稠密点对 + 置信度 |
| DKM | RGB；ResNet-50，1/1 到 1/32 | 1/32、1/16 上 GP 核回归 + 卷积解码 → 逐级卷积细化 | 双向稠密 warp + certainty，再采样 |
| RoMa | RGB；粗级 DINOv2（冻结），细级 VGG19-BN | 1/14 上 GP → Transformer 对 64×64 锚点分类 → 逐级卷积细化 | 同 DKM |

**监督**

| 方法 | 监督信号与 GT 来源 | 损失 |
|---|---|---|
| SP+SG | SP：合成形状 + 单应适应伪标签；SG：GT 点对和不可匹配集合，来自单应或深度+位姿 | SP：65 类交叉熵 + 描述子 hinge；SG：分配矩阵负对数似然 |
| LoFTR | 深度+位姿重投影：粗级 GT 匹配矩阵，细级亚像素偏移 | 粗级：focal 加权负对数似然（只算正样本）；细级：按不确定度加权的 L2 |
| DKM | 深度+位姿得到每个像素的 warp；深度一致性掩码（< 0.05） | 每级 EPE + 0.01×BCE |
| RoMa | 同 DKM | 粗级：锚点交叉熵 + 0.01×BCE；细级：鲁棒回归 + 0.01×BCE |

**分辨率**

| 方法 | 官方训练数据与分辨率 | 本项目推理分辨率 |
|---|---|---|
| SP+SG | SP：COCO 240×320；SG：MegaDepth | 长边 1600 |
| LoFTR | MegaDepth，长边 840（论文）或 640 | 长边 640 |
| DKM | MegaDepth，540×720 | —（未使用） |
| RoMa | MegaDepth，560×560 | 粗匹配 560²，细化 864² |

---

## 1. SuperPoint + SuperGlue

### 1.1 SuperPoint：检测器加描述子

**思想。** 一个共享编码器接两个头：检测头输出每个像素是关键点的概率，描述头输出稠密描述子。监督分两步获得：先在合成几何图形上训出检测器 MagicPoint，再用单应适应在真实图像上生成伪标签。

**结构（本项目输入 1600²）。**

```
img 1×1600×1600 (灰度, [0,1])
 └ 编码器: conv3×3 ×8 (64,64 | 64,64 | 128,128 | 128,128)，每两层后 maxpool，共 3 次
   → x 128×200×200 (1/8)
     ├ 检测头: conv3×3→256, conv1×1→65  → 65×200×200
     │    softmax(65 类，含 1 个「无关键点」dustbin) → 去掉 dustbin → 64×200×200
     │    pixel-shuffle(8×8) → 1600×1600 热图 → NMS → 阈值 → top-k → 关键点 N×2
     └ 描述头: conv3×3→256, conv1×1→256 → 256×200×200, L2 归一化
          在关键点处双线性插值 → 再次 L2 归一化 → 256×N
```

出处：`SG@ddcf11f:models/superpoint.py:117-134`（层定义），`:160-167`（65 通道 softmax 加 pixel-shuffle），`:189-196`（描述子与采样），`:80`（`sample_descriptors`，s=8）。

- 检测头相当于把每个 8×8 格子做一次 65 类分类：64 个像素位置，加一个「这个格子里没有关键点」。
- 推理默认值：`nms_radius=4`，`keypoint_threshold=0.005`，`remove_borders=4`（`superpoint.py:103-109`）。本项目的 spsg 配置改为 `nms_radius=3`、`max_keypoints=2048`。
- ⚠论文≠代码：论文说每层卷积后接 BatchNorm，描述子用双三次插值（论文 §3.3、§6）；发布的推理代码里没有 BN 层，采样用的是双线性 `grid_sample`。

**训练（论文 §3.4、§4、§5、§6；无官方训练代码）。**

1. **MagicPoint**：只用检测头，在 Synthetic Shapes 上训练。这个数据集是实时渲染的三角形、四边形、线段、立方体、棋盘格、星形，角点标签已知，训练 200k 次迭代。
2. **单应适应**：对 COCO 2014 训练集的 8 万张图（灰度，240×320）各做 $N_h=100$ 次随机单应，在每个变换结果上跑 MagicPoint，把输出变换回原图后取平均，平均响应就是伪标签：
   $$\hat F(I)=\frac1{N_h}\sum_{i=1}^{N_h}H_i^{-1}f(H_i(I))$$
   这个过程迭代两轮。
3. **联合训练 SuperPoint**：对每张图随机采样一个单应 $H$，得到图像对、两张图各自的伪标签，以及格子之间的对应关系 $s$：
   $$\mathcal L=\mathcal L_p(X,Y)+\mathcal L_p(X',Y')+\lambda\,\mathcal L_d(D,D',S),\quad \lambda=10^{-4}$$
   - $\mathcal L_p$：在每个格子上做 65 类交叉熵。
   - $\mathcal L_d$：对所有格子对计算描述子 hinge 损失。两个格子的中心经 $H$ 变换后距离 ≤ 8 px 时 $s=1$。
     $$l_d=\lambda_d\, s\max(0,m_p-d^\top d')+(1-s)\max(0,d^\top d'-m_n)$$
     其中 $m_p=1$，$m_n=0.2$，$\lambda_d=250$。
   - 优化器 Adam，lr 0.001，batch 32；增强包括高斯噪声、运动模糊和亮度变化。

### 1.2 SuperGlue：注意力图网络加最优传输

**思想。** 匹配被看成两组关键点之间的部分分配问题。用关键点的位置和描述子初始化节点特征；self 注意力让每个点看本图的上下文，cross 注意力让它看另一张图；最后由可微的 Sinkhorn 算法求解带 dustbin 的最优传输，不能匹配的点分配给 dustbin。

**结构（N、M 为两图关键点数，本项目各 ≤ 2048）。**

```
关键点 p: N×2, 分数 c: N, 描述子 d: 256×N
 ├ 位置归一化: (p - 图像中心) / (0.7·max(H,W))
 ├ 关键点编码器 MLP [3→32→64→128→256→256]((x,y,c))，加到 d 上 → x⁽⁰⁾ 256×N
 ├ 注意力 GNN: 18 层 = 9×(self, cross)，每层 4 头，残差更新 x ← x + MLP([x ‖ message])
 ├ final_proj (1×1 conv) → 匹配描述子 f
 ├ 分数矩阵 S_ij = <f_i, f_j>/√256 → N×M
 ├ 加 dustbin 行和列（共享的可学习标量 α） → (N+1)×(M+1)
 └ 对数域 Sinkhorn T 次迭代 → 分配矩阵 P̄
```

出处：`SG@ddcf11f:models/superglue.py:65-72`（位置归一化），`:75-84`（关键点编码器），`:124-140`（GNN），`:143-172`（对数域 Sinkhorn 与 dustbin），`:197-204`（默认配置），`:256-278`（打分与取匹配）。

- **取匹配**：在 $\bar P$ 中按行、列分别取 argmax，要求互为最近邻，且 $\exp(\bar P_{ij})>0.2$（`match_threshold`）。
- Sinkhorn 次数默认 100（论文 §4，$T=100$），本项目的 spsg 配置用 20。
- 有 indoor 和 outdoor 两套权重，本项目用 outdoor。
- ⚠论文≠代码：论文附录 E 说位置「按图像最大边归一化」，代码实际是先减去中心，再除以 0.7×最大边。

**训练（论文 §3.3、附录 E；无官方训练代码）。**

- **损失**：分配矩阵的负对数似然，正样本和不可匹配点都监督：
  $$\mathcal L=-\sum_{(i,j)\in\mathcal M}\log\bar P_{ij}-\sum_{i\in\mathcal I}\log\bar P_{i,M+1}-\sum_{j\in\mathcal J}\log\bar P_{N+1,j}$$
- **GT 构造**：用 GT 单应，或者深度加位姿，算出所有关键点之间的 N×M 重投影误差矩阵。
  - $\mathcal M$：误差同时是所在行和所在列的最小值，且小于阈值（单应 3 px，indoor 5 px，outdoor 3 px）。
  - $\mathcal I$、$\mathcal J$：单应数据里，没有进入 $\mathcal M$ 的点都算不可匹配；indoor 和 outdoor 数据还要求最小重投影误差分别大于 15 px 和 5 px。处在两个阈值之间的点有歧义，不加监督。
- **SuperPoint 冻结**：只训 SuperGlue。每个 batch 实时跑 SuperPoint 提取特征，关键点数固定；不够时补随机点，这样便于组 batch，也能增加鲁棒性。
- **超参（单应 / indoor / outdoor）**：Adam，lr 1e-4，分别在前 200k / 100k / 50k 步保持不变，之后指数衰减到 900k 步；batch 32 / 64 / 16 对；每张图 512 / 400 / 1024 个点。outdoor 模型从单应模型初始化。

---

## 2. LoFTR

**思想。** 不用检测器。在 1/8 分辨率的网格上，把每个格子都当成候选点；用 Transformer（self/cross 交替）让特征同时带上全局上下文和另一张图的信息，于是弱纹理区域也能区分开。粗匹配用 dual-softmax 求得，然后在 1/2 分辨率的局部窗口里，用一个小 Transformer 加期望坐标细化到亚像素。

**结构（本项目输入 640²，官方 outdoor 推理用 840 或 1200）。**

```
img0, img1: 1×640×640 (灰度, [0,1])，两张图拼成一个 batch 共用 BN
 └ ResNet-FPN_8_2: conv7×7/2 → 3 段 BasicBlock (128, 196, 256) → FPN 逐级上采样融合
   ├ 粗特征 F_c: 256×80×80 (1/8)
   └ 细特征 F_f: 128×320×320 (1/2)
粗级:
 F_c + 正弦位置编码 → 展平成 6400×256
 └ LoFTR 粗模块: 4×(self, cross)，线性注意力，8 头 → F̃_c0, F̃_c1: 6400×256
 └ S = <F̃_c0, F̃_c1>/256 / τ (τ=0.1) → 6400×6400
   P_c = softmax_行(S) ⊙ softmax_列(S)              (dual-softmax)
   粗匹配 = {P_c > θ_c (0.2)} ∩ 互为最近邻 ∩ 去掉离边 2 格以内的格子
细级 (对每个粗匹配 (i,j)):
 └ 从 F_f 中，以 i、j 对应位置为中心取 5×5 窗口 → 25×128（两图各一份）
   粗特征降维到 128 维后与之拼接，经线性层融合
 └ LoFTR 细模块: 1×(self, cross) → 25×128
 └ 图 0 窗口中心的特征与图 1 的 25 个特征做点积 → softmax → 5×5 热图
   期望坐标 (∈[-1,1]²) × 2(窗口半径) × 2(1/2 分辨率) → 在 640 输入上的偏移 ≤ ±4 px
输出: mkpts0 (粗格子中心)，mkpts1 (粗位置加细偏移)，mconf
```

出处：`LoFTR@df7ca80:src/loftr/loftr.py:39-75`（整体流程），`src/loftr/backbone/resnet_fpn.py:43-118`（ResNetFPN_8_2），`src/config/default.py:4-63`（默认结构超参），`src/loftr/utils/coarse_matching.py:106-119`（dual-softmax），`:174-196`（阈值、去边、互为最近邻），`src/loftr/loftr_module/fine_preprocess.py:39-57`（取窗口并拼接粗特征），`src/loftr/utils/fine_matching.py:43-69`（期望坐标、方差、最终坐标）。

- 只有图 0 一侧取的是粗格子中心，没有细化；细化只移动图 1 一侧的点。
- `outdoor_ds.ckpt` 对应配置 `configs/loftr/outdoor/buggy_pos_enc/loftr_ds.py`，其中 `TEMP_BUG_FIX=False`（旧版位置编码的写法）。默认配置 `default.py:23` 里这个值是 True，加载官方权重时必须改成 False。
- 粗模块还有一种 `sinkhorn` 匹配方式（带 dustbin 的最优传输，3 次迭代），outdoor_ds 用的是 dual-softmax。

**监督与损失。**

- **GT 生成**（`src/loftr/utils/supervision.py:47-109`）：把图 0 每个粗格子的中心，用深度、位姿和内参投影到图 1，取整得到所在的粗格子；再从图 1 反向投影回来，检查能否回到原格子（互为最近邻），能回去就把 `conf_matrix_gt[i,j]` 置 1。代码注释说明不做深度一致性检查，因为实验中效果更差（`:60`）。
- **细级 GT**（`:123-143`）：把图 0 粗格子中心的精确投影位置，减去当前预测的图 1 格子中心，再除以（2×窗口半径 2），得到落在 $[-1,1]$ 内的目标偏移。
- **粗级损失**（`src/losses/loftr_loss.py:52-85`）：论文写的是在 GT 匹配上取负对数似然（§3.5）：
  $$\mathcal L_c=-\frac1{|\mathcal M_c^{gt}|}\sum_{(i,j)\in\mathcal M_c^{gt}}\log P_c(i,j)$$
  ⚠论文≠代码：outdoor_ds 的实际配置是 `COARSE_TYPE='focal'`、`SPARSE_SPVS=True`，即只对 GT 正样本计算带 focal 权重的损失 $-\alpha(1-P)^\gamma\log P$（$\alpha=0.25$，$\gamma=2$），负样本不加损失。dual-softmax 本身的归一化已经会压低错误位置的概率。
- **细级损失**（`:122-150`）：按不确定度加权的 L2。权重用热图的标准差 $\sigma$ 计算，取 $1/\sigma$ 后按均值归一化，并且 detach，防止网络靠增大 $\sigma$ 来降低损失。GT 落在窗口外（$\|\cdot\|_\infty\ge1$）的样本不计入：
  $$\mathcal L_f=\frac1{|\mathcal M_f|}\sum\frac1{\sigma^2(\hat i)}\|\hat j-\hat j_{gt}\|_2$$
  （论文 §3.5 的形式；代码中权重是 $1/\sigma$ 的归一化值，距离取平方。）总损失为 $\mathcal L=\mathcal L_c+\mathcal L_f$，两者权重都是 1。
- **训练技巧**（`coarse_matching.py:198-236`）：训练时，送进细级的粗匹配数量取「候选总数×30%」。名额先给预测出的匹配，至少留 200 个名额用随机抽取的 GT 匹配补齐，保证细级始终有正确的样本可学。

**训练设置（outdoor）。** 从随机初始化开始端到端训练（论文 §3.6）。MegaDepth，图像长边 resize 到 840（论文 §4.2；仓库另有 640 版配置），右下补零成正方形，深度图补到 2000²。场景均衡采样：每个子场景每个 epoch 取 100 对。AdamW，weight decay 0.1；学习率 `CANONICAL_LR=8e-3`，对应 batch 64，按实际 batch 线性缩放（`train.py:73-76`）；前 1875 步 warmup，在第 8、12、16、20、24 个 epoch 降学习率，共训 30 个 epoch。⚠论文≠代码：论文写的是 ScanNet 上用 Adam、lr 1e-3、batch 64，与 outdoor 配置不同。

---

## 3. DKM → RoMa

### 3.1 DKM

**思想。** 稠密匹配：为图 A 的每个像素估计它在图 B 中的坐标，即 warp $W^{A\to B}$（归一化坐标，范围 $[-1,1]$），同时估计这个像素是否可匹配，即 certainty。分两步：
1. **全局匹配器**：在粗尺度上，把「A 的像素对应 B 的哪里」写成核回归。以特征相似度作为核，回归 B 的坐标嵌入（高斯过程的后验均值），再由卷积解码器把结果解码成 warp 和 certainty。
2. **warp 细化器**：从 1/16 到 1/1 逐级修正。每级把 B 的特征按当前 warp 采样到 A 上，与 A 的特征、局部相关等拼接起来，预测 warp 和 certainty 的增量。

最后按 certainty 采样出点对。

**结构（官方训练分辨率 540×720；为方便和 RoMa 对比，这里按 560² 写）。**

```
img A,B: 3×560×560 → ResNet-50 (ImageNet 初始化，参与训练)
  金字塔: 1: 3×560² (原图) | 2: 64×280² | 4: 256×140² | 8: 512×70² | 16: 1024×35² | 32: 2048×18²
全局匹配器 (分别在 32 和 16 两个尺度上，1/16 时以 1/32 的结果为上下文):
  f_A, f_B 经 1×1 conv 降到 512 维
  GP: 核 K(a,b)=exp((cos(a,b)-1)/T)，T=0.2
      B 的坐标嵌入 φ_B = cos(8π·conv(网格坐标))，256 维
      μ_A = K_AB (K_BB + σ²I)⁻¹ φ_B   (σ²=0.1)        → 256×h×w
  DFN 解码器 (卷积，含 RRB/CAB 模块): [μ_A ‖ f_A] → 3 通道 = warp(2) + certainty(1)
warp 细化器 (在尺度 16, 8, 4, 2, 1 上):
  输入 = [ f_A, 按当前 warp 采样的 f_B, 位移嵌入, B 上 warp 落点附近的局部相关 (半径 7/3/2，只在 16/8/4 上用) ]
  8 个 depthwise 5×5 卷积块 → Δwarp(2) + Δcertainty(1)
  warp ← warp + s·Δ/(4·w)；certainty 累加；每级之间 detach 梯度
输出: 原分辨率的双向 warp (H×W×2) 加 certainty logit
```

出处：`DKM@ef57565:dkm/models/model_zoo/DKMv3.py:8-150`（组装），`dkm/models/dkm.py:128-147`（CosKernel），`:342-373`（GP 前向），`:247-257`（DFN），`:77-127`（ConvRefiner），`:456-537`（Decoder 逐级流程），`dkm/models/encoders.py:30-49`（ResNet-50 金字塔）。

- **推理**：两个方向都算（`symmetric`）。outdoor 默认先在训练分辨率上算一遍，再在 `upsample_res=(864,1152)` 上把细化器重跑一遍（`dkm.py:569`）。
- **采样**（论文 §3.4）：certainty 高于 0.05 的一律当作 1；先按 certainty 多采一批候选，再在 4 维匹配空间里做核密度估计（KDE），按密度的倒数重新采样，得到分布均衡的点对。论文的评测取 5000 对。

**损失**（`dkm/losses/depth_match_regression_loss.py:61-128`；论文 §3.5）。在每个尺度 $l$ 上：
$$\mathcal L^l=\sum_{\text{grid}}p^l\,\|W^l-\hat W^l\|_2\;+\;0.01\cdot\mathrm{BCE}(\hat p^l,p^l)$$

- $W^l$ 是 GT warp：把图 A 的网格用深度和位姿投影到图 B 得到。$p^l$ 是可匹配掩码：投影后的深度与 B 的深度图一致（相对误差 < 0.05，`dkm/utils/utils.py:286-288`）。⚠该函数的 docstring 写的是 0.2，实际代码是 0.05。
- 细尺度（$l\le8$）上，如果上一级的 EPE 已经超过 $(2/512)\cdot4\cdot l$（归一化坐标），就把 $p$ 置 0。这样粗级的错误不会拿来惩罚细级，细级只学局部修正。
- 粗级 warp 用的是普通的 L2（EPE）回归。RoMa 正是从这里改起。

**训练**（论文 §4.1，`experiments/dkm/train_DKMv3_outdoor.py:177-205`）。MegaDepth，540×720，batch 32；AdamW，weight decay 0.01；学习率解码器 4e-4、主干 2e-5（代码按每卡 8 对写成 `gpus×1e-4` 和 `gpus×5e-6`，4 卡时与论文一致）。共训 250k 步，在第 166,666 和 225,000 步各乘 0.2。

### 3.2 RoMa：相对 DKM 改了什么

**思想。** DKM 的粗匹配依赖在 MegaDepth 上训练的 ResNet 特征，遇到训练集里没见过的外观变化就容易失败。RoMa 改了三处：(1) 粗特征换成冻结的 DINOv2，借用大规模自监督预训练带来的鲁棒性，冻结也减少了过拟合；(2) 粗匹配解码器换成不带坐标编码的 Transformer，不再直接回归坐标，而是对 B 上的锚点做分类，这样能表达多峰的匹配分布；(3) 损失按尺度分开设计：粗级用分类做回归，细级用鲁棒回归。细化器的结构基本沿用 DKM。

**结构差异（粗匹配 560²，上采样细化 864²）。**

```
粗特征: DINOv2 ViT-L/14 (冻结，no_grad) → 1024×40×40 (patch 14；代码里这一级仍记作 "16")
        → 1×1 conv + BN → 512
细特征: VGG19-BN (ImageNet 初始化，参与训练)，取每个 maxpool 之前的输出
        1: 64×560² | 2: 128×280² | 4: 256×140² | 8: 512×70²
        → 1×1 conv + BN 分别降到 9 / 64 / 256 / 512 维
全局匹配器 (只在 "16" 这一个尺度上):
  GP (同 DKM，gp_dim=512) → μ_A 512×40×40
  Transformer 匹配解码器: token = [μ_A ‖ f_A] 1024 维 × 1600 个；5 个 block，8 头，无位置编码
  → 4097 通道 = B 上 64×64 锚点网格的 logits (4096) + certainty (1)
  warp = 取概率最大的锚点及其上下左右 4 个邻居，按概率加权平均锚点坐标
细化器: 尺度 16/8/4/2/1，结构同 DKM (局部相关半径 7/3/2)
上采样: 把输入 resize 到 864²，只重算 VGG (不再跑 DINOv2)，
        从 560 的 warp 出发，在尺度 8/4/2/1 上重跑细化器
```

出处：`RoMa@77f8d68:romatch/models/model_zoo/roma_models.py:226-395`（`roma_model` 组装：解码器 5 个 block、64² 锚点、细化器、投影层），`romatch/models/encoders.py:6-27`（VGG19 金字塔），`:29-68`（CNNandDinov2，DINOv2 不参与训练），`romatch/models/transformer/__init__.py:30-46`（解码器前向），`romatch/utils/utils.py:301-322`（`cls_to_flow_refine`，锚点转 warp），`romatch/models/matcher.py:400-520`（Decoder 逐级流程），`:846-935`（`match`：上采样、certainty 衰减、对称），`:613-645`（采样）。

- **推理细节**：`roma_outdoor` 默认粗匹配 560、上采样 864（`romatch/models/model_zoo/__init__.py:35-36`）。
  - `attenuate_cert=True`：低分辨率 certainty 中为负的部分乘 0.5 后，从最终 certainty 中减去，用来压低粗级就不确定的区域。
  - warp 落在 $[-1,1]$ 之外的像素，certainty 置 0。
  - 采样方式 `threshold_balanced`：阈值 0.05；先多采 4 倍候选，再做 KDE（std 0.1）；邻居少于 10 个的点几乎不会被选中。

**损失**（`romatch/losses/robust_loss.py:43-161`；论文 §3.4）。

- **粗级**（「16」）：以分类做回归。把 GT warp 最近的那个锚点当作类别标签，在可靠像素（$p>0.99$）上计算交叉熵，再加 $0.01\cdot\mathrm{BCE}$(certainty)：
  $$\mathcal L_{coarse}=\mathrm{CE}(\text{logits}_{4096},\ \arg\min_k\|G_k-W_{gt}\|)+0.01\,\mathrm{BCE}$$
- **细级**（16/8/4/2/1 的细化器输出）：鲁棒回归，再加 $0.01\cdot\mathrm{BCE}$。设 $x$ 为 EPE（归一化坐标），$c_s=c\cdot s$：
  $$\rho(x)=c_s^{\alpha}\Big(\big(\tfrac{x}{c_s}\big)^2+1\Big)^{\alpha/2},\quad \alpha=0.5,\ c=10^{-4}$$
  在 0 附近它近似 L2，远处的梯度按 $|x|^{-1/2}$ 衰减，所以离群点不会主导训练。⚠论文写 $c=0.03$，单位是像素；代码在归一化坐标下取 $10^{-4}$，在 560 分辨率下约等于 0.028 px，两者一致。
- **局部掩码**：同 DKM，上一级 EPE 超过 $(2/512)\cdot d_s\cdot s$ 的像素不监督，$d_s=\{1{:}4,\,2{:}4,\,4{:}8,\,8{:}8\}$（`train_roma_outdoor.py:214-220`）。
- **GT warp 与掩码**：同 DKM，用深度加位姿投影，相对深度误差 < 0.05（`romatch/utils/utils.py:325-354`）。
- 粗级和细级之间梯度是切断的，两部分的编码器也不共享，所以 $\mathcal L=\mathcal L_{coarse}+\mathcal L_{fine}$ 不需要调权重（论文 §3.4）。

**训练**（论文 §4.2，`experiments/train_roma_outdoor.py:23,186-227`）。只在 MegaDepth 上训练，分辨率 560×560（论文的消融实验用 448）。学习率以 batch 8 为基准：解码器 1e-4，VGG 5e-6，按实际 batch 线性缩放；AdamW，weight decay 0.01；总样本数为 32×250k，在 90% 处降一次学习率。增强有水平翻转和 ±32 px 平移（`shake_t`），没有旋转（`rot_prob=0`）。

---

## 4. 训练数据对比

每行一个方法（都指 outdoor 版本），分成三张表。

**数据与样本**

| 方法 | 图像来源 | 一个样本包含什么 |
|---|---|---|
| SuperPoint | 合成形状；COCO 2014 训练集 8 万张 | 一张图 + 随机单应 $H$ → 图对 + 两份伪标签 |
| SuperGlue | MegaDepth（去掉 PhotoTourism 测试场景） | 图对 + 深度 + 内参 + 位姿；关键点由 SP 实时提取 |
| LoFTR | MegaDepth，368 个场景索引（每个场景按难度再拆 3 个子集） | 图对 + 深度 + 内参 + 相对位姿 |
| DKM / RoMa | MegaDepth（`train_loftr` 划分） | 同 LoFTR |
| AnyMatch | Any-syn：由 GLDv2 单张图合成 50 万对 | 同 LoFTR 格式，其中一张换成 IR、深度图或法向图 |

**选对与 GT**

| 方法 | 选对方式 | GT 怎么来 |
|---|---|---|
| SuperPoint | 不需要 | 关键点：单应适应伪标签；对应：$H$ |
| SuperGlue | 重叠分数在 [0.1, 0.7] | 单应或深度+位姿重投影；3 px 内为匹配，5 px 外为不可匹配 |
| LoFTR | 重叠分数 [0.1, 0.7] 的预计算索引，场景均衡采样 | 深度+位姿重投影到粗格子，加互为最近邻检查 |
| DKM / RoMa | 重叠 > 0.01 与 > 0.35 两个子集混合，场景加权 α=0.75 | 深度+位姿得到逐像素 warp；深度一致性掩码（< 0.05） |
| AnyMatch | 由单张图生成，不需要选对；SGCV 剔除差样本 | 合成时已知的深度和 [R\|T]；修补出的像素也监督 |

**分辨率与增强**

| 方法 | 训练分辨率 | 增强 |
|---|---|---|
| SuperPoint | 240×320 | 噪声、模糊、亮度 |
| SuperGlue | 论文只给出评测分辨率（长边 1600） | 补随机关键点 |
| LoFTR | 长边 840（或 640），补成正方形 | 无（代码中留了 TODO） |
| DKM / RoMa | DKM 540×720；RoMa 560² | ±32 px 平移；RoMa 另加水平翻转 |
| AnyMatch | Any-syn 图像 512²，沿用各模型自己的加载器 | 模态替换是主要的「增强」 |

出处：SP 见论文 §4–§6；SG 见论文附录 E；LoFTR 见 `configs/data/megadepth_trainval_{640,840}.py` 和论文 §4.2；DKM 见 `train_DKMv3_outdoor.py:186-196`；RoMa 见 `train_roma_outdoor.py:200-212`；AnyMatch 见论文 §3、§4.1 和下文。

### 4.1 AnyMatch 的数据怎么造（论文 §3）

AnyMatch 不改网络结构，只改训练数据：用单张照片合成「多视角 + 多模态」的图像对，然后用这些数据微调官方的 LoFTR、RoMa、EDM 权重。

```
GLDv2 单张图 (等比缩放后中心裁剪到 512²)
 ├ 视角变换:
 │   MoGe v1 估计单目相对深度 D_r → 随机仿射扰动 D = 1/(α/D_r + β)，α∈(0.5,2)，β∈(0,0.3)
 │   随机内参: 主点在图像中心，归一化焦距 f∈(0.58,0.88) → 反投影成点云
 │   随机外参: 旋转 ±7.5°，t_x,t_y ±0.3，t_z ±0.5 → 渲染新视角图，同时得到遮挡掩码
 │   Stable Diffusion v2 修补遮挡区域 → I_inp；同时得到 {K, [R|T], D, D'}
 ├ 样本筛选 (SGCV): 用 RoMa 在 (I_sin, I_inp) 上做稠密匹配，与 GT 投影比较
 │   PCK@5px ≥ 0.6 的对保留，否则丢弃
 └ 模态变换 (作用于 I_sin): IR 用 DiffV2IR 生成；深度图用 MoGe v1；法向图用 MoGe v2；
     事件相机数据用运动模拟 (只用于测试)
     → 多模态对 (I_modal, I_inp)，GT 直接沿用视角变换得到的
```

- **训练时怎么用**：训练只用 RGB-IR、RGB-Depth、RGB-Normal 三种组合（论文 §4.1，沿用 MINIMA 的做法）。数据整理成 MegaDepth 的格式。加载器是在 LoFTR 原 MegaDepth 加载器上加的一层：每个样本随机选一种模态，再随机选一侧的图换成该模态的版本，另一侧保持可见光，深度和位姿不变（`AnyMatch@259ad34:third_party/LoFTR_AnyMatch/src/datasets/megadepth.py:113-195`）。损失函数与上游相同，RoMa 的 `robust_loss.py` 与上游没有差异。
- **微调超参**（论文 §4.1）：4 张 RTX 4090。
  - LoFTR：每卡 batch 4，lr 8e-4，训 10 个 epoch。
  - RoMa：每卡 batch 2，编码器 lr 6.25e-8、解码器 1.25e-6，训 4 个 epoch。
  - EDM：lr 1e-4，训 10 个 epoch。
  - 论文的消融（表 4）显示，从官方权重微调明显好于从头训练。
- **公开情况**：README 只给出单视图源图（GLDv2、SA-1B）的获取说明和各模块的生成脚本，没有给出生成好的 Any-syn 训练集。
