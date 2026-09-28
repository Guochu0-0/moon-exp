# 在线自监督为什么没塌缩：RIPE / RIPE++ / DISK 对照 S2（#48）

> 问题：RIPE++ 的匹配器损失 ℒ_match = −E[R]，参考几何来自当前模型自己的 RANSAC，却没有塌缩；我们的 S2（每步用当前模型重估伪仿射）约 1000 步塌到 A ≈ [I|0]。原因是什么？给 #49（粗级闭式期望从 zero-shot 单独训练）什么约束？
>
> 标注约定：**【原文】** = 论文 LaTeX 源 / 官方代码里读到的，给出处；**【推断】** = 本文推断，未经实验验证。出处全文见 [sources.md](sources.md)。
> 读的版本：RIPE++ arXiv:2608.19693 源码包与 PDF；RIPEpp 代码 `0666e62`；RIPE++ 的 LightGlue 训练在作者的 glue-factory fork `192baa3`；RIPE arXiv:2507.04839 源码包与代码 `b173418`；DISK arXiv:2006.13566；SCENES arXiv:2401.10886；LoFTR `df7ca80`。
> S2 的代码和记录在 `scenes-baseline` 分支（草稿 PR #33），下文 `finetune/*`、`runs/S2/*`、`docs/design/rl-modeling.md` 都指该分支上的版本。

## 结论速览

1. **RIPE++ 匹配器的起点、冻结和数据三点都和 S2 不同。** 【原文】LightGlue 先用合成单应做**有真值**的预训练，RL 只是第二阶段微调。训练匹配器时提取器冻结，关键点和描述子是离线缓存的。训练对是 MegaDepth 的真实多视角图像对，没有做预对齐。
2. **RIPE++ 里没有任何项专门防「全判内点」。** 【原文】外点 −1 和「不可匹配」惩罚 η = 1e-4 防的是反方向的退化，也就是「全判不可匹配」。它的 kp_penalty 其实是在**鼓励**多匹配。论文也没有报告训练曲线，并且代码按**测试集**（MegaDepth1500 AUC@5）选模。
3. **S2 塌到恒等，最可能的原因是 LoFTR 自身结构。** 【推断】LoFTR 粗级在 cross-attention 之前给两张图加的是**同一套绝对位置编码**（`loftr.py:58-59`），所以「按位置匹配」本来就是一条走得通的捷径。我们的图像对又已经粗对齐，而且小于半个粗格（约 3.2 原图 px）的伪平移在 `np.rint` 后得到的就是恒等格。LightGlue 的 cross-attention 不带位置编码，RIPE 系提取器则是每张图单独算特征，两者都没有这条捷径。
4. 因此，把 RIPE++ 的写法照搬到 LoFTR 粗级，**预计仍会塌**。【推断】S3a 只监督内点格，已经很接近 RIPE++ 的形式，结果照样塌了。
5. 给 #49 的顺序：先做零训练成本的诊断，用 zero-shot 和 S2 ckpt 在**打乱的负样本对**上看恒等匹配的数量。然后依次试：①无约束基线 → ②**负样本对**（RIPE 式，负样本对上的内点给 −1）→ ③再加 L2-SP 锚定到 zero-shot → ④再加已知随机平移增强，配合变换一致性 → ⑤冻结 backbone 或匹配数上限，作为补充。见第 8 节。

---

## 0. S2 到底怎么训的（仓库事实）

- **损失**（`finetune/pseudo.py:96-145`）：先用本步的细级匹配跑仿射 RANSAC（3 px）得到 A。对每个**图内的光学粗格** i，按 A 算出 SAR 侧的格子 j（`np.rint`，`pseudo.py:54-63`），只对正样本 (i, j) 做 focal（sparse_spvs 形式，不监督负样本）。细级按 A 算窗口内偏移，用 l2_with_std。内点 < 20 的对不监督（`train.py` 的 `--min-inliers`）。
- **训练设置**：全部参数都训，模块保持 eval()（`finetune/model.py:26-28`）。AdamW，lr 1e-5，bs 1，8000 步。**没有**数据增强（`finetune/data.py`），**没有**锚定或正则。
- **塌缩过程**（`runs/S2/extra/train_collapse.json`）：step 100–800，匹配数从 330 涨到约 2000，内点率 55–60%。step 900 时匹配数 3683、内点 3492；step 1000 时 5773 个匹配几乎全是内点，loss 从 1.8 掉到 0.03。此后一直是 5776/5776，loss 为 0。Val AUC@5 0.013，中位误差 22 px（`runs/S2/metrics.json`）。
- **S3a**（`--coarse-set inliers`）只监督本步内点所在的格子，同样走到 5776 个内点、loss ≈ 0，Val 在 step1000 为 0.264，step2000 降到 0.250（`runs/S2/notes.md`）。
- 数据前提：光学–SAR patch 在构建时已粗对齐（`CONTEXT.md:10`）。

---

## 1. 参考几何从哪来

**【原文】RIPE++ 提取器**：每步用**当前**网络检测、描述，再做 MNN 和 F-RANSAC（OpenCV USAC_MAGSAC，阈值 1.0 px），直接当 reward。没有 teacher、EMA 或离线结果（`3_method.tex:42-45,80-91`；代码 `ripepp/train.py:500-515`、`conf/inl_th/constant.yaml`，值为 1.0）。

**【原文】RIPE++ 匹配器**：同样用**当前** LightGlue 的匹配（阈值 0.1 + mutual）估 F（OpenCV，1.5 px），内点 +1、外点 −1（`3_method.tex:186`；fork `gluefactory/models/utils/losses.py:191-207,317-362`；`ripepp+lightglue_megadepth_RL.yaml:62-63,75`）。reward 在最后一层算一次，中间层复用这份 reward 做深监督（`rl_lightglue.py:399-417`）。
- 但**起点不是随机初始化**：README 写明训练「follows the same two-step approach as the original LightGlue training」，先做 Homography Pre-Training，再用 `train.load_experiment=<预训练输出>` 进入 RL 微调（fork `README.md:65-85`）。预训练配置用 `ground_truth: matchers.homography_matcher` 和标准 LightGlue 监督损失（`ripepp+lightglue_homography.yaml:1-43`），也就是**已知合成单应的真值监督**。论文 §4.3 的说法是「retained the original two-stage training protocol: pretraining on synthetic image pairs … followed by fine-tuning on MegaDepth」（`4_experiments.tex:88`）。

**【原文】RIPE**：同样是在线估计，用当前网络的 MNN 加 PoseLib F-RANSAC。另外有负样本对：负样本对上的一致匹配 reward 取反（`r/sec/3_method.tex:143-165`）。

**【原文】DISK**：reward 来自 **GT 深度和位姿**，不是自估几何（`disk/tex/4-method.tex:40`）。**SCENES** 的 bootstrapping 是**离线**的：用 base model 为 100 万对图各估一次 F（`scenes/sec/4_experiments.tex:110`），而且每个 batch 按 1:1 混入 MegaDepth 的 GT 监督对（同上 :113）。

**对 AnyMatch-LoFTR 粗级**：参考几何这一点，RIPE++ 和 S2 **相同**，都是每步用当前模型重估。所以「在线自估」本身不是区别。区别在第 2、3 点和第 7 节讲的结构捷径。

## 2. 训练时冻结了什么

- **【原文】RIPE++ 提取器**：不冻结。VGG-19 用 ImageNet 预训练初始化，全部训练（`conf/backbones/vgg.yaml`；`4_experiments.tex:5`；RIPE `4_experiments.tex:6`）。
- **【原文】RIPE++ 匹配器**：**提取器冻结**，只训 LightGlue。`TwoViewPipeline` 默认 `extractor.trainable: False`（fork `two_view_pipeline.py:22-26`）；RL 配置用 `load_features.do: true` 读离线导出的 RIPE++ 关键点和描述子（`ripepp+lightglue_megadepth_RL.yaml:18-19`）。因此匹配器改变不了「有哪些点、描述子长什么样」，只能在固定的 512/2048 个点之间重新分配匹配。
- **S2**：全部参数都训。
- **对 AnyMatch-LoFTR 粗级**【推断】：可以对应到「冻结 backbone（ResNet-FPN），只训粗级 transformer」。但恒等捷径来自位置编码加 attention，在 transformer 内部，所以只冻 backbone **不能**消除捷径。它只是降低漂移速度，可以作为补充约束。

## 3. 图像对怎么构造，有没有已知的相对几何

- **【原文】RIPE++ 提取器**：MegaDepth（DISK 子集）的真实多视角正样本对；「no data augmentation beyond normalization, resizing the longer side to 560 pixels and padding」（`4_experiments.tex:9`；`conf/data/disk_megadepth.yaml`）。**没有**已知相对几何。
- **【原文】RIPE++ 匹配器**：阶段 1 在 revisitop1m 上做合成单应（difficulty 0.7、max_angle 45°），相对几何**已知**并当真值用。阶段 2 是 MegaDepth 真实对（overlap 0.1–0.7），相对几何未知（`ripepp+lightglue_homography.yaml:1-24`；`…_megadepth_RL.yaml:1-13`）。
- **【原文】RIPE++ Medical**（SCARED）：相隔 60 帧的视频帧组成训练对，**另加随机仿射增强**（旋转 ±30°、平移至多 10%、缩放 0.9–1.1），用途是「to mitigate overfitting」（`4_experiments.tex:68`）。增强的 T 没有被当成监督使用。
- **【原文】DISK**：MegaDepth 三元组，GT 深度；「otherwise we employ no data augmentation」（`5-experiments.tex:14`）。
- **对「恒等塌缩」是否关键**【推断】：
  - MegaDepth 对的真实对应**远离**恒等（视角变化大），从有真值预训练出发的模型一开始就远离恒等。我们的对是**粗对齐**的，zero-shot 的 A 本来就在恒等附近，也就是正好落在吸引域里。
  - 对 F 矩阵来说，「每点对应自身」是可以成立的（反对称 F，对应纯平移），但 RIPE 系提取器每张图独立出点和描述子，没法对任意一对图输出「同像素 ↔ 同像素」。LightGlue 只能在已给定的关键点之间配对，两张图的关键点坐标一般不重合，所以恒等解在它的动作空间里基本不存在。
  - 结论：RIPE++ **没有靠已知几何防塌**。它不塌主要因为数据本身离恒等很远，动作空间里也没有恒等解。我们两条都不满足。

## 4. 防退化的项及权重

| 项 | 方法 | 形式 | 权重【原文】 | 防的是哪种退化 |
|---|---|---|---|---|
| 外点惩罚 | RIPE++ 提取器 | RANSAC 外点 ρ_out | −0.1（ρ_in = 1）（表 5；`outlier_penalty_scheduler/constant.yaml`） | 「匹配很多但被 RANSAC 滤掉」不受罚（`3_method.tex:71-77`） |
| 外点惩罚 | RIPE++ 匹配器 | ν_out | −1.0（ν_in = 1）（表 5；RL yaml :62-63） | 同上 |
| 「没找到匹配」 | 两者 | λ，填在所有非匹配项上 | −1e-7（表 5；`fp_penalty`） | 可以忽略 |
| 关键点 logprob 正则 | RIPE++ 提取器（代码） | kp_penalty × logprob | −7e-7，按 beta 线性爬升（`train_default.yaml:63`，`train.py:557-590`）。**论文表 5 未列** | 防「不出关键点」 |
| 熵正则 ω·ℒ_H | RIPE++ 提取器 | 每个 8×8 cell 的 64 类 softmax 熵，**最小化** | 1e-6；1e-5 性能下降；1e-4 训练崩溃（`4_experiments.tex:100-101`，表 3，表 5，`heatmap_entropy_scheduler/constant.yaml`）。代码只在 `mask_matching` 选中的 cell 上取均值（`train.py:453-484`） | 让热图更尖，**不是**防塌缩 |
| 不可匹配惩罚 η·ℒ_nm | RIPE++ 匹配器 | 惩罚「期望的不可匹配点数」 | η = 1e-4（表 5；RL yaml :72 `kp_penalty: 0.0001`）。代码是对 dustbin 概率求和（`losses.py:231-238`） | 防「全判不可匹配」（`3_method.tex:187-190`） |
| 描述子对比损失 ψ·ℒ_desc | RIPE / RIPE++ 提取器 | HardNet：RANSAC 内点对拉近，最难负样本推开（margin） | 论文 ψ = 5.0；**RIPE++ 代码 weight 2.5**（`encoder_hard_net.yaml`）。RIPE 中 ψ ≤ 0.005 训不起来（RIPE 表 `desc_loss`） | 描述子的唯一梯度来源，自带「推开负样本」项 |
| 负样本对 | RIPE | 负样本对上一致匹配 reward 取反 | ±1 | 在负样本对上出一致匹配 |
| 关键点罚分 λ_kp，fp 退火 | DISK | λ_tp = 1、λ_fp = −0.25、λ_kp = −0.001；前 5 个 epoch 从 0 线性升到全值 | （`5-experiments.tex:18`） | 防「不出点」这个 reward = 0 的局部极大 |
| 网格采样 | RIPE / DISK | 每个 8×8 cell 只采 1 个点 | — | RIPE 称它和描述子损失一起防止「塌到对极点」（RIPE suppl「Towards collapsing to the epipoles」） |

- **【原文】没有 KL / L2-SP 锚定**，也没有 EMA teacher 或匹配数上限（RIPE++ 全文和代码都没找到）。
- **【原文】钻空子的实例**：在 RIPE++ 表 6 中，InfoNCE 描述子损失使内点数上升，位姿精度却没有上升（`X_suppl.tex:111`）。
- **【推断】关于「全判内点」**：RIPE++ 里没有任何项针对它。外点 −1 在「全部是内点」的不动点上不起作用，因为那时已经没有外点了。η 项在惩罚 dustbin，方向上**更鼓励**多匹配。所以对我们要防的退化，这张表只能提供思路（负样本对、网格/数量约束、推开负样本的对比项），不能照抄。
- **【推断】适用性**：
  - 负样本对：**适用**。见第 8 节。
  - 网格采样 / 匹配数上限：部分适用。LoFTR 每个粗格本来就只有一个位置，「5776 全匹配」相当于网格被填满；给匹配数设上限能挡住计数暴涨，但挡不住恒等。
  - 熵正则：在粗级**不适用**。粗级 dual-softmax 本来就尖，熵正则只会加快坍缩到 one-hot；在细级 5×5 上是另一回事。
  - η 项：粗级没有 dustbin，**不适用**。

## 5. 其他在线自监督训练匹配器或检测器的工作

| 工作 | 参考信号 | 在线？ | 报告过塌缩吗 | 怎么防 |
|---|---|---|---|---|
| DISK（NeurIPS 2020）【原文】 | GT 深度/位姿 | 是（reward 在线算，但参考几何是真值） | 随机初始化时 reward 平均为负，网络会「不再采样任何点」，停在 reward = 0 的局部极大 | λ_fp、λ_kp 在前 5 个 epoch 从 0 退火（`5-experiments.tex:18`）；按验证 mAA 选 ckpt |
| RIPE（ICCV 2025）【原文】 | 当前模型 MNN + F-RANSAC | 是 | 讨论过塌到对极点的可能，「never observed」；ψ 过小、ε 过大时训不起来 | 负样本对；HardNet 描述子损失；网格采样；ε 在前 1/3 训练中线性爬升（`r/sec/4_experiments.tex:14`） |
| RIPE++ 提取器【原文】 | 当前模型 MNN + F-RANSAC | 是 | ω = 1e-4 时训练崩溃 | 外点 −0.1；描述子损失；ω 很小 |
| RIPE++ 匹配器【原文】 | 当前 LightGlue 匹配 + F-RANSAC | 是 | 未报告；只有最终数字，按测试集选模（RL yaml :91-93 `select_by_test: true`，`best_key_test: megadepth1500/rel_pose_error@5°`） | 有真值预训练的起点；冻结提取器；lr 1e-5；外点 −1 |
| SCENES（2024）【原文】 | base model 估的 F | **否**（离线一次） | —（摘要与正文未报告塌缩） | 每个 batch 1:1 混入 MegaDepth GT 对 |
| 2403.12702（#52 起点）【原文，摘要】 | EM 伪标签 | 迭代（EM 轮次） | — | **冻结基础模型、只训 adapter**；加重建（信息一致性）损失，以「maintain the robustness of the FM's representation」 |
| 合成单应系（SuperPoint 单应自适应、SiLK）【原文，SiLK 摘要 / RIPE 相关工作 §2】 | 已知合成变换 | —（几何已知，不需要自估） | 不存在自估引起的塌缩 | 几何是真值 |
| AltO / SSHNet（多模态无监督单应，已有调研）【原文，转引 `label-free-reward` 调研】 | 相似度 | 是 | **塌到常数编码 + 恒等**；OPT-SAR 上一批方法停在恒等水平 | AltO 用交替优化 + Barlow Twins |

- #52 目前只有票面，没有评论或调研产出（2026-09-28 查）。其中唯一带「迭代」性质的是 2403.12702 的 EM；它防漂移的两个手段是冻结基础模型和重建锚定。
- 【推断】规律：凡是报告过「在线自估也能训」的，要么有**真值预训练起点加冻结**（RIPE++ 匹配器），要么有**把特征推开的对比项加独立单图特征**（RIPE / RIPE++ 提取器），要么有**负样本对**（RIPE）。凡是塌到恒等的（AltO、SSHNet、S2），都是**多模态、粗对齐或恒等附近本来就是好解**的情况。

## 6. 复核 `docs/design/rl-modeling.md` 第 7 节的待复核数字

**表 6 是哪张**【原文】：PDF 里表 6 是补充材料的「Ablation of the different components」（`rsc/tables/ablations_supplementary.tex`）。表 1–5 依次是 MegaDepth、SCARED、主消融、LightGlue、超参，表 7 是 Aachen（pdftotext 核对）。

**表 6 数字**（MegaDepth1500，AUC@5/10/20）：

| pos only | 描述子 | distance reward | curriculum | entropy | #inl | %inl | AUC@5 | @10 | @20 |
|---|---|---|---|---|---|---|---|---|---|
| | contrastive | | | | 297 | 34 | 51.83 | 65.37 | 75.94 |
| ✓ | contrastive | | | | 427 | 41 | 52.42 | 66.78 | 78.03 |
| ✓ | contrastive | ✓ | | | 384 | 40 | **54.75** | 68.33 | 78.92 |
| ✓ | contrastive | | ✓ | | 347 | 38 | 52.57 | 65.84 | 76.43 |
| ✓ | InfoNCE | | | | 430 | 45 | 51.72 | 65.0 | 75.9 |
| ✓ | InfoNCE | ✓ | | | 395 | 43 | 50.02 | 63.66 | 74.66 |
| ✓ | InfoNCE | | ✓ | | 370 | 41 | 49.18 | 61.93 | 72.85 |
| ✓ | InfoNCE | ✓ | ✓ | | 362 | 40 | 51.04 | 64.29 | 74.6 |
| ✓ | contrastive | ✓ | | 1e-6 | 374 | 40 | **53.01**\* | 67.12 | 78.01 |
| ✓ | contrastive | | | 1e-6 | 366 | 41 | **56.59**\* | 69.44 | 79.18 |

\* 原表最后两行的列错位：印成「67.12 / 78.01 / 53.01」和「69.44 / 79.18 / 56.59」。末行与表 3 的「56.59 / 69.44 / 79.18」对得上，所以两行都是 AUC@5 被挪到了最后一列。上表已按 AUC@5/10/20 的顺序改正。【原文 + 推断：错位的判定】

**设计文档 3.9 节的错误**：原文写「加上后 AUC@5 从 54.75 **降**到 52.57」。**这是错的。** 52.57 是 curriculum 那一行，不是 distance reward。正确读法【原文】：
- 不加熵正则时，distance reward 让 AUC@5 从 52.42 **升到** 54.75（+2.33）；
- 与熵正则一起用时，AUC@5 从 56.59 **降到** 53.01（−3.58）；
- 补充材料原话：「Curriculum learning, entropy regularization, and the distance-based reward each individually improve results, though their combination does not yield further cumulative gains」（`X_suppl.tex:112`）。代码默认 `use_distance_based_rewards: False`。
- 附带：LightGlue 匹配器的 RL 配置里 `reward_based_on_sampson_distance: false`（RL yaml :65），匹配器也没用分级 reward。

**熵正则权重**【原文】：正文（`4_experiments.tex:100-101`）、表 3、表 5（ω = 1e-6）和代码（`heatmap_entropy_scheduler/constant.yaml: 0.000001`）**四处一致**：1e-6 最好，1e-5 退化（表 3 中 AUC@5 49.62），1e-4 崩溃。表 3 中 AUC@5 从 52.42 到 56.59，+4.17，与「约 +4」一致。实际存在的出入是以下几处：
1. 正文把 ℒ_H 叫作「negative entropy」，但式 (4) 写的是 H = −Σ p log p 本身，并且是被最小化的（`3_method.tex:101-107`）。名字和公式不一致，实际做法是最小化熵。
2. 代码里另有一个 `linear_with_plateaus` 调度（0 → 1e-4），但不是默认。熵项只对被选中参与匹配的 cell 求平均，正文说的是每个 cell。
3. 描述子损失权重：论文 ψ = 5.0（表 5），RIPE++ 代码是 `weight: 2.5`（`encoder_hard_net.yaml`）。RIPE 代码是 5.0。
4. 不可匹配项：正文式中 ℒ_nm = Σ P_i(x_i)（这是「可匹配」概率），文字却说是「expected number of keypoints … non-matchable」。代码惩罚的是 dustbin 概率之和（`losses.py:231-238`），与文字一致，与式子相反。
5. RIPE（前作）：超参表 ε = −7e-8，消融表最好的是 −7e-7（63.48），代码是 −7e-7。
6. 小出入：表 1 中 RIPE++ AUC@5 为 56.58，表 3 为 56.59。

**要改的设计文档内容**（由主会话决定是否改）：
- 3.9 节「54.75 降到 52.57」改为上面的读法。
- 3.10 节权重描述正确，可以去掉「待复核」；可补一句「熵正则防的是热图发散，不是防塌缩」。
- 第 4 节 RIPE++ 行的「熵正则」只属于提取器，匹配器上没有熵正则；匹配器的正则只有 η = 1e-4 的不可匹配惩罚。

---

## 7. 为什么 S2 塌、RIPE++ 匹配器不塌：机制对照

| 维度 | RIPE++ 匹配器【原文】 | S2 / S3a（仓库） | 对恒等塌缩的影响【推断】 |
|---|---|---|---|
| 起点 | 合成单应**有真值**预训练 | zero-shot AnyMatch（其他域的有监督模型） | 两边都有好的起点，差别不大 |
| 冻结 | 提取器冻结，点和描述子固定 | 全训 | RIPE++ 的动作空间被固定点集限制住了 |
| 位置信息 | 只在 self-attention 里用**相对**旋转编码，cross-attention 不带位置（fork `lightglue.py:123-150,156-`） | 两图加**同一套绝对正弦 PE**，然后做 self/cross attention（LoFTR `loftr.py:58-64`） | **关键差异**：LoFTR 可以只靠位置把 i 配到 i |
| 数据相对几何 | MegaDepth 大视角，远离恒等 | 粗对齐，真值 ≈ 恒等 + 小偏移 | 我们的起点就在恒等吸引域里 |
| 粗网格量化 | 无（稀疏点坐标连续） | 伪仿射在粗格上取整（`pseudo.py:61`）；平移 < 约 3.2 原图 px 时目标就是恒等格 | 形成向恒等的正反馈 |
| 监督范围 | 只对**实际输出的匹配**给 ±1，其余约 0 | S2 对**所有图内格**监督；S3a 只对内点格监督 | S2 更快；S3a 也塌了，说明「只监督匹配」不够 |
| 「全判内点」时的梯度 | 仍然是 +1，没有防护 | 同样 | 两边一样，所以不是 RIPE++ 做对了什么 |
| 选模 | 按**测试集**峰值 | 按 Val 峰值 | RIPE++ 即使后期退化也可能看不出来 |

**核心推断**：RIPE++ 不塌，不是因为它的损失有防塌的项，而是**问题结构里不存在一个「自洽但错误」、网络又够得着的解**：F 矩阵下，固定点集之间凑不出大量一致匹配，而且数据远离恒等。LoFTR 粗级加上粗对齐数据，正好提供了这样一个解（绝对 PE → 恒等），而自洽类 reward 分不清它和真解。**这一条可以直接检验**，见下节诊断 0。

---

## 8. 给 #49 的建议（按顺序）

每一步都配 #49 已计划的随机 reward 对照。除 #49 已列的三项监控外，再加两项：**负样本对上的恒等一致匹配数**，以及**粗级 dual-softmax 中 conf[i,i] 的平均值**。

0. **诊断，零训练成本，先做**：用 zero-shot 和 S2 step1000 的 ckpt，在**打乱的负样本对**（光学和 SAR 来自不同 ROI）上跑推理加仿射 RANSAC，统计匹配数、内点数和 ‖A − [I|0]‖。
   - 如果 S2 ckpt 在负样本对上也给出近 5776 个恒等内点，说明是**位置捷径**，下面第 2 条就是对症的。
   - 如果 zero-shot 在负样本对上已经偏向恒等，说明捷径在起点就存在。
   - 可选：把 SAR 输入平移 k 个粗格（仓库已有 `--inject-shift`，但它是 roll），看 zero-shot 和 S2 的 A 是否跟着平移。
1. **无约束基线**（#49 原计划）：lr 1e-5，只加粗级 −Σ P·r，r 取 ±1（只对实际匹配），从 zero-shot 出发，约 2000 步，每 250 步记录一次。预期会塌，用来确认塌缩速度。
2. **加负样本对（最优先的约束）**：每个 batch 混入与正样本对等量的打乱对，负样本对上的 RANSAC 内点匹配给 −1。可参照 RIPE：负样本对上 reward 取反，非匹配给小正值（`r/sec/3_method.tex:143-165`）。理由【推断】：恒等解在**任何**一对上都自洽，负样本对是唯一直接惩罚「与内容无关的一致性」的信号。月面不同 ROI 基本不可能被错标成负样本，所以 RIPE++ 放弃负样本的理由（错标）在这里不成立。权重从 1:1 开始。
3. **在第 2 步基础上加 L2-SP 锚定到 zero-shot**（仓库已有 `--w-l2sp`）：相当于 RLHF 式的 KL 锚定，RIPE++ 没有这一项。它不能单独防恒等，但能拖慢漂移，给按 Val 选模留出窗口。权重从让 l2sp 项和主损失同量级开始扫。
4. **已知几何增强加变换一致性**：对 SAR 施加随机的已知平移或小仿射 T（平移 ≥ 1–2 个粗格，打破「粗对齐 ⇒ 恒等」），并加一项 ‖A_aug − T∘A_orig‖（或在 reward 里对不一致的对降权）。只加增强、不用 T，**不够**，因为恒等在增强后的对上仍然自洽。只有把 T 用作约束才起作用。代价是两次前向。
5. **补充约束，放在最后**：
   - 冻结 backbone，只训粗级 transformer：对应 RIPE++ 的「冻结提取器」，但捷径就在 transformer 里，预计作用有限；
   - 匹配数上限：每对只保留 conf 最高的 K 个，K 取 zero-shot 的中位数，约 300–500。这样能挡住「5776 全匹配」的计数暴涨，但挡不住恒等；
   - 不建议在粗级加 RIPE++ 的熵正则（方向是让分布更尖，对防塌没有帮助）。
6. **判据**：第 2 步如果能把负样本对上的恒等内点压到接近 zero-shot 水平，并且 Val AUC@5 不低于 zero-shot，就沿 2+3 继续。第 2 步如果仍然塌，就说明除了位置捷径还有别的机制（例如在正样本对上塌到「一致的整体偏移」），这时应转向 #49 里的「粗级只作辅助项，配合 S1 离线标签」。

## 9. 未确认的部分

- RIPE++ 匹配器 RL 阶段是否用过其他未写进 yaml 的命令行覆盖（例如 alpha 调度），无法从公开代码确定。`RLLoss.alpha` 的来源没有追到。
- RIPE++ 没有发布匹配器训练曲线，所以「没塌」只是从最终指标推出来的。加上按测试集选模，**不能排除**后期退化被选模掩盖。
- 位置捷径假设尚未实验验证，由第 8 节诊断 0 检验。
