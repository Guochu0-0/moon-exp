# 评价指标源码调研（05 自/无监督匹配 + 06 单应估计与形变配准）

> 调研日期：2026-09-23
> 范围：`D:\科研\光SAR配准\论文\05_自无监督匹配\`（17 篇）+ `06_单应估计与形变配准\`（3 篇）。
> 依据优先级：**源码（commit 锁定 permalink）> 论文原文（注 "论文 PDF p.X"，X 为 PDF 页序，不是期刊印刷页码）**。同目录 AI 总结 .md 只用来定位，结论都没有引用它。
> 源码和论文不一致的地方，一律记代码，并标注【不一致】。"未找到"的意思是：在列出的仓库或论文里查过，确实没有。
> 记号：「推断」= 我的判断，不是原文或代码直接给出的。

---

## 0. 结论速览（对我们的仿射评价协议最有用的几条）

1. **05 目录里只有 S2M2-SAR 一篇在光-SAR 上评测**（SEN1-2、QXS-SAROPT，模板匹配形式：只求平移），且**没有公开源码**。XCP-Match 做的是可见光-红外，不是 SAR。其余都是自然图像、医学或事件相机。
2. **单应角点误差的计算方式事实上已经统一**：4 个角点分别用 GT 单应和估计单应映射，取平均欧氏距离，再对"误差 ≤ t 的累积比例曲线"在 [0, t] 上求面积并除以 t 得 AUC。glue-factory（RIPE / RIPE++ 用）、SiLK、GeoFormer/immatch 的实现几乎逐行相同。**阈值档位各家不同**：RIPE 用 1/3/5 px，SiLK 用 1/2/3 px，GeoFormer 和 GIM 用 3/5/10 px，RaCo 用 1/3(/5) px。**误差都是在缩放后的图上算的**：短边 480（RIPE、SiLK、GeoFormer）或 640（RaCo，论文所述）。
3. **失败样本各家都没有剔除**，但记法不同：glue-factory 记 `inf`；immatch/GeoFormer 的 HPatches 记 `NaN`，FIRE 记 `1e6`；SiLK 记 `阈值+1`（在该阈值下等价于失败）。这些记法在 AUC 里的效果都是"算进分母、不算进分子"。
4. **鲁棒估计器与阈值差异很大，而且会影响排名**：glue-factory 的 poselib 默认阈值 1.0 px，RIPE 配置改成 0.5 px；并且 `ransac_th=-1` 时会**在测试集上扫描阈值 {0.5,…,3.0}、挑 mAA 最高的一档**。SiLK 和 GeoFormer 用 OpenCV `cv2.RANSAC`，阈值 3 px。RaCo 做位姿评测时对每个方法单独挑最优内点阈值（论文所述）。GIM 的 HPatches 表直接抄各基线原论文的数字，估计器不统一。
5. **无监督方法怎么选模型**：RIPE++ **用带 GT 位姿的验证集（IMW2020 val 的 200 对，或 SCARED val）上的 AUC@5° 保存 `best` checkpoint**。RIPE 只把验证结果记到 wandb，最后保存 final 权重。SiLK 和 GeoFormer 用自生成的合成单应验证集，分别监控 `val.f1` 和 `val_loss`，不需要人工标注。CA-Unsupervised 的代码里没有验证逻辑。
6. **人工标注真值点协议可以直接借鉴**：CA-Unsupervised 每对图人工标 6 个点，按"点的平均 L2"逐对计算，再按场景类别取均值。但代码里对每个点**取正、反两个方向误差中的较小者**，偏乐观。后来作者发布了经传统特征预匹配修正的 v2 坐标，总体误差从 1.82 降到 0.88，**说明标注噪声本身接近 1 px 量级**。FIRE（GeoFormer）的做法：控制点平均误差；失败记 1e6；在 1..25 px 上求 AUC；按难度组分别算 AUC 再平均得 mAUC；另外按 MAE>50 或 MEE>20 划为"不准确"。

---

## 1. 筛选表

| 论文 | 光-SAR / 遥感跨模态评测？ | 是否细看 | 源码 URL |
|---|---|---|---|
| R2D2 (NeurIPS'19) | 否（HPatches MMA、Aachen 定位） | 否 | https://github.com/naver/r2d2 （论文所列，未克隆） |
| DISK (NeurIPS'20) | 否（IMC2020 mAA；HPatches MMA 及 5px 内 MMA-AUC） | 否 | https://github.com/cvlab-epfl/disk （未克隆） |
| CAPS (ECCV'20) | 否 | **是**（HPatches 单应精度） | https://github.com/qianqianwang68/caps |
| PoSFeat (CVPR'22) | 否（HPatches MMA/MMAscore、Aachen、ETH SfM） | 否 | https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat （未克隆） |
| SiLK (ICCV'23) | 否 | **是**（HPatches 单应精度与 AUC） | https://github.com/facebookresearch/silk |
| DeDoDe (3DV'24) | 否（MegaDepth-1500 pose AUC、IMC2022） | 否 | https://github.com/Parskatt/DeDoDe （未克隆） |
| GIM (ICLR'24) | 否（ZEB 零样本 pose AUC@5°；HPatches 角点 AUC） | **是（简）** | https://github.com/xuelunshen/gim |
| RIPE (ICCV'25) | 否 | **是（参考算法）** | https://github.com/fraunhoferhhi/RIPE （评测在 https://github.com/JohannesK14/glue-factory） |
| RIPE++ (2026) | 否（MegaDepth-1500、SCARED1500 pose AUC） | **是（参考算法）** | https://github.com/fraunhoferhhi/RIPEpp （评测同上） |
| XCP-Match / CrossModalCompletionPretrain (IJCAI'25) | **跨模态但不是 SAR**（可见光-红外：METU-VisTIR、RoadScene、TriModalHuman） | **是（仅论文）** | 未找到（论文无链接，检索也没有结果） |
| S2M2-SAR / SemiSupMultiscale_SAROpt (arXiv'25) | **是**（SEN1-2、QXS-SAROPT） | **是（仅论文）** | 未找到（论文无链接，检索也没有结果） |
| SSMB (2026) | 否（Blur-HPatches MMA@3、pose AUC） | 否 | 论文说代码会发布，未给链接 |
| TraqPoint / TrackAwarePolicyGradient (2026) | 否（MegaDepth-1500/ScanNet pose AUC、Aachen） | 否 | 未找到 |
| RaCo (3DV'26) | 否 | **是**（HPatches 单应角点 AUC） | https://github.com/cvg/RaCo |
| LabelFreeEventImage (2026) | 否（事件-图像，MVSEC/TUM-VIE pose AUC@5/10/20） | 否 | https://github.com/ZhonghuaYi/nexus2-official （论文所列，未克隆） |
| FrozenBackboneAdapt (2026) | 否（医学 MRI，Learn2Reg LUMIR：Dice/HD/NDV/TRE） | 否 | 未找到 |
| MuM (2025) | 否（探针、MegaDepth/ScanNet-1500 pose AUC、EPE） | 否 | https://github.com/davnords/mum （未克隆） |
| CA-Unsupervised (ECCV'20，06) | 否（自建视频帧单应数据集） | **是** | https://github.com/JirongZhang/DeepHomography |
| GeoFormer (ICCV'23，06) | 否（HPatches、ISC-HE、FIRE 视网膜） | **是** | https://github.com/ruc-aimc-lab/GeoFormer |
| H-ViT (CVPR'24，06) | 否（3D 脑 MRI 形变配准：Dice/HD95/SDlogJ/%\|J\|≤0） | 否：是稠密形变场，不是单应/仿射，协议不可迁移 | https://github.com/mogvision/hvit （检索得到，未克隆） |

未细看各篇的主评测（一行）：
- R2D2：HPatches MMA@1–10px（D2-Net 协议）加 Aachen Day-Night 定位（论文 PDF p.7–8）。
- DISK：IMC2020 stereo/multiview mAA；HPatches MMA@1–10px，并汇总为 5px 以内的 AUC（论文 PDF p.8 图 5）。
- PoSFeat：HPatches MMA，以及对 MMA@1..10 按 (2−0.1·thr) 加权得到的 MMAscore；Aachen；ETH SfM。
- DeDoDe：MegaDepth-1500 pose AUC@5/10/20；MegaDepth 重复率；IMC2022。
- SSMB：Blur-HPatches MMA@3px；多套 pose AUC。
- TraqPoint：MegaDepth-1500/ScanNet pose AUC@5/10/20；Aachen (0.25m,2°)/(0.5m,5°)/(1m,10°)。
- LabelFreeEventImage：MVSEC/TUM-VIE 相对位姿 AUC@5/10/20。
- FrozenBackboneAdapt：LUMIR 官方平台的 Dice、HD95、NDV、TRE。
- MuM：MegaDepth/ScanNet-1500 相对位姿 AUC、稠密匹配 EPE 与鲁棒性、线性探针。
- H-ViT：OASIS 上 Dice、HD95、SDlogJ；其余数据集 Dice 与 %|J|≤0。

---

## 2. RIPE（ICCV 2025，参考算法）

- 源码：`fraunhoferhhi/RIPE@b173418`（训练与验证），`JohannesK14/glue-factory@192baa3`（论文表格的评测，README 指向这里）。
- 评测入口：`python -m gluefactory.eval.hpatches --conf ripe+NN`、`gluefactory.eval.megadepth1500 --conf ripe+NN`（[README L113-L114](https://github.com/fraunhoferhhi/RIPE/blob/b173418008f4cb77a2ebffb570a7cab69e87cd08/README.md#L113-L114)）。

**指标**
- HPatches：单应角点误差 AUC@1/3/5 px（键名 `H_error_ransac@{1,3,5}px`）。[hpatches.py L142-L155](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/hpatches.py#L142-L155)。README 给出的复现值为 0.3793/0.5893/0.692（[README L276-L278](https://github.com/fraunhoferhhi/RIPE/blob/b173418008f4cb77a2ebffb570a7cab69e87cd08/README.md#L276-L278)）。论文 PDF p.7 写"AUC for the thresholds of 1, 3 and 5 pixels"，两者一致。
- 同一流程还会输出：匹配点的对称单应误差精度 `prec@1px/3px`（[utils.py L137-L156](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/utils.py#L137-L156)）；不经 RANSAC 的加权 DLT 角点 AUC `H_error_dlt@*`（[utils.py L241-L261](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/utils.py#L241-L261)）；以及各标量的中位数 `m*`（[hpatches.py L135-L140](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/hpatches.py#L135-L140)）。论文只报了 RANSAC 那一组 AUC。
- MegaDepth-1500：pose error 取 max(旋转角误差, 平移角误差)，报 AUC@5/10/20°（[megadepth1500.py L144-L145](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/megadepth1500.py#L144-L145)）。

**误差在什么上算**
- 角点：`corners0 = [[0,0],[W,0],[W,H],[0,H]]`，分别用 GT 单应和估计单应映射，取 4 点欧氏距离的均值（[geometry/homography.py L336-L342](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/geometry/homography.py#L336-L342)）。注意角点取的是 W、H，而 SiLK 和 immatch 取 W−1、H−1。
- 在**缩放后的分辨率**上计算：短边缩放到 480（[hpatches.py L30-L49](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/hpatches.py#L30-L49)），GT 单应随缩放同步变换，`H = T1 @ H @ inv(T0)`（[datasets/hpatches.py L103](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/datasets/hpatches.py#L103)）。
- 样本集：默认剔除 8 个大图序列（[datasets/hpatches.py L46-L56](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/datasets/hpatches.py#L46-L56)），每个序列取 1→2..6 共 5 对（[L71-L77](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/datasets/hpatches.py#L71-L77)），合计 108×5 = 540 对（「推断」：按 116−8 计算）。

**成功判定与阈值**
- 没有单独的"成功率"指标。AUC 的计算：误差排序后求累积召回，在 [0,t] 上做梯形积分再除以 t（[utils/tools.py L133-L145](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/utils/tools.py#L133-L145)）。多档：1/3/5 px。

**失败样本**
- 估计器失败时记 `H_error_ransac = inf`（[utils.py L224-L232](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/utils.py#L224-L232)）。DLT 抛异常时单应置为全 inf（[L243-L257](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/utils.py#L243-L257)）。位姿失败同样记 inf（[L181-L184](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/utils.py#L181-L184)）。inf 算进 AUC 的分母，等价于"记为失败且保留"。
- 是否失败由 `success = M is not None` 判定（[robust_estimators/homography/poselib.py L16-L32](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/robust_estimators/homography/poselib.py#L16-L32)）。poselib 在匹配数少于 4 时是返回 None 还是返回一个退化矩阵，**未核实**（没有运行，也没读 poselib 的 C++ 源码）。

**鲁棒估计器**
- PoseLib `estimate_homography`，参数 `max_reproj_error = ransac_th`，其余用 poselib 默认值（迭代数等未在配置里设置）。
- 流水线默认 `ransac_th: 1.0`。**`ransac_th = -1` 时会依次试 [0.5,1.0,…,3.0]，在测试集上选 mAA 最高的一档**（[hpatches.py L101-L105](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/hpatches.py#L101-L105)，[utils.py L264-L289](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/eval/utils.py#L264-L289)）。
- RIPE 的配置把 HPatches 和 MegaDepth 都**固定为 0.5 px**，HPatches 的 top_k 改为 1024（[ripe+NN.yaml L1-L24](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/configs/ripe%2BNN.yaml#L1-L24)）。论文 PDF p.7 只说"poselib … restricted the number of keypoint to 1024"，**没有写 0.5 px**。
- 匹配：所有方法都用 MNN，都用 poselib，即对比方法共用同一个估计器（论文 PDF p.7 所述；fork 里的 `*+NN.yaml` 配置结构相同）。
- 【不一致】MegaDepth 的缩放：论文 PDF p.7 说稀疏方法用长边 1200，但 `ripe+NN.yaml` 写的是 `resize: 1600`。README 复现值 0.5511 与论文表里的 55.11 一致，说明表中数字就是用 1600 跑出来的。

**聚合方式**：所有样本对放在一起算一条 AUC 曲线（按样本聚合，不分场景平均）；其余标量取中位数。

**模型选择**
- 训练过程中每 2000 步在 IMW2020 验证集的 200 对（id 硬编码）上算 pose AUC，**这一步需要 GT 相机位姿**（[train.py L148-L152](https://github.com/fraunhoferhhi/RIPE/blob/b173418008f4cb77a2ebffb570a7cab69e87cd08/ripe/train.py#L148-L152)、[L395-L397](https://github.com/fraunhoferhhi/RIPE/blob/b173418008f4cb77a2ebffb570a7cab69e87cd08/ripe/train.py#L395-L397)、[benchmarks/imw_2020.py L52-L64](https://github.com/fraunhoferhhi/RIPE/blob/b173418008f4cb77a2ebffb570a7cab69e87cd08/ripe/benchmarks/imw_2020.py#L52-L64)）。验证时用 poselib 相对位姿，`max_epipolar_error 0.5`，失败记 inf（[imw_2020.py L120-L152](https://github.com/fraunhoferhhi/RIPE/blob/b173418008f4cb77a2ebffb570a7cab69e87cd08/ripe/benchmarks/imw_2020.py#L120-L152)）。
- 代码只把验证结果记到 wandb，**不据此保存 best**，最后保存 `_final.pth`（[train.py L403-L406](https://github.com/fraunhoferhhi/RIPE/blob/b173418008f4cb77a2ebffb570a7cab69e87cd08/ripe/train.py#L403-L406)）。论文发布的权重是 final 还是人工挑的，论文没有说明。

---

## 3. RIPE++（2026，参考算法）

- 源码：`fraunhoferhhi/RIPEpp@0666e62`；评测同样用 `JohannesK14/glue-factory@192baa3`（[README L67-L68](https://github.com/fraunhoferhhi/RIPEpp/blob/0666e62bf00569870bbcc1fe4b2e03976fc563f8/README.md#L67-L68)）。
- **论文只报相对位姿，不报 HPatches**：MegaDepth-1500 和 SCARED1500 的 AUC@5/10/20°，长边 1200，top 2048，MNN，PoseLib，RANSAC 内点阈值 0.5（SCARED 用 1.0）（论文 PDF p.10–11、p.13）。基线 RaCo 和 DaD 搭配 ALIKED-n16 描述子，**所有方法都用 0.5 阈值**（论文 PDF p.11）。
- 代码配置：`ripepp+NN.yaml` 里 MegaDepth 设为 1200 和 poselib 0.5，与论文一致（[ripepp+NN.yaml L14-L21](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/configs/ripepp%2BNN.yaml#L14-L21)）。HPatches 段写的是 `inference_conf: topper: 1024`（[L30-L37](https://github.com/JohannesK14/glue-factory/blob/192baa367afc800614b224093b733433e012592b/gluefactory/configs/ripepp%2BNN.yaml#L30-L37)）。`topper` 疑似 `top_k` 的笔误；因为 `detectAndCompute(**inference_kwargs)` 接受任意 kwargs（[models/ripepp.py L139](https://github.com/fraunhoferhhi/RIPEpp/blob/0666e62bf00569870bbcc1fe4b2e03976fc563f8/ripepp/models/ripepp.py#L139)），这个覆盖**可能没生效**（「推断」，未运行验证）。
- 失败处理和 AUC 与 RIPE 相同（共用 glue-factory；训练验证用的 `imw_2020.py` 失败时也记 inf，[imw_2020.py L132-L164](https://github.com/fraunhoferhhi/RIPEpp/blob/0666e62bf00569870bbcc1fe4b2e03976fc563f8/ripepp/benchmarks/imw_2020.py#L132-L164)）。
- 训练时的估计器：用 OpenCV `usac_magsac` 估计 F 矩阵，代替 poselib（论文 PDF p.10；`conf/estimator/F_opencv.yaml`）。仓库还提供 `H_poselib` 单应估计器（[H_estimator_poselib.py L1-L25](https://github.com/fraunhoferhhi/RIPEpp/blob/0666e62bf00569870bbcc1fe4b2e03976fc563f8/ripepp/geometric_matching/robust_estimator/H_estimator_poselib.py#L1-L25)）。对我们来说，**如果把 reward 模型换成仿射或单应，可以直接照这个接口写**。
- **模型选择（关键）**：每 1000 步在验证集上评测，**若 AUC@5° 超过历史最好，就保存 `model_{name}_best.pth`**（[train.py L304-L310](https://github.com/fraunhoferhhi/RIPEpp/blob/0666e62bf00569870bbcc1fe4b2e03976fc563f8/ripepp/train.py#L304-L310)、[L768-L791](https://github.com/fraunhoferhhi/RIPEpp/blob/0666e62bf00569870bbcc1fe4b2e03976fc563f8/ripepp/train.py#L768-L791)）。验证集是 IMW2020 预定义子集或 SCARED val 的 200 对（`min_overlap 0.4`），**都依赖 GT 位姿**（`conf/val/imw2020.yaml`、`conf/val/scared.yaml`）。也就是说，这个"只用无标注图像对训练"的方法在选 checkpoint 时用了带标签的验证集。论文没有说明最终发布的是 best 还是 final。

---

## 4. SiLK（ICCV 2023）

- 源码：`facebookresearch/silk@7b9614b`；评测入口是 `etc/mode/run-hpatches-tests-silk.yaml`，它继承 `run-hpatches-tests-default.yaml`。
- **指标**：重复率@1/2/3、单应估计精度（HEA）@1/2/3、单应 AUC@1/2/3、MMA@1/2/3、平均关键点数与匹配数（[run-hpatches-tests-default.yaml L1-L71](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/etc/mode/run-hpatches-tests-default.yaml#L1-L71)）。
- **误差在什么上算**：4 个角点取 (0,0)、(W−1,0)、(0,H−1)、(W−1,H−1)，GT 和估计单应分别映射后取欧氏距离均值（[hpatches_metrics.py L204-L248](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/lib/metrics/hpatches_metrics.py#L204-L248)）。
- **成功判定**：`mean_dist <= correctness_thresh` 即为正确；HEA 是正确样本的比例（[L245](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/lib/metrics/hpatches_metrics.py#L245)、[L270-L277](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/lib/metrics/hpatches_metrics.py#L270-L277)）。AUC 的积分上限就是单个阈值（[L175-L188](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/lib/metrics/hpatches_metrics.py#L175-L188)、[L250-L268](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/lib/metrics/hpatches_metrics.py#L250-L268)）。
- **失败**：单应为 None（匹配少于 4 个，或 OpenCV 返回 None）时，记 correctness 0、误差 `thresh + 1.0`（[L210-L215](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/lib/metrics/hpatches_metrics.py#L210-L215)），不剔除。
- **估计器**：MNN 加 `cv2.findHomography(..., cv2.RANSAC)`，**不传阈值，即使用 OpenCV 默认的 3.0 px**；匹配少于 4 个返回 None（[lib/matching/mnn.py L107-L134](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/lib/matching/mnn.py#L107-L134)）。论文 PDF p.5 写"use OpenCV RANSAC"。DISK、R2D2、LoFTR 的结果以缓存的 h5 预测读入，再走同一套度量和 RANSAC（`etc/mode/run-hpatches-tests-cached-*.yaml`），所以**对比方法共用同一个估计器**。
- **尺度**：`min_img_size: [480, 480]`（[etc/datasets/hpatches/test.yaml L12](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/etc/datasets/hpatches/test.yaml#L12)），论文 PDF p.5 说短边 480。
- **聚合**：所有样本对一起算均值或 AUC。
- **模型选择**：`ModelCheckpoint(monitor="val.f1", save_top_k=10)`，验证集是 COCO val 上的**自生成合成单应对**，不需要人工标签（[etc/mode/train-silk.yaml L36-L38](https://github.com/facebookresearch/silk/blob/7b9614b4a66361a0003aaa6fe9298ccdb267714b/etc/mode/train-silk.yaml#L36-L38)）。

---

## 5. RaCo（3DV 2026）

- 源码：`cvg/RaCo@35790eb` **只有推理代码**。README 说"The training and evaluation code will be released in Glue Factory"（[README L215](https://github.com/cvg/RaCo/blob/35790eb48074ed14839d0fb496b8806caa4e766b/README.md#L215)）。在上游 `cvg/glue-factory@2d17e3b` 里搜 "raco" 没有结果，所以**评测代码未找到**，以下都依据论文。
- 论文协议（PDF p.9–10 附录）：
  - HPatches 共 540 对，短边缩放到 640；每张图固定 1024 个关键点。
  - **对应关系用 GT 单应生成**：互为最近邻且重投影距离 3 px 以内。在全部对应上跑 **PoseLib 的 DLT（不用 RANSAC）**。角点误差 AUC 取 1 px 和 3 px。
  - 这是检测器评测，不涉及描述子匹配，因此和 RIPE 的"MNN 描述子 + RANSAC"**不可直接比较**。
- README 另外给了 RaCo+LightGlue+ 的 HPatches AUC@1/3/5，分 DLT 和 PoseLib 两列（[README L158-L165](https://github.com/cvg/RaCo/blob/35790eb48074ed14839d0fb496b8806caa4e766b/README.md#L158-L165)）。**同一方法换估计器，AUC@1px 从 40.4 变成 44.7**，可以直接说明估计器对结果影响有多大。
- 位姿评测（PDF p.10）：**对每个方法单独在 {0.5,…,3.0} 中挑最优内点阈值**。论文没有说明是在测试集还是验证集上挑的。
- 失败处理、模型选择：论文未说明，代码未找到。

---

## 6. CAPS（ECCV 2020）

- 论文（PDF p.10–11）：采用 SuperPoint 的"corner correctness"协议，4 角点平均误差小于 ε 即为正确，报 ε = 1/3/5 px 下的精度（表 1）；每张图最多 1000 个关键点，在 MNN 匹配上"robustly estimate"单应。**RANSAC 的具体类型和阈值没有写**。
- 源码 `qianqianwang68/caps@1cb601a`：只有 HPatches 特征提取脚本 `extract_features.py`，数据准备指向 D2-Net 仓库（README L54-L59）。**单应评测代码未找到**；唯一的评测脚本是 `test/eval_pose_megadepth.py`。
- 失败处理、聚合方式、模型选择：未找到。

---

## 7. GIM（ICLR 2024，简）

- HPatches（论文 PDF p.9）：OpenCV RANSAC，4 角点平均重投影误差，AUC@3/5/10 px。**"We take the numbers from the original paper for each baseline"**，所以基线和 GIM 的估计器与阈值并不统一。
- 源码 `xuelunshen/gim@f09105a`：**没有 HPatches 评测代码**（未找到）。ZEB 零样本评测的默认设置是 `--ransac MAGSAC --ransac_threshold 0.5`（[test.py L121-L124](https://github.com/xuelunshen/gim/blob/f09105a0555eef5c93db032d97b9e3d8a4cccb20/test.py#L121-L124)），报 pose AUC@5°，并对 12 个数据集求平均排名（论文 PDF p.5）。

---

## 8. XCP-Match / CrossModalCompletionPretrain（IJCAI 2025，仅论文）

- 模态：可见光-红外（热红外），**不是 SAR**。
- 单应任务（论文 PDF p.6–7）：在 RoadScene 上**随机合成单应作为 GT**，参数为平移 [−10,10]、旋转"[−10,−10]"（原文如此，疑为笔误）、缩放 [0.8,1.2]、剪切 [−0.1,0.1]、透视 [−0.001,0.001]。原文 "The evaluation metrics still use AUC. Since the homography matrix describes geometric transformations in planar scenes, we approximate the scene as planar to estimate the camera pose"，也就是说**单应任务报的是 pose AUC（@5/10/20），不是角点误差**，写法少见，也和标准协议不一致。
- RANSAC 阈值 3（px），长边缩放到 640（PDF p.6，位姿任务段）。配准任务用分割标注的 LTA/IoU（TriModalHuman）。
- 失败处理、模型选择：论文未说明；源码未找到。

---

## 9. S2M2-SAR / SemiSupMultiscale_SAROpt（arXiv 2025，唯一的光-SAR 评测，仅论文）

- 数据（论文 PDF p.9）：
  - SEN1-2 用固定随机种子打乱，前 90.7%（256,000 对）作训练，其余 26,384 对作测试。
  - QXS-SAROPT 按 80/20 划分（16,000 / 4,000）。
  - 任务是**模板匹配**：192×192 模板在 256×256 参考图中定位，**只估平移**。
- 指标（PDF p.9 §4.1.4）：
  - **RMSE(All)**：预测匹配位置与 GT 位置欧氏距离的"平均"，覆盖全部测试样本。名字叫 RMSE，按文字描述实际是平均欧氏距离，是否先平方再开根不明确。
  - **CMR(T=1/5)**：预测落在 GT 周围 T px 以内的样本比例，即成功率。
  - **RMSE(T=5)**：**只在误差 ≤ 5 px 的样本上**求平均误差，即条件误差，会把失败样本排除掉。
  - 另有每对推理时间（ms）。
- 失败样本：模板匹配总会输出一个位置，不存在"无输出"。大误差在 RMSE(All) 里保留，在 RMSE(T=5) 里被剔除。
- 基线：Semi-I2I 先翻译再做 FFT-NCC；MARU-Net、OSMNet 用官方实现；FFT+U-Net 是重新实现的（PDF p.9）。
- 模型选择：PDF p.11 图 6、图 7 是在"Validation Set"上跟踪伪标签的 RMSE 和 FMR(T=5)，**这需要 GT**；论文没说是否据此选 checkpoint，也没说明验证集从哪里划分。
- 源码：未找到。

---

## 10. CA-Unsupervised / DeepHomography（ECCV 2020，06）

- 源码：`JirongZhang/DeepHomography@3e811b7`；评测入口 `Oneline-DLTv1/test.py`。
- **真值**：测试集 4.2k 对，每对人工标注 6–8 个均匀分布的对应点（论文 PDF p.10）。代码固定取前 6 个点（[test.py L141-L155](https://github.com/JirongZhang/DeepHomography/blob/3e811b7d06f84de34eae056baaf90ca886ae3908/Oneline-DLTv1/test.py#L141-L155)）。
- **误差**：点到点 L2 距离，`p2 − H·p1`，经透视归一化（[L14-L28](https://github.com/JirongZhang/DeepHomography/blob/3e811b7d06f84de34eae056baaf90ca886ae3908/Oneline-DLTv1/test.py#L14-L28)）。**每个点取正反两个方向误差中的较小者**，注释解释为"the data annotator has no fixed left or right when labelling"（[L146-L152](https://github.com/JirongZhang/DeepHomography/blob/3e811b7d06f84de34eae056baaf90ca886ae3908/Oneline-DLTv1/test.py#L146-L152)）。这样做偏乐观：随机或恒等变换也可能被"碰巧"判为较小误差。
- **聚合**：每对取 6 点平均，再按 5 个场景类别（RE/LT/LL/SF/LF）分别求均值（[L161-L190](https://github.com/JirongZhang/DeepHomography/blob/3e811b7d06f84de34eae056baaf90ca886ae3908/Oneline-DLTv1/test.py#L161-L190)）。论文表中的 Avg 列，代码里**没有计算**（未找到）；「推断」是 5 个类别均值的平均。
- **成功判定**：论文 PDF p.13 另报"3 px 内的内点百分比"（表 2b）。**这部分代码未找到**。
- **失败**：网络直接回归 H，总有输出，不存在失败分支。I₃ₓ₃（不做变换）作为参考下界一起报告。
- **对比方法的估计器**：SIFT/ORB/LIFT/SOSNet 分别与 RANSAC、MAGSAC 两两组合（论文 PDF p.10），阈值未说明。
- **标注噪声**：README 给出 v2 坐标（用传统描述子预匹配修正人工标注）后，本方法总体误差从 1.82 降到 0.88（[README L9-L10](https://github.com/JirongZhang/DeepHomography/blob/3e811b7d06f84de34eae056baaf90ca886ae3908/README.md#L9-L10)）。「推断」：纯人工点标注的噪声约 1 px，低于这个量级的方法间差异在 v1 标注下无法分辨。
- **模型选择**：`train.py` 每 4000 次迭代保存一次（[L75-L90](https://github.com/JirongZhang/DeepHomography/blob/3e811b7d06f84de34eae056baaf90ca886ae3908/Oneline-DLTv1/train.py#L75-L90)），没有验证循环。仓库里有 `Data/Val_List.txt`（18,200 对），但代码没有用到。

---

## 11. GeoFormer（ICCV 2023，06）

- 源码：`ruc-aimc-lab/GeoFormer@8b9506e`，评测部分改自 immatch（image-matching-toolbox）。入口是 `eval_Hpatches.py`、`eval_FIRE.py`、`eval_ISC.py`。

**HPatches / ISC-HE**
- 角点：(0,0)、(0,h−1)、(w−1,0)、(w−1,h−1)，在缩放后尺寸上计算（`w/scale`），误差取 4 点距离均值（[hpatches_helper.py L213-L240](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/hpatches_helper.py#L213-L240)）。HPatches 的 `imsize: 480`（[eval_configs/geoformer.yml L7-L11](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_configs/geoformer.yml#L7-L11)），论文 PDF p.6 说短边 480。
- 指标：一是 1/3/5/10 px 下的"Hest Correct"比例；二是 AUC@1/3/5/10，按 all/i/v 三组分别计算（[L36-L56](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/hpatches_helper.py#L36-L56)、AUC 实现在 [L13-L25](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/hpatches_helper.py#L13-L25)）。论文报 AUC@3/5/10 及三者均值 mAUC。
- 失败：`H_pred is None` 时记 `corner_dist = np.nan`，并累计 `h_failed`（[L222-L226](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/hpatches_helper.py#L222-L226)）。NaN 在 `np.sort` 后排到最后，算进分母、不算进分子，等价于失败。
- 估计器：命令行默认 `--h_solver cv --ransac_thres 3`，即 `cv2.findHomography(RANSAC, 3)`（[eval_Hpatches.py L96-L100](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_Hpatches.py#L96-L100)）。函数签名里的默认值却是 degensac 加阈值 2（[L12-L18](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_Hpatches.py#L12-L18)），不同入口会得到不同结果。
- 样本集：遍历 `hpatches-sequences-release` 下的全部序列，不剔除大图（论文 PDF p.6 称 57+59 序列）。但 MMA 汇总里硬编码 `n_i = 52, n_v = 56`（[L61-L62](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/hpatches_helper.py#L61-L62)），和"全部 116 个序列"不一致。这只影响 MMA，不影响单应 AUC。
- 【不一致】README 的复现输出为 AUC@3/5/10 = 0.7206/0.7997/0.8768，论文表 2 中 GeoFormer 是 68.0/76.8/85.4（PDF p.6）。

**FIRE（视网膜，GT 控制点，与我们的"真值点"协议最接近）**
- 误差：用估计单应把控制点映射过去，取与 GT 的**平均距离**。另外计算最大误差 MAE 和中位误差 MEE，**MAE > 50 或 MEE > 20 判为"不准确"**（[fire_helper.py L159-L182](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/fire_helper.py#L159-L182)）。
- 失败：单应为 None 时记误差 `1e6` 并计为 failed（[L152-L157](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/fire_helper.py#L152-L157)）。最后输出 Failed / Inaccurate / Acceptable 三类百分比（[L234-L235](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/fire_helper.py#L234-L235)）。
- AUC：在阈值 1..25 px 上取离散成功率的平均，**按 S/A/P 三个难度组分别计算，再取算术平均得到 mAUC**（[L11-L42](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/fire_helper.py#L11-L42)）。这是按组平均，不是按样本平均。
- 剔除了一个标注有误的对 `control_points_P37_1_2.txt`（README，以及代码里 `assert len(p_error) == 48`）。
- 估计器：`cv2.RANSAC`，阈值 15，推理尺寸 768，误差在原始 2912 分辨率上计算，估计的单应会被缩放回原图（[eval_FIRE.py L101-L105](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_FIRE.py#L101-L105)、[fire_helper.py L141-L147](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/eval_tool/immatch/utils/fire_helper.py#L141-L147)）。

**模型选择**：`ModelCheckpoint(monitor='val_loss', save_top_k=5)`，val 与训练用同一个自生成合成单应管线，无人工标签（[lightning/train_homo_geoformer.py L107-L110](https://github.com/ruc-aimc-lab/GeoFormer/blob/8b9506e6e9c0e61848724955fb514824039b1ff7/lightning/train_homo_geoformer.py#L107-L110)）。

---

## 12. 横向对照表

| 方法 | 任务/数据 | 误差定义 | 阈值档 | 失败计入 | 估计器（阈值） | 共用估计器？ | 聚合 | 模型选择 | 依据 |
|---|---|---|---|---|---|---|---|---|---|
| RIPE | HPatches 单应 | 4 角点 (0..W,0..H) 平均距离，480 短边 | AUC@1/3/5 | inf，保留 | PoseLib H，0.5 px（配置写死；论文未写） | 是（MNN+PoseLib） | 全样本 AUC；标量取中位数 | 带 GT 的 IMW val 只记日志，存 final | 代码 |
| RIPE | MegaDepth-1500 位姿 | max(R,t) 角误差 | AUC@5/10/20° | inf | PoseLib 0.5 | 是 | 全样本 | 同上 | 代码；分辨率与论文【不一致】 |
| RIPE++ | MegaDepth/SCARED 位姿 | 同上 | AUC@5/10/20° | inf | PoseLib 0.5 / 1.0 | 是（论文：全部 0.5） | 全样本 | **带 GT 的 val AUC@5 选 best** | 代码+论文 |
| SiLK | HPatches 单应 | 4 角点 (W−1,H−1) 平均距离，480 | HEA 与 AUC @1/2/3 | 误差 = 阈值+1 | OpenCV RANSAC，默认 3 px | 是（缓存预测走同一流程） | 全样本 | 合成单应 val 的 f1 | 代码 |
| RaCo | HPatches 检测器 | 4 角点平均距离，640 | AUC@1/3 | 未说明 | **GT 对应 + PoseLib DLT**（无 RANSAC） | 是（论文） | 未说明 | 未说明 | 论文（代码未找到） |
| CAPS | HPatches 单应 | 4 角点平均距离 | 精度@1/3/5 | 未说明 | 未说明 | 未说明 | 未说明 | 未说明 | 论文 |
| GIM | HPatches 单应 | 4 角点平均距离 | AUC@3/5/10 | 未说明 | OpenCV RANSAC | **否（基线数字抄原文）** | — | — | 论文 |
| XCP-Match | RoadScene 合成 H（可见光-红外） | **pose AUC**（平面近似） | @5/10/20 | 未说明 | RANSAC 3 | 未说明 | 未说明 | 未说明 | 论文 |
| S2M2-SAR | SEN1-2 / QXS 模板平移 | 预测位置与 GT 的欧氏距离 | CMR@1/5；RMSE(All)；RMSE(≤5) | 总有输出；RMSE(≤5) 剔除失败 | 无（热图 argmax） | — | 按样本平均 | 带 GT 的 val 监控（是否选模未说明） | 论文 |
| CA-Unsup | 自建视频对，人工 6 点 | 点 L2，**取双向较小值** | 均值；3px 内点率（代码未找到） | 无失败分支 | 网络直接回归；基线 RANSAC/MAGSAC | 否 | 每对 6 点均值，再按类别求均值 | 无 val（每 4k 步存一次） | 代码 |
| GeoFormer | HPatches / ISC-HE | 4 角点 (w−1,h−1) 平均距离，480 | Correct 与 AUC @1/3/5/10 | NaN（等价失败） | cv2.RANSAC 3（入口）/ degensac 2（函数默认） | 是（immatch 统一包装） | 全样本；另分 i/v | 合成 val_loss | 代码；README 与论文【不一致】 |
| GeoFormer | FIRE 控制点 | 控制点平均距离 | AUC@1..25 离散；Failed / Inaccurate(MAE>50 或 MEE>20) / Acceptable | **1e6**，计为 Failed | cv2.RANSAC 15 | 是 | **按组求 AUC 再平均（mAUC）** | 同上 | 代码 |

---

## 13. 源码清单

| repo URL | commit | 本地路径 | 评测入口文件 |
|---|---|---|---|
| https://github.com/fraunhoferhhi/RIPE | b173418008f4cb77a2ebffb570a7cab69e87cd08 | D:\Code\refs\RIPE | 训练验证：`ripe/benchmarks/imw_2020.py`；论文表格：见 glue-factory fork |
| https://github.com/fraunhoferhhi/RIPEpp | 0666e62bf00569870bbcc1fe4b2e03976fc563f8 | D:\Code\refs\RIPEpp | `ripepp/benchmarks/imw_2020.py`、`ripepp/benchmarks/scared.py`、`ripepp/train.py`（best 选择） |
| https://github.com/JohannesK14/glue-factory | 192baa367afc800614b224093b733433e012592b | D:\Code\refs\glue-factory-JohannesK14 | `gluefactory/eval/hpatches.py`、`gluefactory/eval/megadepth1500.py`、`gluefactory/eval/utils.py`；配置 `gluefactory/configs/ripe+NN.yaml`、`ripepp+NN.yaml` |
| https://github.com/cvg/glue-factory | 2d17e3b3bd7d30f0c828d4c4d3eac4ecefbf283d | D:\Code\refs\glue-factory | 仅用于确认其中没有 RaCo 评测配置 |
| https://github.com/facebookresearch/silk | 7b9614b4a66361a0003aaa6fe9298ccdb267714b | D:\Code\refs\silk | `etc/mode/run-hpatches-tests-silk.yaml` → `lib/metrics/hpatches_metrics.py`、`lib/matching/mnn.py` |
| https://github.com/cvg/RaCo | 35790eb48074ed14839d0fb496b8806caa4e766b | D:\Code\refs\RaCo | 无评测代码（README 声明将发布到 glue-factory） |
| https://github.com/qianqianwang68/caps | 1cb601a2b77f505c4ad1bcc172568badffa9b86b | D:\Code\refs\caps | 只有 `extract_features.py`（HPatches 特征）和 `test/eval_pose_megadepth.py`；无单应评测 |
| https://github.com/xuelunshen/gim | f09105a0555eef5c93db032d97b9e3d8a4cccb20 | D:\Code\refs\gim | `test.py`（ZEB）；无 HPatches 评测 |
| https://github.com/JirongZhang/DeepHomography | 3e811b7d06f84de34eae056baaf90ca886ae3908 | D:\Code\refs\DeepHomography | `Oneline-DLTv1/test.py` |
| https://github.com/ruc-aimc-lab/GeoFormer | 8b9506e6e9c0e61848724955fb514824039b1ff7 | D:\Code\refs\GeoFormer | `eval_Hpatches.py`、`eval_FIRE.py`、`eval_ISC.py` → `eval_tool/immatch/utils/{hpatches,fire}_helper.py` |
| S2M2-SAR | — | — | 未找到源码 |
| XCP-Match | — | — | 未找到源码 |
| https://github.com/mogvision/hvit | 未克隆 | — | 不在细看范围（3D 形变配准） |

---

## 14. 未核实 / 待办

- poselib `estimate_homography` 在匹配少于 4 个时返回 None 还是退化矩阵，决定了 glue-factory 里的"失败"到底是 `inf` 还是一个很大的有限误差。需要读 poselib 源码或实际运行确认。
- RIPE++ HPatches 配置中的 `topper: 1024` 是否生效，需要运行确认。
- RaCo 的评测代码尚未发布。CAPS 和 GIM 的 HPatches 单应评测代码在官方仓库里都没有。
