# 出处（#48）

检索和阅读日期：2026-09-28。行号对应下列版本。

## 论文（arXiv 源码包 `https://arxiv.org/e-print/<id>`，必要时对照 PDF）

| 文献 | 链接 | 读了什么 |
|---|---|---|
| RIPE++：Künzel, Eisert, Hilsmann, LIMIT@ECCV 2026 | [arXiv:2608.19693](https://arxiv.org/abs/2608.19693) | `sec/3_method.tex` 全文（reward 式 (3)、熵正则式 (4)、匹配器式 (8)–(10)、ℒ_nm）；`sec/4_experiments.tex`（实现细节 :4-12，SCARED 增强 :68，LightGlue 两阶段 :88-91，消融 :95-103）；`sec/X_suppl.tex`（distance reward、curriculum、表 6 评述 :108-113，负样本 :115-120，DISK↔LightGlue 推导）；`rsc/tables/*.tex`；PDF 经 pdftotext 核对表号（表 6 = ablations_supplementary） |
| RIPE：Künzel, Hilsmann, Eisert, ICCV 2025 | [arXiv:2507.04839](https://arxiv.org/abs/2507.04839) | `sec/3_method.tex`（负样本对 reward :143-165，描述子损失 :179-188，ℒ_low :197-201）；`sec/4_experiments.tex`（ImageNet VGG :6，ε 线性爬升 :14，超参表 :26）；`sec/X_suppl.tex`（ψ/ε 消融，「Towards collapsing to the epipoles」:72-77）；`tables/desc_loss.tex`、`tables/kp_penalty.tex` |
| DISK：Tyszkiewicz, Fua, Trulls, NeurIPS 2020 | [arXiv:2006.13566](https://arxiv.org/abs/2006.13566) | `tex/4-method.tex:36-46`（reward 用 GT 深度，闭式期望）；`tex/5-experiments.tex:10-22`（无增强，λ_tp/λ_fp/λ_kp 与退火，按验证 mAA 选模） |
| SCENES：Kloepfer et al., 2024 | [arXiv:2401.10886](https://arxiv.org/abs/2401.10886) | `sec/3_method.tex:148-171`（bootstrapping）；`sec/4_experiments.tex:109-115`（离线估 F、1:1 混入 MegaDepth GT 对） |
| Learning Cross-view Visual Geo-localization without Ground Truth | [arXiv:2403.12702](https://arxiv.org/abs/2403.12702) | 仅读摘要（arXiv API）：冻结 FM + adapter、EM 伪标签、重建损失 |
| SiLK | [arXiv:2304.06194](https://arxiv.org/abs/2304.06194) | 仅读摘要；合成单应训练的归类另见 RIPE §2 Related Work |

## 代码

| 仓库 | 版本 | 读了什么 |
|---|---|---|
| [fraunhoferhhi/RIPEpp](https://github.com/fraunhoferhhi/RIPEpp) | `0666e62`（2026-09-02） | `conf/train_default.yaml`（lr、fp/kp_penalty、batch、train_mode），`conf/heatmap_entropy_scheduler/*.yaml`，`conf/outlier_penalty_scheduler/constant.yaml`，`conf/inl_th/constant.yaml`，`conf/descriptor_loss/encoder_hard_net.yaml`（weight 2.5），`conf/data/disk_megadepth.yaml`，`conf/backbones/vgg.yaml`；`ripepp/train.py:385-690`（熵项、rl_loss、kp_penalty、描述子损失合成）；`ripepp/losses/rl_loss.py`；`ripepp/losses/hardnet_loss.py`；`ripepp/utils/utils.py:277-370`（get_rewards）；`ripepp/geometric_matching/matching_pipeline.py` |
| [JohannesK14/glue-factory](https://github.com/JohannesK14/glue-factory)（RIPE++ 匹配器训练） | `192baa3`（2026-09-02） | `README.md:62-96`（两阶段训练）；`gluefactory/configs/ripepp+lightglue_homography.yaml`（合成单应 + homography_matcher 真值）；`gluefactory/configs/ripepp+lightglue_megadepth_RL.yaml`（:18-19 缓存特征，:62-75 reward/penalty/阈值，:91-93 按测试集选模，:100 lr）；`gluefactory/models/utils/losses.py:129-362`（RLLoss）；`gluefactory/models/matchers/rl_lightglue.py:387-438`；`gluefactory/models/matchers/lightglue.py:123-200`（SelfBlock 用旋转编码，CrossBlock 不带位置）；`gluefactory/models/two_view_pipeline.py:22-26`（extractor.trainable False） |
| [fraunhoferhhi/RIPE](https://github.com/fraunhoferhhi/RIPE) | `b173418` | `conf/train.yaml`（desc_loss_weight 5.0，kp_penalty −7e-7） |
| [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR) | `df7ca80`（AnyMatch 推理代码与之相同，见 `configs/baselines/anymatch_loftr.json`） | `src/loftr/loftr.py:58-64`（两图加同一 PositionEncodingSine 后进入 coarse transformer）；`src/loftr/utils/position_encoding.py` |

## 仓库内

- `scenes-baseline` 分支（草稿 PR #33）：`finetune/pseudo.py`、`finetune/train.py`、`finetune/model.py`、`finetune/data.py`；`runs/S2/{exp.toml,notes.md,metrics.json,extra/train_collapse.json}`；`docs/design/rl-modeling.md`（3.6、3.9、3.10、4、7 节）。
- main：`CONTEXT.md`；`docs/research/label-free-reward/README.md`（AltO、SSHNet 的恒等塌缩，转引）；`docs/research/matcher-rl-feasibility/`；`docs/research/rl-registration-survey.md`。
- Issue #26（S1/S2 结果与更正）、#49、#52（截至 2026-09-28 无评论）。
