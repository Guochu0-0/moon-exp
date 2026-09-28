# 来源清单（matcher-rl-feasibility）

读取日期：2026-09-27。代码都是按下列 commit 从 raw.githubusercontent.com / HF 取的文件，行号对应这些 commit。

## 代码

| 仓库 | commit | 读过的文件 |
|---|---|---|
| [MnYangs/AnyMatch](https://github.com/MnYangs/AnyMatch) | `259ad343ffa9ca0703965f56880b3b5d2ecbc9e1`（与本库 submodule 相同） | `README.md`（TODO 列表）、`weights/readme.md`；`third_party/LoFTR_AnyMatch/`：`src/loftr/utils/coarse_matching.py`、`src/loftr/utils/fine_matching.py`、`src/loftr/utils/supervision.py`、`src/losses/loftr_loss.py`、`src/config/default.py`、`configs/loftr/outdoor/loftr_ds_dense.py`、`src/datasets/megadepth.py`、`train.py`、`scripts/reproduce_train/outdoor_ds.sh`；`third_party/RoMa_AnyMatch/`：`train_roma_outdoor.py`、`romatch/losses/robust_loss.py`、`romatch/models/matcher.py`、`romatch/datasets/megadepth.py`（与 RoMa_minima 逐文件 diff：只有 `train_roma_outdoor.py:268` 一行不同） |
| [LSXI7/MINIMA](https://github.com/LSXI7/MINIMA) | `796e7721174f9f829b79b3702bf8c2ae9a3d447a` | `README.md`、`.gitmodules`、`train_orders/README.md`、`train_orders/minima_{roma,loftr}.sh`、`train_orders/minima_{roma,loftr}_train_config.yaml` |
| [LSXI7/RoMa_minima](https://github.com/LSXI7/RoMa_minima) | `1b807e62b4228f2f3664c348d296c1e1748abfa9`（本库 submodule 固定的是 `0d3fd22`，更旧） | `train_roma_outdoor.py`、`romatch/losses/robust_loss.py`、`romatch/models/matcher.py`（`Decoder.forward` 333-423，`sample` 468-495，`match` 612-748）、`romatch/utils/utils.py`（`cls_to_flow_refine` 302-323，`get_gt_warp` 326-355）、`romatch/utils/kde.py`、`romatch/train/train.py`、`romatch/models/model_zoo/roma_models.py`、`romatch/datasets/megadepth.py` |
| [Parskatt/RoMa](https://github.com/Parskatt/RoMa)（上游） | `77f8d68803526dcddfd9b7a46bc76125bdc25f15` | `romatch/models/model_zoo/__init__.py`（默认 coarse_res 560、upsample_res 864） |
| [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)（上游） | `df7ca80f917334b94cfbe32cc2901e09a80e70a8` | `docs/TRAINING.md`（GPU 需求、稠密监督说明） |
| [zju3dv/MatchAnything](https://github.com/zju3dv/MatchAnything) | `8cd8c1129a6d22dabea9405a869e4fad6ff8b630`（2026-09-15） | `README.md`（仓库只有 README + LICENSE + gif；“The training code will be available later.”） |
| [HF Space LittleFrog/MatchAnything](https://huggingface.co/spaces/LittleFrog/MatchAnything) | `6a7bcb589ec8da3a9e861e799122beaa5eba2193`（与本库 submodule 相同） | 文件清单（无 train 入口）；`imcui/third_party/MatchAnything/third_party/ROMA/roma/matchanything_roma_model.py`、`configs/models/roma_model.py`、`src/config/default.py`（ROMA 段）、`src/lightning/lightning_loftr.py` |

本库内部参照：`baselines/adapters/{romatch,matchanything,loftr}.py`、`configs/baselines/{anymatch_loftr,anymatch_roma,ma_roma,minima_roma}.json`（main `5340a4b`）。

## 论文

- LoFTR：Sun et al., CVPR 2021（损失与流水线以代码为准）。
- RoMa：Edstedt et al., CVPR 2024，[arXiv:2305.15404](https://arxiv.org/abs/2305.15404)（DINOv2 冻结、64×64 锚点、Charbonnier α=0.5、约 199 ms/对）。
- MINIMA：Ren et al., CVPR 2025，[arXiv:2412.19412](https://arxiv.org/abs/2412.19412)（MD-syn、4×3090、各模型 bs/epoch/lr）。
- AnyMatch：Yang et al., ECCV 2026，[arXiv:2606.31077](https://arxiv.org/abs/2606.31077)（Any-syn 50 万对、4×4090、官方权重初始化、RoMa 4 epoch）。README 徽章上的 2605.04730 是另一篇论文（ULF-Loc），已核对。
- MatchAnything：He et al., TPAMI 2026，[arXiv:2501.07556](https://arxiv.org/abs/2501.07556) v1（多源数据引擎、约 8 亿对、16×A100、SAR 未进训练集）。
- RIPE++：Künzel, Eisert, Hilsmann，[arXiv:2608.19693](https://arxiv.org/abs/2608.19693)（2026-08-20；用 RL 训 LightGlue，闭式期望 reward；代码 https://github.com/fraunhoferhhi/RIPEpp，未读代码）。
- WarpC：Truong et al., ICCV 2021，[arXiv:2104.03308](https://arxiv.org/abs/2104.03308)（只读了摘要）。
- Yi et al., [arXiv:2607.10082](https://arxiv.org/abs/2607.10082)（事件-图像无标签目标域自蒸馏，只读了摘要）。

论文内容通过 arXiv HTML 版读取。
