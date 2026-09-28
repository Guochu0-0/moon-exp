# 底座微调代码

「AnyMatch-LoFTR 微调代码接入与实测」(#25) 的产物。底座选定见 [`docs/design/rl-modeling.md`](../docs/design/rl-modeling.md)。

## 接入方式

- **不用 Lightning 训练框架**。AnyMatch 的 `third_party/LoFTR_AnyMatch` 与 MINIMA 的训练代码是上游 LoFTR（`df7ca80`）的模型 + `LoFTRLoss` 再套 Lightning；模型文件与上游逐字相同，差别只在配置、数据加载器、画图。那套损失和数据加载器都依赖深度 + 位姿 GT，我们用不上。所以直接复用 baseline 的 `third_party/LoFTR` 与 `LoFTRAdapter` 构造网络，自己写训练循环。服务器的 loftr 环境也没装 Lightning。
- **训练时模块保持 `eval()`**，只打开梯度。原因见 `model.py` 的 docstring：训练态粗匹配会去读 GT，BatchNorm 在小 batch 下也会跑偏。
- **ckpt 格式** `{"state_dict": ...}`，baseline 适配器能直接读，所以评测走原链路：
  `baselines.match <cfg> --weights <ckpt> --name <新名字>` → `baselines.fit` → `workbench eval`。
- 统一的仿射 RANSAC 在 `baselines/ransac.py`，训练侧和评测侧共用同一份实现。

## 组件

| 模块 | 作用 |
|---|---|
| `model.py` `Base` | 按 `configs/baselines/anymatch_loftr.json` 构造底座；`forward` 返回带梯度的 `conf_matrix`、`expec_f`；`state_dict()` 存成适配器能读的格式 |
| `data.py` `PairSet` | 一个 split 的全部 patch 对（Train 无标注，7907 对），输入映射同 baseline 主表 |
| `pseudo.py` | 伪标签损失：RANSAC 仿射当几何伪真值，按上游 LoFTR 原始监督形式出粗级正样本（sparse focal）与细级窗口内偏移（l2_with_std）（#26） |
| `label.py` | SCENES 式离线伪标签：起点模型在 Train 上估一次伪仿射，筛掉匹配 < 100 或内点 < 20 的对 |
| `rl.py` | 细级 RL：以 soft-argmax 为均值的高斯策略（逐匹配噪声 + 共享整体平移），整对 reward（梯度 NCC / CFOG）与逐匹配残差 reward，组内 baseline |
| `coarse.py` | 粗级闭式期望 −Σ P·r（RIPE++ 匹配器写法，r = 当前模型 RANSAC 内点 ±1）、负样本对、塌缩监控量（#49，`runs/C`） |
| `negpairs.py` | 负样本对（光学与 SAR 来自不同 ROI）上的恒等匹配诊断，也用作逐 ckpt 监控 |
| `train.py` | 训练循环：伪标签项（`--labels` 离线；不给则在线重估，会塌缩，见 `runs/S2`）+ RL 项（`--rl-pair`、`--rl-match`）+ L2-SP |

实测（显存、吞吐、RANSAC 耗时）与「lr=0 走训练循环后推理与 zero-shot 逐点一致」的验证记在 #25 的解答里；
当时用的一次性脚本不在库中。

## 训练（154 / 126，loftr 环境）

一键跑（打标 → 训练 → 逐 ckpt 在 Val 上评测）：`GPU=<空卡> scripts/finetune/scenes.sh <name> [train 参数]`，见脚本头注释。例：

```bash
# SCENES 式离线伪标签（S1）
GPU=0 scripts/finetune/scenes.sh S1 --labels $MOON_RESULTS/finetune/labels_b0.jsonl --steps 8000 --save-every 1000
# 从 S1 出发，伪标签项 + 整对 RL
GPU=0 scripts/finetune/scenes.sh R1 --init $MOON_RESULTS/finetune/S1/ckpt_6000.pt \
    --labels $MOON_RESULTS/finetune/labels_b0.jsonl --steps 4000 --save-every 1000 --rl-pair 1
```

粗级闭式期望的塌缩测试（#49）用 Python 队列 `scripts/finetune/ripe49.py`：训练 → 逐 ckpt Val sweep → 负样本对监控 → 塌缩汇总，
任务清单见 `scripts/finetune/jobs/ripe49.txt`。用法：`GPU=<空卡> /opt/envs/loftr/bin/python scripts/finetune/ripe49.py <jobs.txt>`。

## 评测微调后的权重（154 / 126，loftr 环境）

```bash
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights MOON_RESULTS=/remote-home/xufang/YGC/results
CUDA_VISIBLE_DEVICES=<空卡> nice -n 10 /opt/envs/loftr/bin/python -m baselines.match configs/baselines/anymatch_loftr.json \
    --split val --weights <ckpt.pt> --name <run 名> --out $MOON_RESULTS/finetune
/opt/envs/wb/bin/python -m baselines.fit $MOON_RESULTS/finetune/<run 名> --split val --run runs/<id>
```
