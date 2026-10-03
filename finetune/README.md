# 微调训练代码

结构：**公共训练循环 + 模型适配器 + 可组合的训练成分**，每个方法由一份配置文件描述（「训练代码重构」#86）。
优先级：可追溯 > 可读 > 好扩展。

## 接入方式

- **不用 Lightning 训练框架**。AnyMatch 与 MINIMA 的训练代码是上游模型 + 依赖深度 / 位姿 GT 的损失和数据加载器，我们用不上；直接复用 baseline 的适配器构造网络，自己写训练循环（#25、#64）。
- **训练的网络就是评测的网络**：模型适配器复用 `baselines/adapters/`，ckpt 存成适配器能直接读的格式，评测走原链路：
  `baselines.match <推理配置> --weights <ckpt> --name <名字>` → `baselines.fit` → `workbench eval`。
- 原始影像到网络输入的映射和统一的仿射 RANSAC 在 `moonlib/`，baseline 推理与训练共用。

## 模块

| 模块 | 作用 |
|---|---|
| `train.py` | 公共训练循环：读配置 → 建模型、数据、各成分 → 每步按登记顺序求各成分损失、求和、反向 → 写 `log.jsonl`、ckpt、`args.json` |
| `config.py` | 读 `runs/<id>/configs/<方法>.toml`、同实验内 `base` 继承、展开默认值、校验（未知的表或键直接报错） |
| `optim.py` | 优化配方：AdamW、warmup + const / cosine、梯度累积、裁剪；RoMa 用 GradScaler |
| `models/loftr.py` | AnyMatch-LoFTR：模块保持 `eval()`、只开梯度（原因见 docstring） |
| `models/roma.py` | AnyMatch-RoMa（minmax 输入）：只训 decoder（可解冻 VGG），BN 保持 eval |
| `parts/pseudo.py` | `[pseudo]` 离线伪标签监督：LoFTR 按上游监督形式（粗级 focal + 细级 l2_with_std），RoMa 按 RoMa 原损失，GT 换成伪仿射 |
| `parts/cexp.py` | `[cexp]` 粗级 ±1 期望：当前模型 RANSAC 内点 +1、外点 `r_out`；`placebo` 为随机分数对照 |
| `parts/neg.py` | `[neg]` 负样本对：SAR 换成其他 ROI 的，并入同一次前向，由 `[cexp]` 给分 |
| `parts/aug.py` | `[aug]` 已知随机扰动：SAR 已知仿射（标签随之变换）、光度扰动；实现在 `augment.py` |
| `data.py` | `PairSet`：一个 split 的全部 patch 对，可带负样本对与扰动 |
| `geom.py` | 坐标约定与仿射小工具 |
| `label.py`、`label_from_raw.py`、`refine_labels.py` | 打离线伪标签（模型推理 / 已有点对）、标签的整体平移修正 |
| `negpairs.py` | 负样本对上的恒等匹配诊断，驱动用它逐 ckpt 监控 |

没有搬过来的训练信号（需要时从 git 历史找回，最后一版在本重构之前的 main，commit `8de0e03`）：在线伪标签（含细级以 RANSAC 仿射为目标的 l2）、细级逐匹配 RL、细级共同平移 RL、L2-SP、人为注入偏移、只训细级模块。

## 配置

每个方法一份 `runs/<id>/configs/<方法>.toml`。**出现哪个成分的表就用哪个成分**，表里只写与默认不同的参数；读配置就知道这个方法由哪些成分组成。

```toml
# runs/E7/configs/Q4a.toml
[model]
name = "loftr"          # loftr | roma

[optim]
steps = 4000

[run]
save_every = 500

[cexp]
r_out = -0.25

[neg]
```

```toml
# runs/E7/configs/Q4a_photo.toml：同一实验内继承，只写差别
base = "Q4a"

[aug]
geo = false
```

- 全部键与默认值：`[model]` 见 `models/*.py` 的 `PARAMS`，`[optim]`、`[run]` 见 `config.py` 的 `OPTIM`、`RUN`（模型可覆盖，如 RoMa 的 `wd = 0.01`），各成分见 `parts/*.py` 的 `PARAMS`、`MODELS`。
- `base` 只能写同一实验 `configs/` 下另一个配置的名字；文件名以 `_` 开头的只给别人当 base，驱动不单独跑它。
- 新行为只能通过新增参数开启，默认值保持旧行为。修 bug 例外，但在 commit 和受影响实验的 notes 里写明。
- `configs/finetune/` 里是旧方法（S1、Q4、P8、M4、M7）按新格式的改写，用于核对和当模板，不是实验记录。
- 展开配置看效果：`python -c "from finetune import config; import json; print(json.dumps(config.load('runs/E7/configs/Q4a.toml'), indent=1))"`。

加一个新的训练成分：在 `parts/` 下新建一个文件（`PARAMS`、`MODELS`、`fill`、`check`、`data_kw`、`Term`，见 `parts/__init__.py`），登记进 `PARTS`。加一个新底座：在 `models/` 下新建适配器并登记进 `MODELS`。

## 训练（154 / 126 / 150 / A6000，loftr 环境）

在本票的 worktree 里用驱动跑：`GPU=<空卡> /opt/envs/loftr/bin/python scripts/finetune/run.py runs/<id>`。它逐个领取 `configs/` 里的方法：训练 → 逐 ckpt 在 Val 上评测 → 按 Val 峰值补评 Test → 写成正式记录。流程见 `docs/agents/experiments.md`，产物布局见 `workbench/RECORDS.md`「训练产物」。

单独训练（调试）：`python -m finetune.train runs/<id>/configs/<方法>.toml --out <目录> [--stop-at N]`；`--stop-at` 提前停下，lr 调度仍按配置的总步数。

## 单独评测一个 ckpt（loftr 环境）

`run.py` 之外手工评测时，原始点对也放进实验目录（不进 git）：

```bash
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights
CUDA_VISIBLE_DEVICES=<空卡> nice -n 10 /opt/envs/loftr/bin/python -m baselines.match configs/baselines/anymatch_loftr.json \
    --split val --weights runs/<id>/ckpt/<m>/ckpt_<step>.pt --name <m> --out runs/<id>/sweep/<m>/match
/opt/envs/wb/bin/python -m baselines.fit runs/<id>/sweep/<m>/match/<m> --split val --run runs/<id>
```
