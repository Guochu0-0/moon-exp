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

实测（显存、吞吐、RANSAC 耗时）与「lr=0 走训练循环后推理与 zero-shot 逐点一致」的验证记在 #25 的解答里；
当时用的一次性脚本不在库中。

## 评测微调后的权重（154 / 126，loftr 环境）

```bash
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights MOON_RESULTS=/remote-home/xufang/YGC/results
CUDA_VISIBLE_DEVICES=<空卡> nice -n 10 /opt/envs/loftr/bin/python -m baselines.match configs/baselines/anymatch_loftr.json \
    --split val --weights <ckpt.pt> --name <run 名> --out $MOON_RESULTS/finetune
/opt/envs/wb/bin/python -m baselines.fit $MOON_RESULTS/finetune/<run 名> --split val --run runs/<id>
```
