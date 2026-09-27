# 底座微调代码

「AnyMatch-LoFTR 微调代码接入与实测」(#25) 的产物。底座选定见 [`docs/design/rl-modeling.md`](../docs/design/rl-modeling.md)。

## 接入方式

- **不用 Lightning 训练框架**。AnyMatch 的 `third_party/LoFTR_AnyMatch` 与 MINIMA 的训练代码是上游 LoFTR（`df7ca80`）的模型 + `LoFTRLoss` 再套 Lightning；模型文件与上游逐字相同，差别只在配置、数据加载器、画图。那套损失和数据加载器都依赖深度 + 位姿 GT，我们用不上。所以直接复用 baseline 的 `third_party/LoFTR` 与 `LoFTRAdapter` 构造网络，自己写训练循环。服务器的 loftr 环境也没装 Lightning。
- **训练时模块保持 `eval()`**，只打开梯度。原因见 `model.py` 的 docstring：训练态粗匹配会去读 GT，BatchNorm 在小 batch 下也会跑偏。
- **ckpt 格式** `{"state_dict": ...}`，baseline 适配器能直接读，所以评测走原链路：
  `baselines.match <cfg> --weights <ckpt> --name <新名字>` → `baselines.fit` → `workbench eval`。
- 统一的仿射 RANSAC 在 `baselines/ransac.py`，训练侧和评测侧共用同一份实现。

## 用法（154 / 126，loftr 环境）

```bash
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights MOON_RESULTS=/remote-home/xufang/YGC/results
# 最小训练循环 + 实测
CUDA_VISIBLE_DEVICES=<空卡> nice -n 10 /opt/envs/loftr/bin/python -m finetune.smoke configs/baselines/anymatch_loftr.json \
    --split train --steps 30 --batch 1 --out $MOON_RESULTS/finetune/smoke
# 验证接入不改行为：lr=0 走一遍训练循环，存 ckpt，再按 baseline 流程在 Val 上推理，与 zero-shot 原始点对比较
... -m finetune.smoke ... --lr 0 --save --out $MOON_RESULTS/finetune/lr0
... -m baselines.match configs/baselines/anymatch_loftr.json --split val --weights $MOON_RESULTS/finetune/lr0/ckpt.pt \
    --name anymatch_loftr_lr0 --out $MOON_RESULTS/finetune/verify
... -m finetune.compare $MOON_RESULTS/baselines/anymatch_loftr/val.npz $MOON_RESULTS/finetune/verify/anymatch_loftr_lr0/val.npz
```
