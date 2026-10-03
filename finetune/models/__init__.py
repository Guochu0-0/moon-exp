"""模型适配器：把一个底座网络包装成训练循环能用的样子（finetune/train.py）。

每个适配器模块提供：
- PARAMS：配置 [model] 表里本模型的键与默认值（config 指向 configs/baselines/ 下的推理配置，评测也用它）；
- OPTIM / RUN：本模型对全局默认的覆盖（finetune/config.py）；
- Model(cfg, weights_root, device)：.resize（原网格 → 网络输入）、.forward、.params（参与训练的参数）、
  .state_dict()（baselines 适配器能直接读的 ckpt）、.amp（是否用 GradScaler）、.s（原网格 / 输入网格）。
"""
from . import loftr, roma

MODELS = {"loftr": loftr, "roma": roma}
