"""训练成分：每种训练信号一个文件，只带自己的参数。配置里出现同名的表就用这个成分（finetune/config.py）。

每个成分模块提供：
- PARAMS：各模型通用的参数与默认值；MODELS：支持的模型 → 该模型专有的参数与默认值；
- fill(p)：展开时补算的值（如「默认同某参数」）；check(cfg)：与其他成分、模型的组合是否合法，不合法抛 ConfigError；
- data_kw(p)：传给 PairSet 的参数（负样本对、扰动）；
- Term(p, model, ds)：只改数据的成分为 None。调用 term(step) → (加权后的损失, 监控量, 是否参与反向)。
  反向通常由训练循环对返回的损失统一做；显存放不下一次反向的成分（rloo 细化加噪，#192）可以在内部经 step.optim.backward
  分段反向，再在返回的损失里放一个替身项，把累积在中间特征上的梯度随统一反向传回编码器。

登记顺序就是每步计算损失的顺序。
"""
from . import aug, cexp, l2sp, neg, pseudo, rloo, warpc

PARTS = {"pseudo": pseudo, "cexp": cexp, "neg": neg, "aug": aug, "warpc": warpc, "l2sp": l2sp, "rloo": rloo}
