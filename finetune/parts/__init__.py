"""训练成分：每种训练信号一个文件，只带自己的参数。配置里出现同名的表就用这个成分（finetune/config.py）。

每个成分模块提供：
- PARAMS：各模型通用的参数与默认值；MODELS：支持的模型 → 该模型专有的参数与默认值；
- fill(p)：展开时补算的值（如「默认同某参数」）；check(cfg)：与其他成分、模型的组合是否合法，不合法抛 ConfigError；
- data_kw(p)：传给 PairSet 的参数（负样本对、扰动）；
- Term(p, model, ds)：只改数据的成分为 None。调用 term(step) → (加权后的损失, 监控量, 是否参与反向)。

登记顺序就是每步计算损失的顺序。
"""
from . import aug, cexp, neg, pseudo

PARTS = {"pseudo": pseudo, "cexp": cexp, "neg": neg, "aug": aug}
