# 来源

读取日期：2026-10-27。论文正文大多读 arXiv HTML 版；PPO-EWMA 没有 HTML 版，读的是 arXiv PDF（v3）转出的文本。代码通过 GitHub API 读取默认分支。

| 工作 | 链接 | 读了哪些部分 |
|---|---|---|
| Co-rewarding：Zhang, Zhu, Ge, Zhao, Zhou, Li, Feng, Yao, Han，ICLR 2026 | [arXiv:2508.00410](https://arxiv.org/abs/2508.00410)（HTML v3） | §1；§3.2（式 6–12）；§4.1（α 起止值）；§4.2.1–4.2.2（图 3–6、表 1–3）；附录 A（算法 2、3）；附录 B.1（式 13、14）；附录 C.2 表 5；附录 D.2（不更新教师时伪标签准确率约 25%） |
| Co-rewarding 官方代码 | [tmlr-group/Co-rewarding](https://github.com/tmlr-group/Co-rewarding) | `Co-rewarding-II/run_corewarding-II.sh`（alpha_start=0.01、alpha_end=1e-5、interval=1、kl_loss_coef=0.001）；`Co-rewarding-II/verl/workers/sharding_manager/fsdp_vllm.py`（`update_alpha` 余弦日程、`_update_ref_model_sliding_average`）；`Co-rewarding-II/verl/workers/fsdp_workers.py`（`generate_ref_sequences`；ref 模块同时用于 KL 对数概率与教师采样） |
| M-GRPO：Bai, Wu, Ye, Chen，NeurIPS 2025 Workshop on Efficient Reasoning | [NeurIPS 页面](https://neurips.cc/virtual/2025/126709)；[arXiv:2512.13070](https://arxiv.org/abs/2512.13070)（HTML v1）；[OpenReview](https://openreview.net/forum?id=DgfYSvw989) | NeurIPS 页面（研讨会海报）；§1；§2.1（式 1–3）；§2.2（IQR，k=0.75）；§3；表 1–4；图 1、2、5 的文字说明 |
| M-GRPO 代码 | [baibizhe/M_GRPO](https://github.com/baibizhe/M_GRPO) | `verl-intuitor/verl/workers/fsdp_workers.py`（`update_momentum_rollout`，在 `update_actor` 里以 m=0.99 硬编码调用）；`verl-intuitor/verl/trainer/ppo/ray_trainer.py`（策略与动量回答合并） |
| EMA Policy Gradient：Zhang & Ba，ICML 2026 | [arXiv:2602.04417](https://arxiv.org/abs/2602.04417)（HTML v1）；[ICML 2026 日程页](https://icml.cc/virtual/2026/70431) | §1；§4（引理 4.1、4.2，式 9 稳定条件）；§5.1；§6（表 2、表 3，图 3、6–9 的文字说明）；附录 A（lr 1e-6、β=0.001、T_ema=10）；附录 B 表 4；附录 C.4–C.5 |
| EMA-PG 代码 | [LunjunZhang/ema-pg](https://github.com/LunjunZhang/ema-pg) | `README.md`；`math/examples/deepscaler/train_deepscaler_8k_ema_pg.sh`（ref_policy_ema_tau=0.95，update_period=10，kl_loss_coef=0.001） |
| WARP：Ramé et al.（Google DeepMind），2024 | [arXiv:2406.16768](https://arxiv.org/abs/2406.16768)（HTML v1） | §1；§3.1（EMA 锚点，μ=0.01，观察 1）；§3.2–3.4；§4 设置（β=0.1）；§4.1 图 3；§4.4 表 1；§6；附录 D.1、D.2（μ=0.005、β=0.2 的对比结论） |
| PPO-EWMA：Hilton, Cobbe, Schulman，「Batch size-invariance for policy optimization」，NeurIPS 2022 | [arXiv:2110.00641](https://arxiv.org/abs/2110.00641)（PDF v3）；[openai/ppo-ewma](https://github.com/openai/ppo-ewma)（未读代码） | §1；§2（分开的 KL 与 clip 目标）；§4；§5.1–5.3（图 1–4 的文字说明）；附录 A（伪代码与带归一化的 EWMA）；附录 B（表 1–4，β_prox=0.889，重心 8，试过 2 和 32）；附录 G（重心年龄与 KL 系数互换，图 18） |
| 前一份调研 | 分支 `research/self-ref-collapse`：`docs/research/self-ref-collapse/README.md` | 第 0、2 节（Mean Teacher、BYOL、DINO、SRT、clip 空操作）；其 `sources.md` |
| 本库记录 | `runs/E30/notes.md`、`runs/E32/notes.md`（main 分支已提交版本） | E30 表 1（与慢速参照的距离）；E32 摘要、方法、表 2 |

未读原文、只在检索结果中看到：arXiv 2607.02460（Neuron-OPSD，EMA 教师用于无标注自蒸馏）。
