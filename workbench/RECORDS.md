# 实验记录规范（v2）

做实验的人和 agent 读这一份就够了：实验结果写成什么样、写到哪里、用什么写。代码放哪、怎么在服务器上启动和收尾见 `docs/agents/experiments.md`。工作台怎么展示这些记录见 `README.md`，做实验时不需要读。术语（实验、方法、点亮、Notes 等）见仓库根目录的 `CONTEXT.md`。

## 流程

1. 建目录：`python -m workbench new E3 --parent E2 [--init E2/main] --title "一句话标题"`。编号取下一个未用的 `E<n>`；用户已在画布上建好的直接用。
2. 跑实验（先提交代码，见 `docs/agents/experiments.md`）。微调时每个方法写一份 `configs/<方法>.toml`（格式见 `finetune/README.md`），用 `scripts/finetune/run.py runs/<id>` 跑，它按下文「训练产物」写好全部记录；自己写的用 `PredWriter` 写预测，需要时用 `InterWriter` 写中间结果、按下文约定写 TB 日志。产物全部放在 `runs/<id>/` 里，不要写到别处。
3. 按下文「notes.md」一节起草 `runs/<id>/notes.md`：问题、设置、结果、结论都写。
4. 可选：`python -m workbench eval <id>` 生成 `metrics.json`；`python -m workbench check` 检查记录。
5. 提交小文件。大文件由 `.gitignore` 排除，结票时由 `scripts/wt.sh close` 挪到 gpfs 主 checkout，用户需要时在本地 `python -m workbench sync <id>` 拉回。

## 目录

一个实验对应 `runs/<id>/` 一个目录。服务器上的相对路径和本地完全一样。影像不存，读数据集即可。

```
runs/<id>/
  exp.toml                             元数据（由 new 生成，或手写）
  notes.md                             Notes，可以没有
  metrics.json                         由 eval 生成，不要手改
  preds/<method>/<split>.jsonl         每个 pair 一行，存估计的仿射
  preds/<method>/<split>.meta.json     写入时的 commit、dirty、主机、时间
  preds/<method>/<split>_matches.npz   点对，不进 git
  inter/<method>/<name>/meta.json      中间结果的种类、坐标系、说明、单位
  inter/<method>/<name>/<split>.npz    中间结果数据（scalar / points / flow），不进 git
  inter/<method>/<name>/<split>/*.png  中间结果数据（image），不进 git
  tb/<method>/                         TensorBoard 日志，不进 git
  ckpt/<method>/                       权重与训练日志，不进 git，也不同步
  configs/<method>.toml                微调方法的配置（见「训练产物」）
  launch/<method>.json                 每次启动的 commit、主机、GPU、命令、起止时间、完整配置（自动写）
  sweep/<method>/                      逐 ckpt 评测（见「训练产物」）
  code/                                只服务本实验的代码：诊断与画图脚本、一次性命令
  extra/                               附件
```

- **点亮**是派生状态：任一方法有任一 split 的 `preds/<method>/<split>.jsonl` 就算点亮，不需要手填。
- 方法按 `preds/<method>/` 自动发现。普通实验只有一个方法，命名为 `main`；基线实验（如 B0）可以挂多个。变体方法命名为 `<方法>__<变体>`（如 `loftr__minmax`）。
- split 为 `train` / `val` / `test`。

## exp.toml

| 字段 | 含义 |
|---|---|
| `id` | 必填，与目录名一致 |
| `title` | 必填，一句话标题 |
| `parent` | 父实验 id；根节点整行省略 |
| `init` | 可选，写成 `<实验>/<方法>`（如 `B0/roma`），表示从父实验的哪个方法起步 |
| `baseline` | 可选，默认 `false`。基线实验（不训练、作比较起点）写 `true` |
| `date` | 创建时写入，之后不改 |
| `[methods.<m>]` | 可选，`name` 为显示名，`caveat` 为读数时要注意的事（如「点数是采样出来的」） |

不要写其他字段：`status`、`commit`、`hypothesis`、`change`、`verdict`、`next` 是 v1 的，已删除；commit 由 preds 的 meta 记录，其余内容写进 notes.md。

## notes.md

实验唯一的自由文本，也是这个实验的记录。Markdown，不加 front matter，UTF-8，LF 换行。

**读者与分工。** 读者是半年后的研究者本人和导师：不翻 issue、不读代码，只看节点页和 Notes，就能明白做了什么、结果如何、得出什么结论。严谨程度按论文来写。

- agent 按本节起草，结果和结论都写。
- 用户之后修改、补充。用户改过的文字 agent 不再改写；有新结果只在末尾追加，与原有结论冲突时在追加的部分里指出。分不清哪些是用户改过的，就当作改过。

**结构。** 一个实验回答一个问题（换了问题就另开子实验），Notes 按顺序分四节：

| 节 | 写什么 |
|---|---|
| `## 问题` | 这个实验要回答的问题；开跑前的预期也写在这里：预期什么结果、依据是什么 |
| `## 设置` | 方法是什么、和参考方法差在哪里、数据、训练、选模怎么做。修过影响本实验的 bug 也写在这里 |
| `## 结果` | 只放节点页上没有的内容（见下文「数字」） |
| `## 结论` | 对问题的回答，标明哪些是推断；还没回答的问题放在本节末尾 |

不写「下一步」，也不写工作日志：谁在何时如何做的决定不属于 Notes。

**语言。** 用最简单的通用语言。

- 不用本项目自造的词、内部代号（实验 id、ckpt 名等）、文件路径、命令行参数、issue 编号。提到别的实验时说它做了什么，例如「从第 6000 步接着训的那个实验」。
- 领域通用术语（AUC@5、RANSAC、LoFTR、focal loss 等）直接用；其他术语在第一次用到的地方解释。
- 引用论文写 arXiv 号或论文名。
- 附件图片用相对路径嵌入，例如 `![](extra/offset/offset_val.png)`，这是唯一可以出现的路径。

**数字。**

- 节点页已经展示的指标（各方法的 Val / Test AUC、SR、bootstrap CI、与参考方法的配对差等）不在 Notes 里重复。
- 「结果」只放页面没有的：逐 ckpt 曲线、多种子汇总、诊断、与参考方法以外的实验的对比。
- 「结论」可以引用它直接依赖的一两个关键数字。

**多方法。** 同一问题下的变体、种子、对照可以作为多个方法挂在一个实验里。「设置」里每个方法都要讲清楚是什么。Notes 里用方法的显示名（`exp.toml` 的 `[methods.<m>] name`）称呼它，显示名要能自解释；没有显示名就先补上。

**旧实验**（本规范之前的实验，一个实验常常回答了好几个问题）：不拆节点、不挪目录、不改名。Notes 开头列出它实际回答的几个问题，配一张按问题分组的「方法一览」表（显示名、是什么）；结果和结论按问题分小节。预期按当时的思路补写。

**示例**（按上述规范写的一个单问题实验，节选）：

```markdown
## 问题

把 zero-shot 匹配估出的仿射当作伪真值，按 LoFTR 原本的监督方式微调（SCENES，arXiv:2401.10886），能否在不用标注的情况下提高配准精度？

预期：能提高，但幅度有限。zero-shot 在 Val 上 10 px 内的成功率约 85%，多数伪真值大致正确，可以当监督；伪真值自身的误差也会被学进去，所以提升应该很快到顶。

## 设置

- 起点与参考方法：AnyMatch 发布的 LoFTR 多模态权重，不做任何训练。
- 伪真值：起点模型在训练集 7907 对上各推理一次，用 RANSAC（3 px）估仿射；匹配少于 100 个或内点少于 20 个的对丢弃，剩 7880 对。训练过程中伪真值不变。
- 损失：粗级以光学网格点经伪仿射落到的 SAR 网格为正样本，用 focal loss；细级回归窗口内的亚像素偏移，用 LoFTR 原本的细级损失。
- 优化：AdamW，学习率 1e-5 不变，batch 1，共 8000 步，每 1000 步存一次权重，取 Val AUC@5 最高的一步作为结果。
- 方法：「SCENES 式离线伪标签」为主方法；「换种子 1」「换种子 2」只换随机种子，其余相同。

## 结果

- 三个种子的 Val AUC@5 都在前 1000 步内升到 0.26 左右，之后在 0.26–0.27 之间波动，不再上升；峰值分别出现在第 6000、4000、5000 步。
- 从第 6000 步接着训，或把学习率提高到 3e-5（另两个实验），同样停在 0.26–0.27。

## 结论

能提高：Val AUC@5 从 0.220 升到 0.272，换种子结果一致。提升几乎全部在前 1000 步内完成，接着训或提高学习率都不再上涨，平台期更像是伪真值本身的上限（推断）。

还没回答：平台期是伪真值误差造成的，还是因为训练配方没调（没有 warmup、学习率衰减、梯度裁剪）？
```

## 预测：preds

```python
from workbench.records import PredWriter

with PredWriter("runs/E3", "main", "val") as w:
    for pair in pairs:                      # Dataset(root).pairs("val")
        w.write(pair, A, n_inliers=k, matches=M)   # A: 2×3；M: RANSAC 前的全部点，N×5，或 N×4 另给 conf=
        # 失败时：w.write(pair, None, fail="few_inliers", matches=M)
```

写出的 `<split>.jsonl` 每个 pair 一行：

```json
{"pair": "ROI_037/patch_1", "A": [[a, b, c], [d, e, f]], "n_inliers": 57, "sec": 0.12}
{"pair": "ROI_037/patch_2", "A": null, "fail": "few_inliers"}
```

- `pair` 写 `<ROI 目录名>/<文件名 stem>`，与 `Processed_Data/<Split>/<ROI>/{Optical,SAR,Label}/<stem>.*` 一一对应。
- `A` 是把**光学像素坐标 (x 向右, y 向下) 映射到 SAR 像素坐标**的 2×3 仿射，坐标在原始 512 px 网格上。方法内部 resize 过的，必须先映射回原始网格。
- 匹配类方法的仿射由 `cv2.estimateAffine2D`（RANSAC 3 px）在跑实验时算好再写。工作台只消费仿射，自己不估计。
- `A` 为 `None` 表示失败，原因写在 `fail` 里。有标注但没有预测行的 pair 也按失败计。
- 其余关键字参数随意，原样写进这一行。
- `matches` 取 matcher 的坐标约定（整数 = 像素中心），内容是 RANSAC 前的全部点。+0.5 换算、按 conf 截断到 2000 个点、写 npz 都由 `PredWriter` 负责；没有 conf 就不传。0 个点的 pair 照写；出错的 pair 不传 `matches`。
- `PredWriter` 在 close 时写 `<split>.meta.json`（`commit`、`dirty`），commit 不用手记。

## 中间结果：inter

想在工作台上逐 pair 查看的中间量（置信度图、位移场等）。按方法、名称、split 写，可以只覆盖部分 pair：

```python
from workbench.records import InterWriter

with InterWriter("runs/E3", "main", "certainty", "val", kind="scalar", frame="opt", desc="RoMa certainty") as w:
    for pair in pairs:
        w.write(pair, cert)
```

| `kind` | 每个 pair 的数据 |
|---|---|
| `scalar` | H×W |
| `points` | N×3 `(x, y, v)`，坐标是 frame 那张 patch 的原始像素坐标，取 matcher 的约定（+0.5 由写入端负责） |
| `flow` | H×W×2，frame 坐标系下每个像素指向另一模态中对应点的位移 `(dx, dy)`，单位是原始 patch 的 px |
| `image` | uint8 H×W[×3\|4]，原样显示 |

- `frame` 为 `opt` / `sar`：数据所在的坐标系。分辨率任意，显示时拉伸到 patch 大小。
- `desc` 一句话说明，`unit` 单位，可空。
- 形状不符时抛 ValueError。构造时清掉这个 split 的旧数据（重跑即整份替换）；with 块里抛异常时不留半份数据。
- `meta.json` 进 git，数据不进。

## TB 日志

- 自己写的训练用两个 `SummaryWriter`，分别写 `runs/<id>/tb/<method>/scalars/`（标量）和 `runs/<id>/tb/<method>/media/`（图片、直方图）。**必须显式传 `log_dir`**：不传时会写到 `./runs/<时间>_<host>`，和实验目录 `runs/` 撞名。
- 不要用 `add_scalars`（会建子目录）。
- 上游 Lightning 代码把 `save_dir` 指向 `runs/<id>/tb/<method>/`，日志落在 `tb/<method>/<name>/version_N/`；ckpt 放在 version 目录的 `checkpoints/` 下。
- 续训接着写到同一个目录即可，重叠的 step 由读取端处理。
- 服务器的训练环境没装 tensorboard，标量可用 `workbench.tbwriter.ScalarWriter`（纯标准库，读写格式与官方一致）；`finetune.train` 的 `--tb` 就是用它写的。

## 训练产物

微调实验的每个方法 `<m>` 由一份 `configs/<m>.toml` 描述（进 git，要先提交再跑）：用哪个模型、哪些训练成分、与默认不同的参数；同一实验内可用 `base` 继承，文件名以 `_` 开头的只当 base。格式见 `finetune/README.md`。

`scripts/finetune/run.py runs/<id>` 逐个领取这些方法，对每个方法 `<m>` 依次做：训练 → 每个 ckpt 在 Val 上评测 → 按 Val AUC@5 峰值（不含 step 0）选 ckpt，在 Test 上补评 → 把选中 step 的 Val、Test 预测复制成 `preds/<m>/`，即这个方法的正式结果。

```
runs/<id>/
  configs/<m>.toml     方法的配置（进 git，手写）
  ckpt/<m>/            ckpt_<step>.pt、log.jsonl、args.json、train.log、sweep.log（不进 git）
  tb/<m>/scalars/      训练曲线（不进 git）
  launch/<m>.json      启动记录（进 git）：commit、主机、GPU、命令、起止时间、结果、入口、配置文件、展开默认值后的完整配置
  sweep/<m>/S/         逐 ckpt 的工作台记录，方法名 step<N>；exp.toml、metrics.json 进 git，preds/ 不进
  sweep/<m>/match/     逐 ckpt 的原始点对（不进 git）
  sweep/<m>/peak.json  选中的 step 与 Val / Test 指标（进 git）
  sweep/<m>/neg/、collapse.json   负样本对监控（LoFTR 的在线信号任务；neg/ 不进 git）
  preds/<m>/           选中 step 的预测
  .claims/<m>/         队列占位（不进 git）
```

sweep 里的每个 step 不是独立的方法：只有 `preds/<m>/` 出现在工作台上。想换选模规则，就在 Notes 里写明，再把对应 step 的预测复制过去。

## 启动记录

`launch/<m>.json` 是一个列表，每次启动（含续训、重跑）一条，由 `workbench.launch` 在启动时自动写：`start`、`end`、`status`、`commit`、`dirty`、`allow_dirty`、`host`、`cwd`、`cuda_visible_devices`、`cuda_device_order`、`cmd`。不要手写。

微调驱动 `scripts/finetune/run.py` 另记：`entry`（入口）、`config_file`（方法的配置文件 `configs/<m>.toml`）、`args`（展开所有默认值后的完整配置，按表嵌套：`model`、`optim`、`run` 和用到的训练成分）、`config`（评测用的推理配置，`configs/baselines/*.json` 的内容）。字段名与下面事后补记的一致，节点页同样显示。

2026-10-03 以前的旧方法没有自动记录，由 `scripts/backfill_launch85.py` 事后补记，节点页同样显示。补记的每条另有：

| 字段 | 含义 |
|---|---|
| `backfilled` | 补记日期；有这个字段就是事后补记 |
| `reliability` | commit 的可靠程度：`跑时记录`（运行时从 git 取到）/ `跑时记录但 dirty` / `事后补记`（事后按代码文件内容比对出来）/ `推断`（按时间或整个实验共用的 commit 推出来）/ `未知` |
| `tag` | commit 不在 main 上时保全它的 tag，`exp/<实验>-<短 commit>` |
| `entry` | 入口 |
| `args`、`args_source` | 完整参数及其来源；查不到时 `args` 为 null |
| `config` | 模型配置（`configs/baselines/*.json` 的内容） |
| `origin` | 产物原来在服务器上的位置 |
| `env`、`split`、`notes` | 运行环境；基线按 split 各一条时标明 split；来源之间不一致等说明 |

参数里的路径是当时的路径，产物已迁进 `runs/<id>/`，原位置见 `origin`。

## code/

只服务本实验的代码：诊断与画图脚本、一次性命令。微调方法不写在这里，写成 `configs/<m>.toml`。旧实验的任务清单和驱动也留在这里，是当时的记录，对现在的训练代码不再可用。进 git，同样要先提交再跑。被第二个实验用到时，提升到 `scripts/` 或包里。产出的图放 `extra/`。

## extra/

附件，例如诊断图表和它们的数据。只有 notes.md 引用时才会显示出来。

## 评价

指标定义见 `workbench/protocol.py`，不要在实验代码里另算一套写进记录。pair 误差是光学检查点经 A 映射后到 SAR 检查点的平均距离，失败记 ∞；选模用 Val AUC@5。
