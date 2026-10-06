# 实验记录规范（v2）

做实验的人和 agent 读这一份就够了：实验结果写成什么样、写到哪里、用什么写。代码放哪、怎么在服务器上启动和收尾见 `docs/agents/experiments.md`；术语（实验、方法、点亮、Notes 等）见仓库根目录的 `CONTEXT.md`。

## 流程

1. 建目录：`python -m workbench new E3 --parent E2 [--init E2/main] --title "一句话标题"`。编号取下一个未用的 `E<n>`；用户已在画布上建好的直接用。只研究已有结果、不训练的，建成**分析**（见「分析」）。
2. 跑实验，代码先提交（见 `docs/agents/experiments.md`）。微调的每个方法写一份 `configs/<方法>.toml`，用 `scripts/finetune/run.py runs/<id>` 跑，它按「训练产物」写好全部记录；自己写的训练用 `PredWriter` 写预测，需要时用 `InterWriter` 写中间结果、按「TB 日志」写日志。全部产物放在 `runs/<id>/` 里。
3. 按「notes.md」写 `runs/<id>/notes.md`，并做完该节末尾的自查。
4. `python -m workbench eval <id>` 生成 `metrics.json`（分析跳过这一步）；`python -m workbench check` 输出「记录无问题」。
5. 提交小文件。大文件由 `.gitignore` 排除，结票时由 `scripts/wt.sh close` 挪到 gpfs 主 checkout，用户需要时在本地 `python -m workbench sync <id>` 拉回。

## 目录

一个实验对应 `runs/<id>/` 一个目录，服务器上的相对路径和本地相同。影像直接读数据集，目录里只存结果。

```
runs/<id>/
  exp.toml                             元数据（由 new 生成，或手写）
  notes.md                             Notes
  metrics.json                         由 eval 生成
  preds/<method>/<split>.jsonl         每个 pair 一行，存估计的仿射
  preds/<method>/<split>.meta.json     写入时的 commit、dirty、主机、时间
  preds/<method>/<split>_matches.npz   点对（不进 git）
  inter/<method>/<name>/meta.json      中间结果的种类、坐标系、说明、单位
  inter/<method>/<name>/<split>.npz    中间结果数据（scalar / points / flow，不进 git）
  inter/<method>/<name>/<split>/*.png  中间结果数据（image，不进 git）
  tb/<method>/                         TensorBoard 日志（不进 git）
  ckpt/<method>/                       权重与训练日志（不进 git，也不同步）
  configs/<method>.toml                微调方法的配置
  launch/<method>.json                 启动记录（自动写）
  sweep/<method>/                      逐 ckpt 评测
  code/                                只服务本实验的代码
  extra/                               Notes 引用的附件
```

- **点亮**是派生状态：任一方法有任一 split 的 `preds/<method>/<split>.jsonl` 即点亮；分析以 `extra/` 下有文件为准。
- 方法按 `preds/<method>/` 自动发现。普通实验只有一个方法，命名为 `main`；基线实验（如 B0）可以挂多个。变体方法命名为 `<方法>__<变体>`（如 `loftr__minmax`）。
- split 为 `train` / `val` / `test`。

## exp.toml

只用下表的字段。commit 由 preds 的 meta 和启动记录保存，其余文字写进 notes.md。

| 字段 | 含义 |
|---|---|
| `id` | 必填，与目录名一致 |
| `title` | 必填，一句话说出这个实验要回答的问题 |
| `kind` | 分析写 `"analysis"`；普通实验省略这一行 |
| `parent` | 父实验 id；根节点省略 |
| `init` | 可选，`<实验>/<方法>`（如 `B0/roma`），表示从父实验的哪个方法起步 |
| `baseline` | 可选，默认 `false`；基线实验（不训练、作比较起点）写 `true` |
| `date` | 创建时写入，之后保持不变 |
| `[methods.<m>]` | 可选，`name` 为显示名，`caveat` 为读数时要注意的事（如「点数是采样出来的」） |

## notes.md

Notes 是实验唯一的自由文本，写成一篇短论文：Markdown，不加 front matter，UTF-8，LF 换行。agent 起草，用户随后修改；用户改过的文字保持原样，新结果追加在末尾。

写法按两篇公认的指南：Mensh & Kording《Ten simple rules for structuring papers》（PLOS Comput Biol 2017）管篇章结构，Gopen & Swan《The Science of Scientific Writing》（American Scientist 1990）管句子。读者是没参与这项工作的研究者：只读这篇 Notes，就要能明白为什么做、怎么做、发现了什么、意味着什么。范例见 `runs/E4/notes.md`。

**中心结论。** 一篇 Notes 只讲一个中心结论。第一行是一级标题，用一句话说出这个结论，例如 `# 模型依赖的区域解释不了标注点处的误差`。

**篇章。** 依次为摘要、引言、方法、结果、讨论，节名是单独成行的粗体（`**摘要**`）。每一层都按语境—内容—结论组织：整篇如此，每节如此，每段也如此。

| 节 | 写什么 |
|---|---|
| 摘要 | 一段讲完整个故事：语境，缺口，做了什么，发现了什么，结论意味着什么 |
| 引言 | 从大到小：研究走到哪一步、看到什么现象；收窄到尚未解决的缺口；最后说本文怎样回答它 |
| 方法 | 读者复核结论所需的设置：模型、数据、样本、每个量的定义和统计检验。按主题分段，段首用斜体小标题（`*模型与数据。*`）。沿用的做法点名出处即可 |
| 结果 | 一串论断，按回答问题的逻辑排列。每段的斜体小标题就是这段的论断（`*依赖区域离标注点的远近与误差无关。*`），正文给出支撑它的数字和图表 |
| 讨论 | 先说缺口怎样被填补，再写局限，最后写这一结论对后续研究意味着什么、哪些发现值得检验 |

**流向。** 一个话题讲完再进入下一个。并列的内容用平行的句式，按同一顺序排列。

**术语。** 按不熟悉本项目的读者来写。每个量在第一次出现处定义，之后全文（包括图注）用同一个名字。别的实验和本实验的模型都按它们做了什么来称呼。引用论文写论文名或 arXiv 号。

**句子。**
- 句首放读者已经知道的信息，接住上一句；句尾放新的、要强调的信息。
- 先给语境再给新内容：先说在比什么、用哪批样本，再给数字。
- 主语后面紧跟谓语，动作用动词表达。
- 一句话只做一件事。

**图表。** 每张图或表支撑一个论断，图和表各自按出现顺序编号。图表放在引用它的段落之后，下一行是说明：`图 1：` 或 `表 1：` 加上它支撑的论断，必要时补一句怎么读。图用相对路径嵌入（`![](extra/concentration.png)`）。

**自查。** 交付前只读标题、摘要和结果的各个小标题，确认它们已经讲出全部发现，并且彼此一致。

## 分析

只研究已有模型或结果、不训练也不产出逐 pair 预测的实验，建成分析。每个新问题开一个分析，结论写进 Notes。节点页上只有实验信息、启动记录和 Notes；画布上的卡片标「分析」，显示附件数。

```
runs/<id>/
  exp.toml           kind = "analysis"
  notes.md           结论全部写在这里
  code/              分析脚本（进 git，先提交再跑）
  launch/main.json   启动记录：分析代码的 commit、主机、命令
  extra/             Notes 引用的图和它们的数据（csv、json 等，可放在 extra/data/ 下）
```

- 建目录：`python -m workbench new E5 --analysis --title "一句话标题"`。`parent` 可以留空，涉及哪些实验在 Notes 里写明；没有 `init`。
- 分析只写上面这些文件；`check` 发现分析下有 preds 会报错。
- 用 `workbench.launch` 启动，方法名写 `main`：`python -m workbench.launch begin runs/<id> main -- python runs/<id>/code/<脚本>.py …`，跑完 `python -m workbench.launch end runs/<id> main ok`。分析的 commit 以 `launch/` 下的记录为准。
- 节点页上没有指标，结论用到的数字全部写在 Notes 里。

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
- `A` 是把**光学像素坐标 (x 向右, y 向下) 映射到 SAR 像素坐标**的 2×3 仿射，坐标在原始 512 px 网格上；方法内部 resize 过的，先映射回原始网格。
- 匹配类方法在跑实验时用 `cv2.estimateAffine2D`（RANSAC 3 px）算好仿射再写；工作台只读取仿射。
- `A` 为 `None` 表示失败，原因写在 `fail` 里。有标注但没有预测行的 pair 也按失败计。
- 其余关键字参数原样写进这一行。
- `matches` 用 matcher 的坐标约定（整数 = 像素中心），内容是 RANSAC 前的全部点。+0.5 换算、按 conf 截断到 2000 个点、写 npz 都由 `PredWriter` 负责；没有 conf 就省略。0 个点的 pair 照写；出错的 pair 省略 `matches`。
- `PredWriter` 在 close 时写 `<split>.meta.json`（`commit`、`dirty`）。

## 中间结果：inter

想在工作台上逐 pair 查看的中间量（置信度图、位移场等），按方法、名称、split 写，可以只覆盖部分 pair：

```python
from workbench.records import InterWriter

with InterWriter("runs/E3", "main", "certainty", "val", kind="scalar", frame="opt", desc="RoMa certainty") as w:
    for pair in pairs:
        w.write(pair, cert)
```

| `kind` | 每个 pair 的数据 |
|---|---|
| `scalar` | H×W |
| `points` | N×3 `(x, y, v)`，坐标是 frame 那张 patch 的原始像素坐标，用 matcher 的约定（+0.5 由写入端负责） |
| `flow` | H×W×2，frame 坐标系下每个像素指向另一模态中对应点的位移 `(dx, dy)`，单位是原始 patch 的 px |
| `image` | uint8 H×W[×3\|4]，原样显示 |

- `frame` 为 `opt` / `sar`：数据所在的坐标系。分辨率任意，显示时拉伸到 patch 大小。
- `desc` 一句话说明，`unit` 单位，可空。
- 形状不符时抛 ValueError。构造时清掉这个 split 的旧数据，重跑即整份替换；with 块里抛异常时不留半份数据。
- `meta.json` 进 git，数据不进。

## TB 日志

- 自己写的训练用两个 `SummaryWriter`，分别写 `runs/<id>/tb/<method>/scalars/`（标量）和 `runs/<id>/tb/<method>/media/`（图片、直方图）。**显式传 `log_dir`**：缺省时日志会写到 `./runs/<时间>_<host>`，和实验目录 `runs/` 撞名。
- 每个标量用 `add_scalar` 单独写（`add_scalars` 会建子目录）。
- 上游 Lightning 代码把 `save_dir` 指向 `runs/<id>/tb/<method>/`，日志落在 `tb/<method>/<name>/version_N/`，ckpt 放在 version 目录的 `checkpoints/` 下。
- 续训接着写到同一个目录，重叠的 step 由读取端处理。
- 服务器的训练环境没装 tensorboard，标量用 `workbench.tbwriter.ScalarWriter` 写（纯标准库，格式与官方一致）；`finetune.train` 的 `--tb` 就是用它写的。

## 训练产物

微调实验的每个方法 `<m>` 由一份 `configs/<m>.toml` 描述（进 git，先提交再跑）：用哪个模型、哪些训练成分、与默认不同的参数；同一实验内可用 `base` 继承，文件名以 `_` 开头的只当 base。格式见 `finetune/README.md`。

`scripts/finetune/run.py runs/<id>` 逐个领取这些方法，对每个方法依次做：训练 → 每个 ckpt 在 Val 上评测 → 按 Val AUC@5 峰值（不含 step 0）选 ckpt，在 Test 上补评 → 把选中 step 的 Val、Test 预测复制成 `preds/<m>/`，即这个方法的正式结果。

```
runs/<id>/
  configs/<m>.toml     方法的配置（进 git，手写）
  ckpt/<m>/            ckpt_<step>.pt、log.jsonl、args.json、train.log、sweep.log（不进 git）
  tb/<m>/scalars/      训练曲线（不进 git）
  launch/<m>.json      启动记录（进 git，见「启动记录」）
  sweep/<m>/S/         逐 ckpt 的工作台记录，方法名 step<N>；exp.toml、metrics.json 进 git，preds/ 不进
  sweep/<m>/match/     逐 ckpt 的原始点对（不进 git）
  sweep/<m>/peak.json  选中的 step 与 Val / Test 指标（进 git）
  sweep/<m>/neg/、collapse.json   负样本对监控（LoFTR 的在线信号任务；neg/ 不进 git）
  preds/<m>/           选中 step 的预测
  .claims/<m>/         队列占位（不进 git）
```

工作台上只出现 `preds/<m>/`，sweep 里的各个 step 只作为评测记录。换选模规则时，在 Notes 里写明，再把对应 step 的预测复制过去。

## 启动记录

`launch/<m>.json` 是一个列表，每次启动（含续训、重跑）一条，由 `workbench.launch` 在启动时自动写：`start`、`end`、`status`、`commit`、`dirty`、`allow_dirty`、`host`、`cwd`、`cuda_visible_devices`、`cuda_device_order`、`cmd`。

微调驱动 `scripts/finetune/run.py` 另记：`entry`（入口）、`config_file`（`configs/<m>.toml`）、`args`（展开所有默认值后的完整配置，按表嵌套：`model`、`optim`、`run` 和用到的训练成分）、`config`（评测用的推理配置，`configs/baselines/*.json` 的内容）。

2026-10-03 以前的方法由 `scripts/backfill_launch85.py` 事后补记，节点页同样显示。补记的每条另有：

| 字段 | 含义 |
|---|---|
| `backfilled` | 补记日期；有这个字段就是事后补记 |
| `reliability` | commit 的可靠程度：`跑时记录`（运行时从 git 取到）/ `跑时记录但 dirty` / `事后补记`（按代码文件内容比对出来）/ `推断`（按时间或整个实验共用的 commit 推出来）/ `未知` |
| `tag` | commit 不在 main 上时保全它的 tag，`exp/<实验>-<短 commit>` |
| `entry` | 入口 |
| `args`、`args_source` | 完整参数及其来源；查不到时 `args` 为 null |
| `config` | 模型配置（`configs/baselines/*.json` 的内容） |
| `origin` | 产物原来在服务器上的位置 |
| `env`、`split`、`notes` | 运行环境；基线按 split 各一条时标明 split；来源之间不一致等说明 |

补记参数里的路径是当时的路径；产物已迁进 `runs/<id>/`，原位置见 `origin`。

## code/

只服务本实验的代码：诊断与画图脚本、一次性命令，进 git，先提交再跑，产出的图放 `extra/`。微调方法写成 `configs/<m>.toml`。被第二个实验用到的代码提升到 `scripts/` 或包里。旧实验的任务清单和驱动是当时的记录，现在的训练代码已用不了它们。

## extra/

Notes 引用的附件：诊断图表和它们的数据。工作台只显示被 notes.md 引用的文件；Notes 不再引用的图随手删掉。

## 评价

指标定义只在 `workbench/protocol.py` 一处，写进记录的指标都由它算。pair 误差是光学检查点经 A 映射后到 SAR 检查点的平均距离，失败记 ∞；选模用 Val AUC@5。
