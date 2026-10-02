# 实验记录规范（v2）

做实验的人和 agent 读这一份就够了：实验结果写成什么样、写到哪里、用什么写。代码放哪、怎么在服务器上启动和收尾见 `docs/agents/experiments.md`。工作台怎么展示这些记录见 `README.md`，做实验时不需要读。术语（实验、方法、点亮、Notes 等）见仓库根目录的 `CONTEXT.md`。

## 流程

1. 建目录：`python -m workbench new E3 --parent E2 [--init E2/main] --title "一句话标题"`。编号取下一个未用的 `E<n>`；用户已在画布上建好的直接用。
2. 跑实验（先提交代码，见 `docs/agents/experiments.md`）。微调用 `scripts/finetune/run.py`，它按下文「训练产物」写好全部记录；自己写的用 `PredWriter` 写预测，需要时用 `InterWriter` 写中间结果、按下文约定写 TB 日志。产物全部放在 `runs/<id>/` 里，不要写到别处。
3. 假设、改动、看完结果后的分析写进 `runs/<id>/notes.md`。
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
  launch/<method>.json                 每次启动的 commit、主机、GPU、命令、起止时间（自动写）
  sweep/<method>/                      逐 ckpt 评测（见「训练产物」）
  code/                                只服务本实验的代码：任务清单、驱动、诊断与画图脚本
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

实验唯一的自由文本：假设、改动、看完结果后的分析都写在这里。Markdown，不加 front matter，UTF-8，LF 换行。引用附件用相对路径，例如 `![](extra/offset/offset_val.png)`。

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
- 服务器的训练环境没装 tensorboard，标量可用 `workbench.tbwriter.ScalarWriter`（纯标准库，读写格式与官方一致）；`finetune.train`、`finetune.train_roma` 的 `--tb` 就是用它写的。

## 训练产物

`scripts/finetune/run.py` 对一个方法 `<m>` 依次做：训练 → 每个 ckpt 在 Val 上评测 → 按 Val AUC@5 峰值（不含 step 0）选 ckpt，在 Test 上补评 → 把选中 step 的 Val、Test 预测复制成 `preds/<m>/`，即这个方法的正式结果。

```
runs/<id>/
  ckpt/<m>/            ckpt_<step>.pt、log.jsonl、args.json、train.log、sweep.log（不进 git）
  tb/<m>/scalars/      训练曲线（不进 git）
  launch/<m>.json      启动记录（进 git）
  sweep/<m>/S/         逐 ckpt 的工作台记录，方法名 step<N>；exp.toml、metrics.json 进 git，preds/ 不进
  sweep/<m>/match/     逐 ckpt 的原始点对（不进 git）
  sweep/<m>/peak.json  选中的 step 与 Val / Test 指标（进 git）
  sweep/<m>/neg/、collapse.json   负样本对监控（LoFTR 的在线信号任务；neg/ 不进 git）
  preds/<m>/           选中 step 的预测
  .claims/<m>/         队列占位（不进 git）
```

sweep 里的每个 step 不是独立的方法：只有 `preds/<m>/` 出现在工作台上。想换选模规则，就在 Notes 里写明，再把对应 step 的预测复制过去。

## code/

只服务本实验的代码：任务清单、驱动、诊断与画图脚本。进 git，同样要先提交再跑。被第二个实验用到时，提升到 `scripts/` 或包里。产出的图放 `extra/`。

## extra/

附件，例如诊断图表和它们的数据。只有 notes.md 引用时才会显示出来。

## 评价

指标定义见 `workbench/protocol.py`，不要在实验代码里另算一套写进记录。pair 误差是光学检查点经 A 映射后到 SAR 检查点的平均距离，失败记 ∞；选模用 Val AUC@5。
