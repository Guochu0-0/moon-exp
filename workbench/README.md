# 实验工作台 v1

思路树 + 节点详情。每个实验是树上的一个节点，边表示「从哪个实验改出来的」。形态由「实验工作台原型」(#4) 定下，评价口径由「定义评价协议与指标」(#2) 定下。

```bash
export MOON_DATA=G:/Lunar_Optical_SAR_Registration_Dataset   # 或 --data；服务器上指向 YGC/dataset/Moon
python -m workbench serve            # http://127.0.0.1:8765/  （服务器上跑的话，用 ssh -L 转发端口）
python -m workbench eval [id ...]    # 算指标，写 runs/<id>/metrics.json
python -m workbench check            # 按 v2 规则检查记录（旧字段只警告）
python -m workbench new E3 --parent E2 --title "…"   # 按 v2 模板新建实验目录
```

依赖：numpy、tifffile、Pillow，Python ≥ 3.11（用到 tomllib）。不需要 cv2。

## 记录格式（v2）

一个实验对应 `runs/<id>/` 一个目录，全部产物都放在里面。小文件进 git；大文件（点对、中间结果数据、TB 日志、ckpt）由 `.gitignore` 排除，服务器上的相对路径和本地完全一样。影像不存，页面现读本地数据集。术语见仓库根目录的 `CONTEXT.md`。

```
runs/<id>/
  exp.toml                             元数据（手写，或由 new 生成）
  notes.md                             Notes，可以没有
  metrics.json                         派生物，由 eval 生成，不要手改
  preds/<method>/<split>.jsonl         每个 pair 一行，存估计的仿射
  preds/<method>/<split>.meta.json     写入时的 commit 与 dirty
  preds/<method>/<split>_matches.npz   点对，不进 git
  extra/                               附件
```

**点亮**是派生状态：任一方法有任一 split 的 `preds/<method>/<split>.jsonl` 就算点亮，不需要手填。方法列表按 `preds/<method>/` 自动发现。

### exp.toml

| 字段 | 含义 |
|---|---|
| `id` | 必填，与目录名一致 |
| `title` | 必填，一句话标题 |
| `parent` | 父实验 id；根节点整行省略 |
| `init` | 可选，写成 `<实验>/<方法>`（如 `B0/roma`），表示从父实验的哪个方法起步 |
| `baseline` | 可选，默认 `false`。基线实验（不训练、作比较起点）写 `true` |
| `date` | 创建时写入，之后不改 |
| `[methods.<m>]` | 可选，`name` 为显示名，`caveat` 为读数时要注意的事（如「点数是采样出来的」） |

- v1 的 `status`、`commit`、`hypothesis`、`change`、`verdict`、`next` 已删除。读到这些旧字段或未知字段时，`check` 给警告，不报错。
- commit 不手填，由读取端从各方法的 `<split>.meta.json` 汇总：全部相同时是单值，不同时是「多个」（可展开看各方法），没有 meta 时是「未知」。
- 变体方法 `x__v` 没写 `[methods.x__v]` 时，显示名和 caveat 沿父链回退到基础方法 `x`，显示名后面加「· v」。例如 B0m 的 `loftr__minmax` 显示为「LoFTR outdoor_ds · minmax」。
- 一个实验可以有多个方法，比如基线 B0 下挂十几个 zero-shot 方法。普通实验只有一个方法，惯例命名为 `main`。

### notes.md

实验唯一的自由文本：假设、改动、看完结果后的分析都写在这里。Markdown，不加 front matter，UTF-8，LF 换行。文件不存在就当作空。引用附件用相对路径，例如 `![](extra/offset/offset_val.png)`。

### preds/&lt;method&gt;/&lt;split&gt;.jsonl

```json
{"pair": "ROI_037/patch_1", "A": [[a, b, c], [d, e, f]], "n_inliers": 57, "sec": 0.12}
{"pair": "ROI_037/patch_2", "A": null, "fail": "few_inliers"}
```

- `pair` 写 `<ROI 目录名>/<文件名 stem>`，与 `Processed_Data/<Split>/<ROI>/{Optical,SAR,Label}/<stem>.*` 一一对应。
- `A` 是把**光学像素坐标 (x 向右, y 向下) 映射到 SAR 像素坐标**的 2×3 仿射，坐标在原始 512 px 网格上。方法内部 resize 过的，必须先映射回原始网格。
- `A: null` 表示失败，原因写在 `fail` 里。有标注但没有预测行的 pair 也按失败计。
- 其余字段随意，页面会原样带出。
- 匹配类方法的仿射由估计器（`cv2.estimateAffine2D`，RANSAC 3 px）在跑实验时算好再写进来。工作台只消费仿射，自己不估计。

### preds/&lt;method&gt;/&lt;split&gt;.meta.json

```json
{"commit": "<git rev-parse HEAD>", "dirty": false}
```

`PredWriter` 在 close 时写出。`dirty` 只看已跟踪的代码：未跟踪文件和 `runs/` 下的产物不算。迁移来的记录另有 `migrated_from`。

### 点对

`<split>_matches.npz` 以 pair 为 key（`/` 换成 `__`），每个 value 是 N×5 float32 的 `(x_opt, y_opt, x_sar, y_sar, conf)`：

- 内容是 RANSAC 前的全部点，坐标按 +0.5 约定，与 `A` 和标注一致。不存内点标记，内点 / 外点由估计的仿射重新判定。
- 任何方法超过 2000 个点，都按 conf 取前 2000。没有 conf 时这一列填 NaN。
- 0 个点的 pair 写 0×5；出错的 pair 不写 key。

点对不进 git（`runs/*/preds/*/*_matches.npz`），在服务器上生成，需要时拉回本地。

### extra/

附件，例如诊断图表和它们的数据。页面不单独展示附件，只有 Notes 引用它们时才显示出来。

### 写记录的代码

```python
from workbench.records import PredWriter

with PredWriter("runs/B0", "roma", "test") as w:
    for pair in pairs:                      # Dataset(root).pairs("test")
        w.write(pair, A, n_inliers=k, matches=M)   # A: 2×3；M: RANSAC 前的全部点，N×5，或 N×4 另给 conf=
        # 失败时：w.write(pair, None, fail="few_inliers", matches=M)
```

`matches` 取 matcher 的坐标约定（整数 = 像素中心）。+0.5、按 conf 截断到 2000、写 npz 和 meta.json 都由写入端负责。

## 评价

见 `protocol.py`：
- pair 误差 = 光学检查点经 A 映射后到 SAR 检查点的平均距离；失败记 ∞。
- split 内全部有标注 pair 等权。
- 主指标附 bootstrap 95% CI。
- 档位已按「锁定评价阈值档位」(#9) 锁定：主表 AUC@3/5/10 + SR@3/5/10，选模用 Val AUC@5，不设 T_粗；其余档位照算，供附表。改档位只改 `protocol.py` 顶部，再重跑 `eval`。

## 页面口径

- 节点同时列出 Val 和 Test。边上的数字是 Val 主指标相对比较基准的变化。
- 逐对可视化在 Test 上，可以和父节点、「未配准」或任意方法对比。
- 按编号检索支持两种写法：
  - split 内编号，例如 `10` / `#0010`。编号规则：ROI 升序、patch 序号按数值升序，从 0 起，包含无标注 pair。
  - `ROI_037/1` 这样的写法。
- URL hash 可以直接分享，例如 `#E1&pair=ROI_037/patch_1&layer=match`。
- 影像按分位数拉伸显示，SAR 取第 2 波段。这只影响显示，与网络输入的归一化无关。
