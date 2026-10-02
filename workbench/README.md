# 实验工作台

画布 + 节点页。每个**实验**是画布上的一个节点，连线表示「派生自」：子实验是从父实验改出来的。术语见仓库根目录的 `CONTEXT.md`，规格见 #37。

```bash
export MOON_DATA=G:/Lunar_Optical_SAR_Registration_Dataset   # 或 --data；服务器上指向 YGC/dataset/Moon
python -m workbench serve            # http://127.0.0.1:8765/  （服务器上跑的话，用 ssh -L 转发端口）
python -m workbench eval [id ...]    # 算指标，写 runs/<id>/metrics.json
python -m workbench check            # 按 v2 规则检查记录（旧字段只警告）
python -m workbench new E3 --parent E2 [--init E2/main] --title "…"   # 按 v2 模板新建实验目录
python -m workbench sync B0 B0m [--extra certainty] [--tb all]   # 从服务器拉回点对等不进 git 的文件
```

依赖：numpy、tifffile、Pillow，Python ≥ 3.11（用到 tomllib）。不需要 cv2。页面零构建：Python 标准库 HTTP 服务 + 静态页（`static/`），字体自托管，不访问外网。

## 画布

打开 `/` 就是画布。页面每隔几秒、以及窗口重新获得焦点时重拉数据，所以在终端或由 agent 改了 `runs/`（新写的 exp.toml、刚点亮的实验），几秒内就会出现在画布上，不用重启服务。

- **卡片**：标题栏是编号、标题、「基线」标签和状态点（实心绿 = 已**点亮**，空心 = 未点亮）。已点亮的实验每个方法一行，带 AUC@10（Val）数值条（0–1 满刻度）和输出端口，端口按方法名着色：含 `roma` 蓝、含 `loftr` 琥珀、其余淡紫。默认显示前 4 个方法，被子实验用作 init 的方法总是显示，其余折叠（点「另外 n 个方法」展开）。单方法实验只有一行「本实验」。未点亮的是虚线框，写「尚无结果」。
- **连线**：从父实验的 init 方法端口接出（没有 init 时从「子实验」通用端口接出）；接向未点亮实验的连线更淡。
- **浏览**：滚轮缩放，拖空白平移，`.` 适应窗口，`+` `−` 缩放。视口存在浏览器 localStorage，不进 git。
- **新建**：双击空白、按 `E`、点右上角「新建实验」或空白处右键。编号自动取下一个 `E<n>`，随即编辑标题（Enter 确认，Esc 放弃）。创建时就写出 `runs/<id>/exp.toml`。
- **派生**：从输出端口拖到空白处；或在方法行 / 卡片上右键「从 … 派生新实验」。从方法端口派生的自动写 `init = "<父实验>/<方法>"`。
- **改父**：从端口拖到另一个实验上，改写那个实验的 parent / init，已点亮的也可以。会成环时拒绝并提示。右键「断开父实验」让它成为根。
- **改编号 / 改标题**：右键菜单。改编号只在未点亮时允许，子实验的 parent / init 和画布坐标一起改。
- **删除**：右键或选中后按 `Del`（没有选中时删鼠标下的实验、便签或分组框）。只能删未点亮的实验（删掉 `runs/<id>/` 整个目录），它的子实验断开父实验；已点亮实验的删除项置灰，多选删除时跳过并提示。
- **进入节点页**：双击实验或选中后按 Enter，地址是 `#/exp/<id>`；「← 画布」返回，该实验居中并选中。
- **便签**：按 `N`、点右上角「便签」或空白处右键新建，随即编辑；双击编辑（Enter 确认，Shift+Enter 换行，Esc 放弃），右键编辑或删除。
- **分组框**：按 `G`、点右上角「分组框」或空白处右键新建；`Ctrl+G` 把选中对象打成一组并进入重命名。拖标题栏时，整个落在框里的实验和便签一起移动；右下角手柄调大小；双击标题重命名。右键：在此新建实验、重命名、换颜色（4 种轮换）、贴合框内内容、删除（框里的东西不动）。分组框只组织视觉，不影响思路树。
- **选择**：单击选中；`Ctrl` / `Shift` + 单击加选或减选；在多选中单击但没拖动，只留这一个；单击空白取消选择。`Ctrl` + 从空白处拖是框选，`Ctrl+A` 全选实验和便签。拖动选中的对象时整组一起移动。
- 拖动松手后保存坐标。`Esc` 逐级退出：先关菜单 / 快捷键面板，再取消拖线，最后取消选择。`?` 或左下角「快捷键」打开快捷键面板；右下角小地图点哪里就跳到哪里。

所有语义改动都写回各实验的 `exp.toml`（按行改，注释和其余字段保留），布局写回 `runs/canvas.json`；两者都进 git。多个写入方各自读-改-写，以后写的为准，不加锁。

### runs/canvas.json

```json
{
  "experiments": {"B0": {"x": 120, "y": 80}},
  "groups": [{"color": "c1", "h": 260, "id": "g1", "title": "零样本基线", "w": 800, "x": -20, "y": -20}],
  "stickies": [{"h": 80, "id": "s1", "text": "Test 只在定稿时跑", "w": 230, "x": 0, "y": 300}]
}
```

坐标是世界坐标（px），原点在左上；实验卡片宽度固定，只存 x、y。分组框的 `color` 取 `c1`–`c4`（蓝、棕、绿、紫）；便签和分组框只存上面这些字段；经 API 写入时不合法就整体拒绝，手改出的毛病（漏写 color、w、h 之类）在读取时补默认值，不会挡住之后的写入。写入时丢掉目录已不存在的实验，key 排序，缩进 2 格。没有坐标的实验（比如终端或 agent 新建的）由服务端算默认位置：父实验右侧，兄弟实验往下排，避开已有实验；读数据时不写文件，第一次在画布上做改动时才把它们当前的位置存下来。agent 只写 exp.toml（和 notes.md），不用碰 canvas.json。

## 节点页

`#/exp/<id>`：一份严谨的实验记录。只呈现事实，没有自动结论句和判定控件；唯一可编辑的是 Notes。左栏是章节导航（点击跳转，滚动时高亮当前章节），没有内容的章节不显示。

- **实验信息**：编号、标题、父实验、起点方法（init）、子实验、状态（已点亮 / 未点亮，基线另标）、日期、commit（各方法 preds meta 汇总：单值 / 「多个」可展开 / 未知），多方法实验加方法数。
- **比较设置**：方法下拉框（仅多方法实验）；**参考方法**——父实验的各方法、本实验其他方法、其他已点亮实验、未配准、无，默认值标「（默认）」。默认规则：有 init 用 init；父实验有同名基础方法（`<方法>__<变体>` 的 `<方法>`）就用它；否则用父实验 Val 主指标最好的方法；以上都没有时基线为无，其余为未配准。数据集默认 Val，当前方法或参考方法没有 Test 结果时 Test 置灰，切到 Test 时显眼标出。
- **结果**：多方法实验先给方法 × 指标全表（按当前数据集的主指标降序，点一行切换方法；有父实验时末列是相对父实验同名基础方法的 ΔAUC@10）。单方法实验、或选了参考方法时，给方法 / 参考方法 / 差值的全指标表：各自的 95% CI 是评价协议的 bootstrap；差值的 95% CI 是配对 bootstrap（两边按同一组 pair 重采样，次数与种子同协议；Efron & Tibshirani 1993 第 13 章，Koehn 2004）。表为编号三线表，差值只标正负号。图：误差累积分布，以及各误差档（≤5、5–20、>20 px 错配、失败）的占比。指标都按 preds 现算（按文件 mtime 缓存），与 `eval` 写出的 metrics.json 同口径。
- **可视化结果**：逐对查看当前方法相对参考方法的表现，跟随比较设置（方法、参考方法、数据集；切到 Test 时同样显眼标出）。
  - 散点总览：横轴参考方法误差、纵轴方法误差。全图点一下，以该处为中心放大 5 倍；放大后点一个点选中该 pair，点空白再放大 3 倍；「缩小一级」「返回全图」。失败或 >30 px 的点画在灰色边缘带里、按编号错开，带内位置不代表数值。
  - 筛选（以 5 px 为界，附数量）：全部 / 改善 / 退化 / 均未配准 / 均配准。排序：误差下降量大在前（默认）、误差上升量大在前（选「退化」时自动切换）、方法误差大在前、编号；按误差差值排序时失败与 >50 px 都按 50 px 计。可按编号查找。没有参考方法时只有排序和网格，默认方法误差大在前。
  - 缩略图网格：一次 12 张残差图（光学原图上的检查点与原尺寸残差箭头），可再显示 12 张。
  - 详情：方法与参考方法并排，视图有卷帘（默认）、残差、原图；滚轮缩放、拖动平移、双击复原，各面板同步。← → 在当前筛选和排序的列表里移动。失败的一侧显示失败原因和光学原图。
  - 点对视图（本地有点对时出现）：点对连线（全部点，内点 / 外点由估计的仿射按 3 px 残差重新判定后着色）与残差点图（光学图上的匹配点，颜色为到估计仿射的残差）。conf 过滤按分位数取值，方法没有 conf 时隐藏，conf 全部相同时注明不起作用；0 个点的 pair 照常显示；方法的 caveat 显示在视图旁。本地没有点对时给出可直接复制的 `python -m workbench sync <id>`。
- **Notes**：点「编辑」或双击就地编辑（Markdown，Ctrl+S 保存，Esc 取消），保存到 `runs/<id>/notes.md`（UTF-8、LF，第一次保存时才创建）。平时按 Markdown 渲染，相对路径按实验目录解析，所以 `![](extra/offset/offset_val.png)` 能显示附件；不放行原始 HTML。
- **未点亮的实验**：只有实验信息、「尚无结果。」和 Notes。

### API

见 `server.py` 顶部。前端冒烟（画布与节点页）：`python scripts/smoke_canvas.py`（需要 Chrome / Edge 和 websocket-client；在临时目录里复制 B0 / B0m，不动仓库的 `runs/`）。点对视图那几步要 B0 的点对：先 `python -m workbench sync B0`；在 worktree 里跑时用 `--matches-from <主工作区>/runs` 指过去。

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

### 同步

`python -m workbench sync <id>...` 按实验从服务器 `runs/<id>/` 拉回不进 git 的文件：

- 点对：默认拉。
- 中间结果：只拉 `--extra <name>` 点名的，可重复。
- TB 日志：默认只拉 `tb/<m>/scalars/` 和上游格式的 `version_N/` 目录；`--tb all` 拉整个 `tb/`。`checkpoints/` 都排除。
- ckpt：永远不拉。

主机取 `--host`、环境变量 `MOON_SYNC_HOST`，默认 `xufang154外网`；远端根目录取 `--remote-root`、`MOON_SYNC_ROOT`，默认 `/remote-home/xufang/YGC/moon-exp`。传输只用 ssh 与 tar（Windows 11 自带），不需要 rsync。本地已有且大小、mtime 都没变的文件跳过，所以重复执行很便宜。

## 评价

见 `protocol.py`：
- pair 误差 = 光学检查点经 A 映射后到 SAR 检查点的平均距离；失败记 ∞。
- split 内全部有标注 pair 等权。
- 主指标附 bootstrap 95% CI。
- 档位已按「锁定评价阈值档位」(#9) 锁定：主表 AUC@3/5/10 + SR@3/5/10，选模用 Val AUC@5，不设 T_粗；其余档位照算，供附表。改档位只改 `protocol.py` 顶部，再重跑 `eval`。
