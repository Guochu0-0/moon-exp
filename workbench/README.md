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

### API

见 `server.py` 顶部。前端冒烟：`python scripts/smoke_canvas.py`（需要 Chrome / Edge 和 websocket-client；在临时目录里复制 B0 / B0m，不动仓库的 `runs/`）。

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
