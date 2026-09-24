# 实验工作台 v1

思路树 + 节点详情。每个实验是树上的一个节点，边表示「从哪个实验改出来的」。形态由「实验工作台原型」(#4) 定下，评价口径由「定义评价协议与指标」(#2) 定下。

```bash
export MOON_DATA=G:/Lunar_Optical_SAR_Registration_Dataset   # 或 --data；服务器上指向 YGC/dataset/Moon
python -m workbench serve            # http://127.0.0.1:8765/  （服务器上跑的话，用 ssh -L 转发端口）
python -m workbench eval [id ...]    # 算指标，写 runs/<id>/metrics.json
python -m workbench check            # 检查记录的一致性
python -m workbench new E3 --parent E2 --title "…"   # 新建实验目录，自动填入当前 commit
```

依赖：numpy、tifffile、Pillow，Python ≥ 3.11（用到 tomllib）。不需要 cv2。

## 记录格式

一个实验对应 `runs/<id>/` 一个目录，整个目录进 git。影像不存，页面现读本地数据集。

```
runs/<id>/
  exp.toml                             元数据（手写）
  preds/<method>/{val,test}.jsonl      每个 pair 一行，存估计的仿射
  preds/<method>/{val,test}_matches.npz  可选：点对
  extra/                               灵活区产物
  metrics.json                         派生物，由 eval 生成，不要手改
```

### exp.toml

| 字段 | 含义 |
|---|---|
| `id` | 与目录名一致 |
| `title` | 一句话标题 |
| `parent` | 父实验 id；根节点留空 |
| `init` | 可选，写成 `B0/roma`，表示从父实验的哪个方法起步。比较基准的取法：有 `init` 时用它；否则用父实验里 Val 主指标最好的方法；没有父实验时用「未配准」 |
| `date`、`commit` | 跑实验时的日期和 commit。页面据此显示 commit 标题，以及相对父实验 commit 的 diffstat |
| `status` | `baseline` / `running` / `kept` / `dropped`。只依据 Val 判定 |
| `hypothesis`、`change`、`verdict`、`next` | 假设、改动、结论（依据 Val）、下一步计划。多行字符串 |
| `[methods.<m>] name = "…"` | 可选，方法的显示名 |

一个实验可以有多个方法，比如 baseline 节点下挂 10 个 zero-shot 方法。普通实验只有一个方法，惯例命名为 `main`。

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

### 点对（可选）

`<split>_matches.npz` 以 pair 为 key（`/` 换成 `__`），每个 value 是 N×4 的 `(x_opt, y_opt, x_sar, y_sar)`。页面的「点对连线」视图用它。

### extra/

实验自己想刻画的东西都放这里，工作台只负责展示：
- `png` / `jpg` / `svg`：显示为图片
- `html`：用 iframe 嵌入
- `md` / `txt` / `csv`：显示为文本
- 简易图表 JSON：
  - 折线：`{"type": "line", "x": [...], "series": {"reward": [...]}, "xlabel": "step"}`
  - 柱状：`{"type": "bar", "labels": [...], "values": [...]}`

训练曲线也放这里。

### 写记录的代码

```python
from workbench.records import PredWriter

with PredWriter("runs/B0", "roma", "test") as w:
    for pair in pairs:                      # Dataset(root).pairs("test")
        w.write(pair, A, n_inliers=k, matches=M)                # A: 2×3；M: N×4，可省
        # 失败时：w.write(pair, None, fail="few_inliers")
```

## 评价

见 `protocol.py`：
- pair 误差 = 光学检查点经 A 映射后到 SAR 检查点的平均距离；失败记 ∞。
- split 内全部有标注 pair 等权。
- 主指标附 bootstrap 95% CI。
- 档位（AUC / SR / T_粗）目前是**暂定值**，等「锁定评价阈值档位」(#9) 锁定后只改 `protocol.py` 顶部，再重跑 `eval`。

## 页面口径

- 节点同时列出 Val 和 Test。边上的数字是 Val 主指标相对比较基准的变化。
- 逐对可视化在 Test 上，可以和父节点、「未配准」或任意方法对比。
- 按编号检索支持两种写法：
  - split 内编号，例如 `10` / `#0010`。编号规则：ROI 升序、patch 序号按数值升序，从 0 起，包含无标注 pair。
  - `ROI_037/1` 这样的写法。
- URL hash 可以直接分享，例如 `#E1&pair=ROI_037/patch_1&layer=match`。
- 影像按分位数拉伸显示，SAR 取第 2 波段。这只影响显示，与网络输入的归一化无关。
