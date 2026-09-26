# zero-shot baseline runner

「重跑 baseline，量化 zero-shot 失败模式」(#3) 的执行代码。结果进工作台记录 `runs/B0/`，每个方法一个 `preds/<method>/`。

```
各方法自己的环境                              工作台环境（py≥3.11 + cv2）
python -m baselines.match <cfg> --split …  →  python -m baselines.fit <raw>/<method> --split … --run runs/B0
  原始点对 npz + 逐 pair 日志 + meta            估仿射 → preds/<method>/<split>.jsonl
                                             python -m workbench eval B0
```

两步拆开，是因为每个方法的官方环境互不兼容，而统一的 RANSAC 和评价只该有一份实现。原始点对留在服务器 `YGC/results/baselines/`（不进 git），所以改 RANSAC 阈值做敏感性分析时只需重跑 `fit`。

## 约定

- **输入**（`inputs.py`）：主表对所有方法用同一套 float32 [0,1] 映射，光学 `div255`、SAR `p2p98`（S1 = b1+b2 → dB → 下限 −25 → 逐 patch p2–p98 拉伸）。消融映射也在这里登记。
- **适配器**（`adapters/`）：从张量层接入官方 pipeline，跳过官方的 uint8 loader，在模块 docstring 里写明对照的官方代码行；不得不量化的地方写进 `notes`（会进 meta）。分辨率用各方法官方默认。
- **坐标**：适配器输出原 512 网格、0-based、整数 = 像素中心。resize 回映统一用 `(x+0.5)·s−0.5`。`fit` 再整体 +0.5，与标注（ArcGIS 角点原点）对齐。
- **失败**：匹配报错/超时 → `error: …`；点对 < 3 → `few_matches`；RANSAC 内点 < 3 → `few_inliers`；没跑到 → `not_run`。均按 ∞ 计。
- **复现**：配置里固定 seed，每个 pair 前重置；meta 记录配置、权重 sha256、本仓库与方法仓库的 commit、环境版本、机器、GPU、耗时。

## 配置

`configs/baselines/<method>.json`：

```json
{"method": "loftr", "adapter": "loftr", "repo": "third_party/LoFTR",
 "weights": "official/loftr/outdoor_ds.ckpt",            // 相对 --weights-root（$MOON_WEIGHTS）
 "params": {"long_side": 840, "temp_bug_fix": false},    // 传给适配器
 "input": {"optical": "div255", "sar": "p2p98"}, "seed": 0}
```

## 在服务器上跑（154）

先读 `docs/agents/servers.md`：先看 GPU 占用，只用空卡；nice/ionice；环境装在容器本地 `/opt`，不放 gpfs。

```bash
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon
export MOON_WEIGHTS=/remote-home/xufang/YGC/weights
export MOON_RESULTS=/remote-home/xufang/YGC/results
git submodule update --init third_party/LoFTR    # 经工位机隧道：git -c http.proxy=http://127.0.0.1:17890 …

# matcher 侧
CUDA_VISIBLE_DEVICES=<空卡> nice -n 10 /opt/envs/loftr/bin/python -m baselines.match configs/baselines/loftr.json \
    --split val --out $MOON_RESULTS/baselines
# 估计 + 评价
/opt/envs/wb/bin/python -m baselines.fit $MOON_RESULTS/baselines/loftr --split val --run runs/B0
/opt/envs/wb/bin/python -m workbench eval B0
```

`--limit N` 只跑前 N 个 pair；`--resume` 续跑；`--timeout` 单 pair 超时（默认 120 s）。

## 环境

| 环境 | 用途 | 配方 |
|---|---|---|
| `wb` | 工作台、fit、测试 | python 3.11；numpy、opencv-python-headless、tifffile、Pillow、pytest |
| `loftr` | LoFTR | python 3.10；torch + 对应 CUDA；kornia 0.6.x、einops、yacs、loguru、opencv-python-headless、numpy<2 |
| `loftr` + JDK | CoMIR（mpicbg SIFT） | 在 `loftr` 里加 `JPype1`；JDK（`JAVA_HOME`）；`/opt/mpicbg/` 放 mpicbg、ij、jama 三个 jar（scijava maven） |

## 方法

| method | adapter | 权重 | 状态 |
|---|---|---|---|
| identity | identity | — | 链路自检，应与「未配准」一致 |
| loftr | loftr | `official/loftr/outdoor_ds.ckpt` | 已接入 |
