# 在本机用工作台查看实验结果

工作台是本仓库自带的网页，用来浏览实验记录：实验之间的派生关系、各方法的指标和置信区间、逐对的配准误差和残差图、Notes。它在本机运行，不需要 GPU，也不访问外网。

实验结果（每对影像的预测仿射矩阵）已经在仓库里，指标在打开页面时现算。本机只需要另外准备一份**精简数据集**（约 2 GB）：它提供检查点标注，以及显示用的光学图和 SAR 图。

## 1. 准备 Python 环境

需要 Python ≥ 3.11，以及 numpy、tifffile、Pillow：

```bash
conda create -n moon-wb python=3.11 numpy tifffile pillow    # 或用已有的 Python：pip install numpy tifffile pillow
conda activate moon-wb
```

## 2. 获取代码

```bash
git clone https://github.com/Guochu0-0/moon-exp
cd moon-exp
```

没有 git 时，也可以在 GitHub 页面上点 Code → Download ZIP 下载后解压。

## 3. 下载并解压精简数据集

从网盘下载 `moon-dataset-slim.zip`，解压到任意位置（路径里最好没有空格）。解压后的结构是：

```
D:/data/moon-dataset-slim/
  Processed_Data/
    Val/ROI_xxx/{Optical,SAR,Label}/patch_<n>.*
    Test/ROI_xxx/{Optical,SAR,Label}/patch_<n>.*
```

`Val` 和 `Test` 都要有。有的 pair 没有 `Label/*.txt`，表示这对没有标注，这是正常情况。

## 4. 启动

在仓库根目录运行（`--data` 要写在 `serve` **前面**）：

```bash
python -m workbench --data D:/data/moon-dataset-slim serve
```

`--data` 可以指向 `Processed_Data` 本身，也可以指向它的上一层。也可以用环境变量 `MOON_DATA` 代替 `--data`：

```bash
# PowerShell
$env:MOON_DATA = "D:/data/moon-dataset-slim"
python -m workbench serve
```

然后用浏览器打开 <http://127.0.0.1:8765/>。端口被占用时，在 `serve` 后面加 `--port 8770` 换一个。按 Ctrl+C 停止。

## 5. 怎么看

- **画布**：每张卡片是一个实验，连线表示「派生自」。卡片上每个方法一行，数值条是 Val 上的 AUC@10。滚轮缩放，拖动空白处平移，按 `.` 适应窗口。
- **实验页**：双击卡片进入。依次是实验信息、启动记录、比较设置（方法、参考方法、Val / Test）、结果表（含 bootstrap 95% 置信区间）、误差累积分布、逐对可视化（散点总览、残差图网格、卷帘对比），最后是 Notes。
- 各功能的详细说明见 `workbench/README.md`，术语见 `CONTEXT.md`。

## 6. 本机看不到的部分

下面这些依赖实验室服务器上的文件，没有放进仓库和精简数据集，页面上会显示需要 `sync` 的提示，可以忽略：

- 点对视图（匹配点连线、内外点）
- 中间结果视图
- 训练图表、「在 TensorBoard 中打开」

## 7. 关于改动

在画布上拖动卡片、新建或删除实验、编辑 Notes，都只会写入你本机的 `runs/`，不会影响原仓库。想恢复原状，运行 `git checkout -- runs` 和 `git clean -fd runs` 即可。

## 常见问题

- **`需要数据集根目录：--data 或环境变量 MOON_DATA`**：没有给数据集路径，见第 4 步。
- **`unrecognized arguments: --data`**：`--data` 写在了 `serve` 后面，应该写在前面。
- **`ModuleNotFoundError: tomllib`**：Python 版本低于 3.11。
- **打开某个 pair 时报找不到文件**：数据集缺了 `Val` 或 `Test` 中的某些文件，重新解压一次。

---

## 附：精简数据集是怎么做的（维护者用）

```bash
python scripts/slim_dataset.py G:/Lunar_Optical_SAR_Registration_Dataset D:/out/moon-dataset-slim
```

脚本只取 Val 和 Test：光学图和 `Label/*.txt` 原样复制；SAR 只保留工作台显示用的第 2 个波段（`workbench.dataset.SAR_DISPLAY_BAND`），每张写完都会读回并逐像素核对。工作台只用 SAR 做显示，所以精简版上的指标和显示都与原数据集相同。原数据集 Val 和 Test 约 7.4 GB，精简后约 1.9 GB。中断后重跑会跳过已做完的文件。

打包时，zip 的顶层放 `moon-dataset-slim/Processed_Data/...`，这样解压后的结构就和第 3 步一致。
