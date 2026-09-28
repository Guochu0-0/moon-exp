# 基线方法代码接口调研（Group A：LoFTR / XoFTR / SuperPoint+SuperGlue / RIFT2）

> 日期：2026-09-25。所有 `文件:行号` 均指下列 commit，源码位于 `C:\Users\yougu\.claude-work\jobs\928dd2ed\tmp\repos\`。
> 「读到」= 直接读了代码；「推断」= 我的判断，显式标出。未在本机运行任何模型。
>
> | 方法 | 仓库 | 读的 commit | 备注 |
> |---|---|---|---|
> | LoFTR | zju3dv/LoFTR | `df7ca80` | = 上游 HEAD（已 fetch 核对） |
> | XoFTR | OnderT/XoFTR | `e9635d8`（2024-09-08） | 上游 HEAD = `e0fbea4`（2025-08-21），只改 1 行，见 §2.5 |
> | SuperPoint+SuperGlue | magicleap/SuperGluePretrainedNetwork | `ddcf11f` | = 上游 HEAD |
> | RIFT2 | LJY-RS/RIFT2-multimodal-matching-rotation | `0e980ce4124d2abd62727dcf040198db0a18b269`（2022-08-26） | = 上游 HEAD；仓库只有 3 个 commit |
>
> 我们的数据：光学 512×512 uint8 单波段；SAR 512×512×4 float32 Stokes，规范量 S1=b1+b2 → dB → 下限 −25 → 全局 z-score（moon-dataset ADR 0001）。需要原 512 网格、0-based、像素中心为整数的坐标。

---

## 1. LoFTR（`df7ca80`）

### 1.1 最小官方推理路径
- 类：`src.loftr.LoFTR` + 配置字典 `default_cfg`（`src/loftr/__init__.py:1-2`）。`default_cfg` 来自 `src/loftr/utils/cvpr_ds_config.py:10-50`（yacs → 小写 dict，`:4-7`）。
- 权重：`matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])`（README.md:88-90；notebook `demo_single_pair.ipynb` cell 7）。重写的 `load_state_dict` 会把 PL checkpoint 的 `matcher.` 前缀去掉（`src/loftr/loftr.py:77-81`）。权重只在 Google Drive（README.md:57-60）。
- 调用：`matcher(batch)`，`batch={'image0','image1'}`，**原地**往 dict 里写结果，不返回值（`loftr.py:29-75`）；读 `batch['mkpts0_f'] / ['mkpts1_f'] / ['mconf']`（README.md:92-97）。
- 关键配置与官方默认：
  - `COARSE.TEMP_BUG_FIX`：`default_cfg` 里为 **False**（`cvpr_ds_config.py:28`），而训练用 `src/config/default.py:23` 为 True。**`outdoor_ds.ckpt` 是旧权重，必须配 False**：notebook 室外例子直接用 `default_cfg`（cell 7），官方复现脚本用 `configs/loftr/outdoor/buggy_pos_enc/loftr_ds.py`（`scripts/reproduce_test/outdoor_ds.sh:11-12`，该配置 `:3` 设 False）。室内新权重 `indoor_ds_new.ckpt` 才设 True（notebook cell 3）。
  - `MATCH_COARSE.THR=0.2`、`BORDER_RM=2`、`MATCH_TYPE='dual_softmax'`、`DSMAX_TEMPERATURE=0.1`（`cvpr_ds_config.py:32-36`）。
  - 室外官方测试分辨率：长边 840、pad 成方形、df=8（`configs/data/megadepth_test_1500.py:10`；`src/config/default.py:96-99`）。notebook 室外例子则**不缩放**，只裁到 8 的倍数（cell 9）。
  - 位置编码 `max_shape=(256,256)` → 输入每边 ≤ 2048 px（`src/loftr/utils/position_encoding.py:11-14,42`）。
- Sinkhorn（OT）变体需另外 wget SuperGlue 的 `superglue.py`（`coarse_matching.py:75-78`；README.md:67-73）。DS 版不需要。

### 1.2 最窄可调用边界的输入
- `torch.Tensor`，`(N,1,H,W)`，float，**单通道灰度**（`loftr.py:32-34`；backbone 第一层 `nn.Conv2d(1, ...)`，`src/loftr/backbone/resnet_fpn.py:60`）。
- 值域 [0,1]：**归一化只在 loader 里做**（`÷255`：`src/utils/dataset.py:122`；notebook cell 4/9 `torch.from_numpy(img)[None][None]/255.`）。模型 forward 内没有任何归一化/clamp（`loftr.py:39-75`，backbone 直接 conv+BN）。
- resize/pad：**模型内部不做**。官方 loader `read_megadepth_gray` 做「长边 resize → 向下取整到 df 倍数 → cv2.resize → 可选右下 zero-pad 成方形」（`dataset.py:94-125`，`get_resized_wh :55-61`，`get_divisible_wh :64-69`，`pad_bottom_right :72-89`）。
- 尺寸约束：H、W 必须是 8 的倍数（notebook cell 9 注释 "input size shuold be divisible by 8"）；两图可不同尺寸（`loftr.py:45-49` 分支），同尺寸时拼 batch 过 backbone（BN 一起跑，eval 下无影响）。
- mask：可选 `mask0/mask1`，**是 1/8 粗分辨率**（dataset 用 `F.interpolate(scale_factor=0.125, mode='nearest')` 生成，`src/datasets/megadepth.py:119-125`），而 `loftr.py:35` 注释写 `(N,H,W)` 容易误导。只用于 padding；我们 512² 无需 pad → 不传。
- 小坑：`read_megadepth_gray(padding=False)` 会在 `torch.from_numpy(None)` 处报错（`dataset.py:120-123`），所以官方 loader 只能配 padding=True 用。

### 1.3 喂非 [0,1] / 非自然分布数据
- 没有 clamp、没有 uint8 cast（forward 路径全 float）。z-score 的负值/大于 1 的值会原样进 conv（读到：无任何处理）。
- `cv2.resize` 对 float32 可用；只有官方 loader 里的 `cv2.imread` 假设读文件（8-bit）。
- 推断：模型只见过 `/255` 的自然灰度，输入均值≈0.4、全非负；直接喂 z-score（均值 0、有负值）属于分布外，BN 统计也不匹配。按「情形 A」应把 SAR 先映射成 uint8 再 `/255`。

### 1.4 输出
- `mkpts0_f`、`mkpts1_f`：`(M,2)` float tensor，`(x,y)`；`mconf (M,)`；`m_bids (M,)` 批索引（`coarse_matching.py:253-259`，`fine_matching.py:71-74`）。
- 坐标系：**模型输入张量的网格**。若 batch 带 `scale0/scale1`（`[w/w_new, h/h_new]`，`dataset.py:114`），则自动乘回原图尺度（`coarse_matching.py:242-250`，`fine_matching.py:68`）；notebook 路径不带 scale → 坐标在 resize 后的图上（cell 9）。
- 像素约定：粗匹配点 = `网格索引 × 8`（`coarse_matching.py:245-250`），没有 +3.5/+4 的中心偏移；训练监督同样用 `scale*grid`（`src/loftr/utils/supervision.py:49-50`），并用标准内参投影 → 推断：输出就是「整数 = 像素中心、0-based」约定，无需 ±0.5。
- **亚像素只在 image1 侧**：`mkpts0_f = mkpts0_c`（image0 永远落在 8 px 网格上），`mkpts1_f = mkpts1_c + coords_normed*(W//2)*scale`，W=5、scale=2 → ±4 px 内亚像素（`fine_matching.py:67-69`；`fine_preprocess.py:13,40-43`）。
- 小 bug：`fine_matching.py:68` 判断 `'scale0' in data` 却用 `scale1`（同尺度时无影响）。
- 数量上限（推断）：512² → 64×64 粗网格，去掉 2 格边界 → 最多 60×60=3600 对（互为最近邻，`coarse_matching.py:175-195`）。实际数量取决于 conf>0.2 的比例，跨模态通常远低于此（推断，未实测）。

### 1.5 环境
- 官方：python 3.8、pytorch 1.8.1、cudatoolkit 10.2（`environment.yaml`）；kornia==0.4.1、einops==0.3.0、opencv 4.4.0.46、pytorch-lightning 1.3.5（`requirements.txt`）。纯推理只需 `torch einops yacs kornia`（README.md:54）。无编译算子。
- 现代环境风险（推断+读到）：
  - `torch>=2.6` 的 `torch.load` 默认 `weights_only=True`，PL checkpoint 里有非张量对象，可能需要 `weights_only=False`（推断）。
  - 推理路径用到 `kornia.geometry.subpix.dsnt`、`kornia.utils.grid.create_meshgrid`（`fine_matching.py:5-6`），新版 kornia 仍有这两个路径（推断，需实装验证）。
  - `np.bool` 只在评测 `src/utils/metrics.py:128`，推理路径不受影响。
  - 替代：kornia 自带 `kornia.feature.LoFTR`（README.md:31-44），但那是另一份实现和权重下载路径，**不等同于官方仓库路径**。

### 1.6 适配器草图
```python
import numpy as np, torch
from copy import deepcopy
from src.loftr import LoFTR, default_cfg     # LoFTR 仓库根目录在 sys.path

cfg = deepcopy(default_cfg)                  # temp_bug_fix=False，配 outdoor_ds.ckpt
m = LoFTR(cfg); m.load_state_dict(torch.load("weights/outdoor_ds.ckpt", weights_only=False)['state_dict'])
m = m.eval().cuda()

def loftr_match(img0_u8: np.ndarray, img1_u8: np.ndarray):
    """img*: (512,512) uint8。SAR 需先由调用方映射到 uint8（情形 A）。"""
    t = lambda a: torch.from_numpy(a).float()[None, None].cuda() / 255.
    batch = {'image0': t(img0_u8), 'image1': t(img1_u8)}   # 512 是 8 的倍数，不 resize、不 pad
    with torch.no_grad():
        m(batch)
    return batch['mkpts0_f'].cpu().numpy(), batch['mkpts1_f'].cpu().numpy(), batch['mconf'].cpu().numpy()
```
- 不缩放即为原 512 网格 0-based 像素中心坐标。若要走官方测试分辨率（长边 840），用 `cv2.resize` 到 840 并在 batch 里放 `scale0/scale1 = torch.tensor([[512/840, 512/840]])`，输出即自动回到 512 网格（`coarse_matching.py:243-244`）。选哪个分辨率属于实验设计决定。

- 注意（推断）：官方「乘 scale」回原图是 `x_orig = x_new * s`，而 `cv2.resize` 的几何是像素中心对齐，严格应为 `x_orig = (x_new+0.5)*s − 0.5`。两者差 `0.5*(s−1)`，840→512 时约 −0.2 px；不缩放则无此问题。

---

## 2. XoFTR（`e9635d8`；上游 HEAD `e0fbea4`）

### 2.1 最小官方推理路径
两条官方路径，都在 notebook 里：
- **路径 W（包装器）**：`DataIOWrapper(XoFTR(config['xoftr']), config=config['test'], ckpt=...)`，然后 `from_paths(p0, p1)` 或 `from_cv_imgs(img0, img1)`（`notebooks/xoftr_demo.ipynb` cell 3-4；`src/utils/data_io.py:21-109`）。官方评测 `test_relative_pose.py:22,29` 也走这个包装器。
- **路径 M（裸模型）**：`matcher = XoFTR(config=config["xoftr"]); matcher.load_state_dict(torch.load(ckpt)['state_dict'], strict=True)`，自己构造 batch 调 `matcher(batch)`（`xoftr_demo.ipynb` cell 8、10；`xoftr_demo_batch.ipynb` cell 3-5）。
- 配置：`get_cfg_defaults(inference=True)` → `lower_config`（cell 3）。`inference=True` 会把 COARSE/MATCH_COARSE/FINE 的 INFERENCE 置 True（`src/config/default.py:199-203`），切到省显存的推理分支（`coarse_matching.py:108-110`）。**必须传 True**，否则走训练分支（`get_coarse_match_training`）。
- 关键旋钮与默认：`MATCH_COARSE.THR=0.3`、`BORDER_RM=2`（`default.py:31-32`）；`FINE.THR=0.1`、`FINE.DENSER=False`（`:40,43`）；`TEST.IMG0_RESIZE=IMG1_RESIZE=640`（长边）、`DF=8`、`PADDING=False`、`COARSE_SCALE=0.125`（`:189-193`）。
- 权重：`weights/weights_xoftr_640.ckpt`（及 840 版），只在 Google Drive（README.md:24）。`load_state_dict` 同样剥 `matcher.` 前缀（`src/xoftr/xoftr.py:90-94`）。

### 2.2 最窄可调用边界的输入
- 模型边界：`(N,1,H,W)` float tensor（`xoftr.py:27-28`；backbone `nn.Conv2d(1,...)`，`src/xoftr/backbone/resnet.py:59`）。
- **重要，与 LoFTR 不同：模型 forward 内部做逐图 z-score**：`image = (image - mean) / (std + 1e-6)`，mean/std 在每张图的 H、W 上计算（`xoftr.py:39-47`）。因此对每张图各自的全局线性灰度变换（`a·x+b`，a>0）**不变**。loader 里的 `/255`（`data_io.py:76`；`src/utils/dataset.py:227`）只是约定，实际不影响结果（推断：仅 fp 误差）。这一点和 README §2 把 XoFTR 归到「情形 A（不再归一化）」**不一致**，应归入「情形 C（逐图标准化）」。
- 包装器输入：`from_cv_imgs` 接受 numpy 图（HxW 或 HxWx3 BGR）。3 通道时 `cv2.cvtColor(BGR2GRAY)`（`data_io.py:45-47`）→ 长边 resize 到 640（**512 会被放大到 640**）→ 向下取整到 8 的倍数 → `cv2.resize`（默认双线性）（`:57-66`）→ 可选 pad（默认关）→ `/255` 并**硬编码 `.cuda()`**（`:76`，CPU 上会失败）。
- 尺寸：H、W 为 8 的倍数（notebook cell 10 注释）；位置编码 `max_shape=(256,256)` → 每边 ≤2048（`src/xoftr/utils/position_encoding.py:11-14`）。两图可不同尺寸（`xoftr.py:49-56`；notebook cell 10 故意用 640×360 与 640×512）。
- mask：1/8 粗分辨率 bool，包装器用 `F.interpolate(scale_factor=0.125, nearest)` 生成（`data_io.py:113-119`）。注意 padding 时 z-score 的 mean/std 会把补的 0 也算进去（`xoftr.py:41-42` 不看 mask）。

### 2.3 喂非 [0,1] / 非自然分布数据
- 无 clamp、无 uint8 cast。模型内 z-score 抹掉全局尺度和偏移，所以直接喂我们 z-score 后的 SAR（float32）与 `/255` 的光学在数学上等价于各自再标准化一次（读到 `xoftr.py:41-47`；结论为推断）。
- 包装器里的 `cv2.cvtColor`/`cv2.resize` 支持 float32（推断：OpenCV 对 CV_32F 支持；float64 的 cvtColor 不支持）。所以 float32 单通道可以直接走 `from_cv_imgs`，只是 `/255` 无意义但无害。
- 但 dB 这种**非线性**映射仍会改变分布（推断），逐图 z-score 管不到。

### 2.4 输出
- 包装器：返回 dict，`mkpts0/mkpts1 (M,2)` numpy、`mconf`、`matches (M,4)`（`data_io.py:86-96`），已乘 `scale=[w/w_new, h/h_new]` 回到**原图**（`:67,87-88`）。
- 裸模型：`batch['mkpts0_f'], ['mkpts1_f'], ['mconf_f'], ['m_bids']`（注意置信度键名是 **`mconf_f`**，不是 LoFTR 的 `mconf`；`fine_matching.py:156-161`），坐标在模型输入网格；也支持 batch 里带 `scale0/scale1`（`fine_matching.py:130-131`）。
- 亚像素：**两侧都是亚像素**：`(窗口内离散位置 + 粗匹配×4 − W_f//2 + tanh(mlp)*0.5) × 2`（`fine_matching.py:117-151`）。粗匹配点同样是 `网格索引 × 8`、无中心偏移（`coarse_matching.py:288-296`）。
- 粗匹配语义与 LoFTR 不同：推理时是 **0→1 最近邻 ∪ 1→0 最近邻（OR），不是互为最近邻**（`coarse_matching.py:256-267`），一个粗格可以出多条匹配 → 匹配更多、外点也更多（推断）。细层默认每个粗匹配只保留窗口内置信度最高的一对（`fine_matching.py:91-94`）。
- 怪行为：若所有细层置信度都 ≤ `fine.thr`，代码**强行把 `mask[0,0,0]` 设 1、置信度设 1**，吐出一条假匹配（`fine_matching.py:87-89`）。下游应把「只有 1 条匹配」视为失败。
- 数量上限（推断）：640² 输入 → 80×80 粗网格，去边后 76² 个格子，OR 语义下最多约 2×5776 条。

### 2.5 上游 HEAD 相对 `e9635d8` 的变化
- `e0fbea4`（2025-08-21）只改 1 行：`src/utils/data_io.py:67` `dtype=np.float` → `np.float32`（`git diff e9635d8 origin/main` 读到）。没有其他改动，模型、配置、权重不受影响。
- 但 `notebooks/xoftr_demo_batch.ipynb` cell 2 里自带一份 `preprocess_image` 拷贝，仍是 `np.float`，上游没修；而且那份拷贝在 `padding=False` 时会因 `mask=None` 调 `F.interpolate` 报错。

### 2.6 环境
- 官方：python 3.8、pytorch 2.0.1、pytorch-cuda 11.8（`environment.yaml`）；numpy==1.23.1、opencv 4.5.1.48、einops 0.3.0、kornia 0.4.1、pytorch-lightning 1.3.5（`requirements.txt`）。无编译算子。
- 推理路径不 import kornia（只有 `src/xoftr/utils/supervision.py:7,10` 用）；用了 `torch.div(..., rounding_mode='trunc')`（需 torch≥1.8）。
- 破坏点：`e9635d8` 的 `data_io.py:67` 在 numpy≥1.24 下 `AttributeError: np.float`（官方钉的 1.23.1 不会触发）；`torch.load(ckpt)` 无 `map_location`/`weights_only`（`data_io.py:40`），在 torch≥2.6 可能要 `weights_only=False`（推断）；`data_io.py:29` 设备自动选择但 `:76` 硬编码 `.cuda()`。`np.bool` 只在评测 `src/utils/metrics.py:146`。

### 2.7 适配器草图（裸模型，避开 np.float 与 .cuda 问题，保留官方预处理）
```python
import numpy as np, torch, cv2
from src.xoftr import XoFTR
from src.config.default import get_cfg_defaults
from src.utils.data_io import lower_config

cfg = lower_config(get_cfg_defaults(inference=True))
m = XoFTR(cfg['xoftr'])
m.load_state_dict(torch.load("weights/weights_xoftr_640.ckpt", map_location='cpu', weights_only=False)['state_dict'], strict=True)
m = m.eval().cuda()

def xoftr_match(img0: np.ndarray, img1: np.ndarray, long_side=640):
    """img*: (512,512) 单通道；uint8 或 float32 均可（模型内逐图 z-score）。"""
    def prep(a):
        a = a.astype(np.float32)
        h, w = a.shape; s = long_side / max(h, w)
        wn, hn = (int(round(w*s)) // 8 * 8, int(round(h*s)) // 8 * 8)
        a = cv2.resize(a, (wn, hn))                                  # 同 data_io.py:57-66
        return torch.from_numpy(a)[None, None].cuda() / 255., np.array([w/wn, h/hn], np.float32)
    (t0, s0), (t1, s1) = prep(img0), prep(img1)
    batch = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        m(batch)
    k0 = batch['mkpts0_f'].cpu().numpy() * s0; k1 = batch['mkpts1_f'].cpu().numpy() * s1   # 同 data_io.py:87-88
    return k0, k1, batch['mconf_f'].cpu().numpy()
```
- `long_side=None` 即不缩放（512 已是 8 的倍数），坐标直接在 512 网格。官方默认是 640（放大 1.25×）；放大时回乘 `s` 会有与 LoFTR 同样的约 −0.1 px 中心约定偏差（`0.5*(s−1)`，s=0.8，推断）。

---

## 3. SuperPoint + SuperGlue（`ddcf11f`）

### 3.1 最小官方推理路径
- 类：`models.matching.Matching(config)`，内部串 `SuperPoint` + `SuperGlue`（`models/matching.py:49-84`）。脚本入口 `match_pairs.py`：构造 config（`match_pairs.py:181-193`）→ `read_image`（`:262-265`）→ `pred = matching({'image0': inp0, 'image1': inp1})`（`:274`）→ 取 `keypoints0/1`、`matches0`、`matching_scores0`（`:275-277`）→ `valid = matches > -1; mkpts0 = kpts0[valid]; mkpts1 = kpts1[matches[valid]]`（`:286-289`）。
- 权重：**在 git 仓库内**（`models/weights/superpoint_v1.pth` 5.2 MB、`superglue_indoor.pth`、`superglue_outdoor.pth` 各 48 MB，本地克隆已确认是实体文件而非 LFS 指针），在各自构造函数里 `torch.load` 自动加载（`models/superpoint.py:136-137`；`models/superglue.py:223-226`）。无需下载。
- 配置默认值（代码 vs 脚本 vs README 推荐的「outdoor」）：

| 旋钮 | 类默认 | `match_pairs.py` 默认 | README 室外推荐（README.md:217-220） |
|---|---|---|---|
| SuperGlue `weights` | `'indoor'`（`superglue.py:199`） | `'indoor'`（`:93-94`） | `outdoor` |
| `resize` | — | `[640, 480]`（`:84-85`） | `1600`（长边） |
| `max_keypoints` | `-1` 不限（`superpoint.py:107`） | `1024`（`:96-97`） | `2048` |
| `nms_radius` | 4（`superpoint.py:105`） | 4（`:103-104`） | 3 |
| `keypoint_threshold` | 0.005（`superpoint.py:106`） | 0.005（`:100-101`） | 0.005 |
| `sinkhorn_iterations` | **100**（`superglue.py:202`） | **20**（`:107-108`） | 20（未改） |
| `match_threshold` | 0.2（`superglue.py:203`） | 0.2（`:110-111`） | 0.2 |
| `resize_float` | — | False（`:89-90`） | **True** |
| `remove_borders` | 4（`superpoint.py:108`） | 不可改 | 4 |

- 许可证：学术/非营利、非商业研究专用（`LICENSE` 开头 3 行）。

### 3.2 最窄可调用边界的输入
- `Matching.forward` 接收 `{'image0','image1'}`：`(1,1,H,W)` float tensor（`superpoint.py:119` `nn.Conv2d(1, ...)`）。**图像张量必须一直留在 dict 里**：SuperGlue 用 `data['image0'].shape` 做关键点坐标归一化（`superglue.py:65-72,245-246`），所以即使自己提供关键点也要传图像（或至少同形状张量）。
- 值域：[0,1]，**归一化只在 loader**：`frame2tensor` 做 `frame/255.`（`models/utils.py:259-260`）。forward 里没有任何归一化（`superpoint.py:145-158` 直接 conv+ReLU）。
- 官方 loader `read_image`：`cv2.imread(GRAYSCALE)` → `process_resize`（长边或 WxH 或 −1 不变）→ `resize_float=True` 时先 `astype(float32)` 再 `cv2.resize`，否则在 uint8 上 resize 再转 float（`utils.py:263-282`；`process_resize :240-256`）。<160 或 >2000 px 只打印警告（`:251-254`）。
- 尺寸：无硬性整除要求。3 次 2×2 池化后 score 图被 reshape 成 `(h*8, w*8)`（`superpoint.py:164-166`），H、W 不是 8 的倍数时右/下余数区域不出点（推断）。512 无此问题。
- 批处理：`Matching` 只支持 batch=1 或每张图关键点数相同（`matching.py:72-79` 的注释与 `torch.stack`）。
- 不支持 mask。

### 3.3 喂非 [0,1] / 非自然分布数据
- 无 clamp、无 uint8 cast（`frame2tensor` 只是 `/255`）。`resize_float=False` 时 `cv2.resize` 在原 dtype 上做，uint8 会有取整（`utils.py:273-274`）。
- **对输入尺度敏感**（推断，基于代码结构）：检测是「65 通道 softmax 后取绝对阈值 `keypoint_threshold=0.005`」（`superpoint.py:161-171`），没有输入归一化，所以整体放大/缩小灰度会改变响应强度和关键点数量。喂 z-score（有负值）属于分布外。SAR 侧应按官方约定映射到 uint8 再 `/255`。

### 3.4 输出
- `pred` dict，每项是长度 = batch 的 list/张量：`keypoints0 [(K0,2)]`（x,y）、`scores0`、`descriptors0 (256,K0)`、`matches0 (K0,)`（−1 = 无匹配）、`matching_scores0`，另一侧对称（`superpoint.py:198-202`；`superglue.py:280-285`）。
- **关键点是整数像素**（`torch.nonzero` 取 score 图上的像素索引，再 `flip` 成 (x,y)，`superpoint.py:170-187`）→ **无亚像素**。
- 像素约定：score 图与输入同分辨率，索引 i ↔ 像素 i；描述子采样用 `kp - s/2 + 0.5`（`superpoint.py:83`），即把 8×8 cell 中心放在 `8i+3.5` → 说明代码采用「整数 = 像素中心」约定（推断，与 0-based 像素中心一致）。
- 坐标系：**resize 后的网格**。`match_pairs.py` 计算的 `scales0/1` 只用来缩放内参（`:299-300`），**不会把关键点映射回原图**。
- 匹配语义：Sinkhorn 后互为最大 + `mscore > match_threshold`（`superglue.py:267-278`）。数量 ≤ min(K0,K1) ≤ `max_keypoints`（1024/2048）。
- 边界：离边 4 px 以内的点被丢弃（`superpoint.py:65-70,176-178`）。

### 3.5 环境
- README：Python ≥3.5、PyTorch ≥1.1（README.md:27-28）；`requirements.txt`：torch≥1.1.0、opencv-python==4.1.2.30、numpy≥1.18.1、matplotlib。纯 PyTorch，无编译算子，**可在 CPU 跑**（`match_pairs.py:179` 有 `--force_cpu`）。
- 现代环境：仓库 `.py` 中没有 `np.float/np.int/np.bool`（grep 读到为空）。`sample_descriptors` 用字符串比较 `torch.__version__ >= '1.3'` 决定 `align_corners`（`superpoint.py:87`）；torch≥1.10 的 `__version__` 是 `TorchVersion`，按版本语义比较，2.x 下为 True（推断）。`torch.load` 加载的是纯 state_dict，torch≥2.6 的 `weights_only=True` 默认值应能通过（推断）。
- 包导入：`models/utils.py` 顶层 import matplotlib 和 threading（`utils.py:45-53`），无头服务器需要 `MPLBACKEND=Agg`（推断）；只用 `models.matching` 则不触发 utils。

### 3.6 适配器草图
```python
import numpy as np, torch, cv2
from models.matching import Matching          # SuperGlue 仓库根目录在 sys.path

cfg = {'superpoint': {'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': 2048},
       'superglue':  {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}}
mt = Matching(cfg).eval().cuda()

def spsg_match(img0_u8: np.ndarray, img1_u8: np.ndarray, resize=-1):
    """img*: (512,512) uint8。resize=-1 不缩放；官方室外推荐 1600（对 512 是放大 3.125×）。"""
    def prep(a):
        h, w = a.shape
        if resize == -1: wn, hn = w, h
        else: s = resize / max(h, w); wn, hn = int(round(w*s)), int(round(h*s))
        a = cv2.resize(a.astype('float32'), (wn, hn))          # resize_float=True，同 utils.py:271-272
        return torch.from_numpy(a/255.).float()[None, None].cuda(), np.array([w/wn, h/hn])
    (t0, s0), (t1, s1) = prep(img0_u8), prep(img1_u8)
    with torch.no_grad():
        p = mt({'image0': t0, 'image1': t1})
    k0, k1 = p['keypoints0'][0].cpu().numpy(), p['keypoints1'][0].cpu().numpy()
    m, c = p['matches0'][0].cpu().numpy(), p['matching_scores0'][0].cpu().numpy()
    v = m > -1
    # 回原图：官方脚本不做；按 cv2.resize 的中心对齐几何应为 (x+0.5)*s-0.5
    return (k0[v] + .5) * s0 - .5, (k1[m[v]] + .5) * s1 - .5, c[v]
```

---

## 4. RIFT2（LJY-RS/RIFT2-multimodal-matching-rotation，`0e980ce4124d2abd62727dcf040198db0a18b269`）

仓库很小：27 个被跟踪文件，全部是 `.m` 源码加 5 组样例图，**没有 mex、p-code、dll 或 exe**（`git ls-files` 读到）。README 只有引用信息，没有 LICENSE 文件。

### 4.1 最小官方推理路径（`demo_RIFT2.m` 是脚本，不是函数）
```
im = im2uint8(imread(path))                         demo_RIFT2.m:6-7
灰度 → 复制成 3 通道                                   :9-15
[key,m,eo] = FeatureDetection(im, 4, 6, 5000)        :19-20
kpts = kptsOrientation(key, m, 1, 96)                :23-24
des  = FeatureDescribe(im, eo, kpts, 96, 6, 6)       :27-28
[idx,~] = matchFeatures(des1',des2','MaxRatio',1,'MatchThreshold',100)   :31
去重：unique(matchedPoints2,'rows')                   :35-36
H = FSC(p1, p2, 'similarity', 3)                     :40
按 H 重投影误差 E<3 选内点                              :41-47
showMatchedFeatures / image_fusion（画图）             :50-53
```
- 各函数签名：
  - `[kpts,m,eo] = FeatureDetection(im, s, o, npt)`：3 通道时 `rgb2gray`（`FeatureDetection.m:3-5`）→ `phasecong3(im,s,o,3,'mult',1.6,'sigmaOnf',0.75,'g',3,'k',1)`（`:7`）→ 最大矩 `m` 做 min-max 归一化到 [0,1]（`:8`）→ `detectFASTFeatures(m,'MinContrast',1e-4,'MinQuality',1e-4)`，`selectStrongest(npt)`（`:10-11`）→ 返回 2×N（x,y）double（`:12`）。
  - `kpts = kptsOrientation(key, im, is_ori, patch_size)`：返回 3×K（x,y,角度°）；**坐标被 `round` 成整数**（`kptsOrientation.m:20-21`）；一个点可以有多个主方向（`:35-38`，峰值 >0.8×最大值 `kptsOrientation.m:5`，`orientation.m:7,21-29`）。
  - `des = FeatureDescribe(im, eo, kpts, patch_size, no, nbin)`：用 log-Gabor 幅值的最大索引图 MIM（`FeatureDescribe.m:5-11`），旋转取 patch、6×6 格 × 6 bin = 216 维，L2 归一化（`:14-40`）；**用了 `parfor`**（`:14`）。
  - `[solution,rmse,cor1_new,cor2_new] = FSC(cor1, cor2, change_form, error_t)`：`change_form ∈ {'similarity','affine','perspective'}`（`FSC.m:1-13`），迭代上限 10000（`:14-18`），返回 3×3 矩阵，把 **cor1（图 1）映射到 cor2（图 2）**（`:51-53,72-78`），内点去重后用 LSM 重新拟合（`:88-99`；`LSM.m`）。

### 4.2 输入
- 最窄边界是 MATLAB 数组，不是文件：`FeatureDetection` 接受 H×W 或 H×W×3 的任意数值数组（`phasecong3` 内部 `double(im)`，`phasecong3.m:483-490`）。文件读取只在 demo 脚本里。
- demo 的转换链：`imread` → **`im2uint8`**（`demo_RIFT2.m:6-7`）→ 灰度复制成 3 通道（`:9-15`）→ `rgb2gray` 转回来（`FeatureDetection.m:3-5`）→ `phasecong3` 里转 double（不除以 255，`phasecong3.m:488-490`）。
- 没有 resize、pad、crop；两图**可以不同尺寸**（各自独立处理）。无 mask 输入。
- **边界死区**：`kptsOrientation` 要求以点为中心的 96×96 窗口完全在图内，否则丢弃（`kptsOrientation.m:24-31`）。对 512×512 来说，只有 x、y ∈ [49, 464]（1-based）的点能留下，**即只有中间 416×416（约 66% 面积）能出匹配**。patch_size 是 demo 参数（`demo_RIFT2.m:23-24,27-28` 的 96）。

### 4.3 喂 float / 非自然分布数据
- **`im2uint8` 是真正的坑**：对 double/single 输入，它假设值域 [0,1]，<0 截成 0、>1 截成 255（MATLAB 语义，推断）。若把 z-score 后的 SAR 直接喂给 demo 链，大半像素会被截断。对 uint8 输入则是空操作。
- 绕过 demo 直接调 `FeatureDetection` 时，不会截断：`phasecong3` 只做 `double(im)`（`phasecong3.m:488-490`）。相位一致性是能量比值，log-Gabor 没有 DC 分量（推断），噪声阈值按最小尺度响应的中位数自适应估计（`noiseMethod=-1`，`phasecong3.m:259,421`），所以对全局线性灰度变换近似不变（推断）。`m` 随后又做了 min-max（`FeatureDetection.m:8`）。
- `kptsOrientation` 用的是 `m`（已归一化），`FeatureDescribe` 用的是 `eo` 的最大索引（与幅值尺度无关）→ 描述子也近似尺度不变（推断）。
- 结论（推断）：给 RIFT2 喂 uint8（按官方链）和喂 float（跳过 `im2uint8`）结果应接近；**但若保留 demo 的 `im2uint8`，float 必须先在 [0,1]**。

### 4.4 输出与匹配
- **自己做匹配**：`matchFeatures(...,'MaxRatio',1,'MatchThreshold',100)`（`demo_RIFT2.m:31`）等于关掉比值检验和阈值，每个图 1 描述子取最近邻（默认 SSD、非 unique，推断自 MATLAB 默认值）→ 原始匹配数 ≈ 图 1 关键点×方向数（上限 5000×多方向，再被边界死区削减）。
- 然后按图 2 坐标去重（`:35-36`），**再自己做 FSC**，模型是 **similarity（不是 affine）**、阈值 3 px（`:40`），最后用 `E<3` 重算内点（`:41-47`）。
- 所以 demo 同时拥有三种输出：原始匹配（`matchedPoints1/2`，`:33-36`）、内点（`cleanedPoints1/2`，`:46-47`）、变换 `H`（3×3，图 1→图 2）。**若我们要统一 RANSAC，应取「FSC 之前」的 `matchedPoints1/2`**。
- 坐标：MATLAB **1-based**，像素中心为整数（`detectFASTFeatures` 的 Location，推断），并被 `round` 成整数（`kptsOrientation.m:20-21`）→ **无亚像素**。转成我们的约定：减 1。
- FSC 的坑：
  - 采样 `a = floor(1+(M-1)*rand(1,n))` 只能取到 1…M−1，**第 M 个点永远不会被抽为最小样本**（`FSC.m:28`）。
  - 当 M ≤ 2（similarity）或所有候选点坐标重复时，`while(1)` 找不到合法样本 → **死循环**（`FSC.m:27-48`）。适配器必须在调用前判 M。
  - `rand` 没有设种子（`FSC.m:25` 被注释），结果不可复现；需要在外面 `rng(seed)`。

### 4.5 环境
- 需要的 MATLAB 工具箱（推断自函数名）：
  - Image Processing Toolbox：`im2uint8`、`rgb2gray`、`imfilter`（`kptsOrientation.m:8-9`）、`imresize`（`FeatureDescribe.m:21`）、`strel/getnhood`（`orientation.m:3-4`）、`im2double`（`extract_patches.m:3`）、`maketform/imtransform`（`image_fusion.m:18-21`）。
  - Computer Vision Toolbox：`detectFASTFeatures`（`FeatureDetection.m:10`）、`matchFeatures`（`demo_RIFT2.m:31`）、`showMatchedFeatures`（`:50`）。
  - Parallel Computing Toolbox：可选；没有时 `parfor` 串行执行，有时会自动开 parpool（启动慢）。
  - `hist`（base MATLAB，已不推荐但仍可用）。
- 无 mex、无二进制，Linux MATLAB 可跑。**不能用 Octave 直接替代**（推断：Octave 没有 `detectFASTFeatures`/`matchFeatures`）。
- 无头：核心函数没有 GUI；只有 demo 的 `figure/showMatchedFeatures/image_fusion`（`:50-53`）需要显示。`demo_RIFT2.m:1` 的 `clc;clear;close all` 和 `addpath <数据文件夹>`（`:2`）也要去掉。封装成函数后可用 `matlab -batch` 运行（推断）。

### 4.6 适配器草图
MATLAB 端（新写的 `rift2_run.m`，只重排官方调用，不改参数）：
```matlab
function rift2_run(in_mat, out_mat, seed)
S = load(in_mat);                          % S.im1, S.im2: uint8 H×W（调用方已映射到 uint8）
im1 = im2uint8(S.im1); im2 = im2uint8(S.im2);          % 同 demo_RIFT2.m:6-7（uint8 输入时为空操作）
im1 = repmat(im1,[1 1 3]); im2 = repmat(im2,[1 1 3]);  % 同 :9-15
[key1,m1,eo1] = FeatureDetection(im1,4,6,5000); [key2,m2,eo2] = FeatureDetection(im2,4,6,5000);
k1 = kptsOrientation(key1,m1,1,96);  k2 = kptsOrientation(key2,m2,1,96);
d1 = FeatureDescribe(im1,eo1,k1,96,6,6); d2 = FeatureDescribe(im2,eo2,k2,96,6,6);
ip = matchFeatures(d1',d2','MaxRatio',1,'MatchThreshold',100);
k1 = k1'; k2 = k2';
p1 = k1(ip(:,1),1:2); p2 = k2(ip(:,2),1:2);
[p2,IA] = unique(p2,'rows'); p1 = p1(IA,:);            % 同 :35-36
H = []; inl = false(size(p1,1),1);
if size(p1,1) >= 3                                     % 防 FSC 死循环
    rng(seed); H = FSC(p1,p2,'similarity',3);          % 官方后处理，仅作参考
    Y = H*[p1'; ones(1,size(p1,1))]; Y = Y(1:2,:)./Y(3,:);
    inl = (sqrt(sum((Y - p2').^2)) < 3)';
end
save(out_mat, 'p1','p2','H','inl');
end
```
Python 端：
```python
import numpy as np, scipy.io as sio, subprocess, tempfile, os
def rift2_match(img0_u8, img1_u8, rift2_dir, seed=0):
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, 'in.mat'), os.path.join(d, 'out.mat')
        sio.savemat(i, {'im1': img0_u8, 'im2': img1_u8})
        subprocess.run(['matlab', '-batch', f"addpath('{rift2_dir}'); rift2_run('{i}','{o}',{seed})"], check=True)
        r = sio.loadmat(o)
    return r['p1'] - 1.0, r['p2'] - 1.0      # 1-based → 0-based；FSC 前的原始匹配
```
- 每对图启动一次 MATLAB 开销很大（推断：约 10 s 量级），批量时应改成一次 `matlab -batch` 循环处理一个目录，或用 MATLAB Engine for Python。

---

## 5. 对比总表（影响 runner 架构的差异）

| 维度 | LoFTR | XoFTR | SP+SG | RIFT2 |
|---|---|---|---|---|
| 语言 / 调用方式 | Py，进程内 | Py，进程内 | Py，进程内 | **MATLAB，外部进程**（`matlab -batch` 或 Engine） |
| 最窄边界 | tensor `(N,1,H,W)` | tensor `(N,1,H,W)` | tensor `(1,1,H,W)` | MATLAB 数组 H×W(×3) |
| 通道 | 1 | 1 | 1 | 1 或 3（内部转灰度） |
| 官方值域 | [0,1]（loader `/255`） | [0,1]（loader `/255`） | [0,1]（loader `/255`） | uint8（`im2uint8`） |
| 模型内归一化 | **无** | **逐图 z-score**（`xoftr.py:39-47`） | **无** | min-max（仅 PC 图，`FeatureDetection.m:8`）；PC 本身近似对比度不变 |
| 对全局线性灰度变换 | 敏感 | 不变 | 敏感（绝对检测阈值） | 近似不变（推断） |
| float 越界会怎样 | 原样进网络，无 clamp | 被 z-score 吸收 | 原样进网络，无 clamp | demo 链经 `im2uint8` **截断到 [0,1]** |
| 官方尺寸 | 室外测试长边 840 + pad；notebook 不缩放 | 长边 640（512 被放大） | 室外长边 1600（512 被放大 3.125×）；脚本默认 640×480 | 不缩放 |
| 尺寸约束 | 8 的倍数；≤2048 | 8 的倍数；≤2048 | 无硬约束 | 无；96 px 描述窗 → 边界 48 px 死区 |
| mask | 可选，1/8 分辨率 | 可选，1/8 分辨率 | 无 | 无 |
| 输出坐标系 | 模型输入网格；带 `scale0/1` 时回原图 | 包装器回原图；裸模型在输入网格 | resize 网格（脚本不回乘） | 原图，**1-based** |
| 像素中心 | 整数（训练监督同约定） | 整数 | 整数（描述子采样 `+0.5` 约定） | 整数（1-based） |
| 亚像素 | **仅 image1**；image0 在 8 px 网格上 | 两侧都有 | 无（整数像素） | 无（`round`） |
| 匹配语义 | 互为最近邻 + conf>0.2 | 0→1 NN **或** 1→0 NN + conf>0.3；fine>0.1 | Sinkhorn 互为最大 + score>0.2 | NN（关掉比值检验），再 FSC(similarity) |
| 自带几何验证 | 无 | 无（notebook 自己用 MAGSAC） | 无（脚本只做位姿评估） | **有：FSC similarity + 3 px** |
| 置信度键 | `mconf` | `mconf_f` | `matching_scores0` | 无（只有距离 `matchmetric`） |
| 空结果 | 返回 0 条 | **强行返回 1 条 conf=1 的假匹配**（`fine_matching.py:87-89`） | 返回 0 条 | FSC 在 M≤2 时**死循环** |
| 权重 | GDrive（`outdoor_ds.ckpt`，需 `temp_bug_fix=False`） | GDrive（640/840 两版） | **仓库自带** | 无需 |
| 固定环境 | py3.8 / torch1.8.1 / cu10.2 | py3.8 / torch2.0.1 / cu11.8 / numpy1.23.1 | torch≥1.1，可 CPU | MATLAB + IPT + CVT（+PCT 可选） |
| 现代环境破坏点 | torch≥2.6 `weights_only`（推断） | `np.float`（上游已修）、硬编码 `.cuda()`、`weights_only`（推断） | 基本无 | 无 mex；需正版工具箱 |
| 确定性 | 确定 | 确定 | 确定 | FSC 未设种子，不确定 |
| 许可 | Apache-2.0 | Apache-2.0 | **学术非商业** | 无 LICENSE 文件 |

### 对 runner 架构的含义（推断）
1. **「SAR → 模型输入」映射必须是方法级配置**，不能全局统一：LoFTR、SP+SG 对灰度尺度敏感，需要一个明确的 dB→uint8 映射；XoFTR 对线性映射不敏感；RIFT2 如果保留 `im2uint8` 就要求 uint8 或 [0,1]。最简单的统一做法：runner 一律给方法 **uint8 单通道**（光学原样，SAR 用一个固定的 dB→uint8 映射），各适配器自己做 `/255`；XoFTR 和 RIFT2 对这个映射的线性部分不敏感，LoFTR、SP+SG 敏感 → 映射方式应作为消融项。
2. **分辨率策略是方法级参数**：三个 Python 方法的官方室外分辨率都**大于 512**（840/640/1600），即官方做法是放大。要不要照做需要明确决定；适配器应接收 `long_side`，并负责把坐标映射回 512 网格（统一用 `(x+0.5)*s−0.5`，而不是官方的 `x*s`）。
3. **适配器的输出契约**应统一为：`kpts0 (N,2), kpts1 (N,2)`（0-based、像素中心为整数、原 512 网格）+ `conf (N,)`（可为 None）+ 可选 `method_H`（仅 RIFT2 有）+ 元数据（是否亚像素、raw 还是 inlier）。统一 RANSAC 只吃 raw 匹配。
4. **健壮性**：runner 要处理 XoFTR 的「1 条假匹配」和 RIFT2 FSC 的死循环（适配器里判 M，外面再加超时）。
5. **进程模型**：RIFT2 必须跑在外部 MATLAB 进程里，且冷启动开销大 → runner 需要支持「批处理型」适配器（一次处理一个目录），而不只是「逐对调用」接口。
6. **环境隔离**：LoFTR（torch1.8/cu10.2）和 XoFTR（torch2.0/cu11.8）的官方环境互不兼容；如果要跑现代单一环境，需要实测验证（推断：两者代码都很薄，现代 torch 大概率能跑，主要改 `weights_only` 和 `np.float`）。

### 与 README（research-baselines）结论的出入
- README §2 把 XoFTR 归在「情形 A：÷255 之后不再归一化」。**代码显示 XoFTR 在 forward 里做逐图 z-score（`src/xoftr/xoftr.py:39-47`），应归入情形 C**。LoFTR 确实是情形 A。
- README 写「LoFTR / XoFTR：÷8（df=8）」属于情形 E：LoFTR 正确；XoFTR 的官方测试/包装器默认还会把长边 resize 到 640（`default.py:189-192`），即 README 情形 D 的描述，也正确。
- README 说 LoFTR 系坐标「先在 resize 后网格输出，再乘 scale 回原图」：XoFTR 包装器正确（`data_io.py:87-88`）；**LoFTR 官方 notebook/README 用法不会乘回**（没有 `scale0/1`），只有 dataset 路径会。
