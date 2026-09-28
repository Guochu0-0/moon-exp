# Group B 代码接口研究：MatchAnything（ELoFTR / RoMa）与 MINIMA（LoFTR / XoFTR / RoMa / SP+LG / ELoFTR）

> 目的：设计 baseline-runner 之前，精确到行地弄清每个匹配器官方代码"想被怎么调用"。
> 所有断言均附 `文件:行号`；「推断」为我的判断，已标出。

## 0. 仓库与版本核对

| 仓库 | 本地路径（`C:\Users\yougu\.claude-work\jobs\928dd2ed\tmp\repos\`） | commit | 完整性 |
|---|---|---|---|
| `zju3dv/MatchAnything` | `MatchAnything` | `8cd8c11` | 只有 README + teaser gif，**没有代码**；README 第 22 行指向 HF Space |
| HF Space `LittleFrog/MatchAnything` | `MA_space` | `6a7bcb5`（detached，工作区干净） | 代码在 `imcui/third_party/MatchAnything/`（下称 **`MA/`**）；RoMa 分叉在 `MA/third_party/ROMA/`；**权重不在仓库里**（Space 启动时 `gdown` 拉 Google Drive 的 `weights.zip`，`imcui/ui/app_class.py:26`） |
| `LSXI7/MINIMA` | `MINIMA` | `796e772` | 5 个子模块均已检出、有文件：`third_party/LightGlue@edb2b83 (v0.2)`、`LoFTR_minima@9ba0871`、`RoMa_minima@0d3fd22`、`XoFTR@e9635d8`、`glue_factory_minima@76e7209`（`git submodule status` 无 `-`/`+` 前缀） |

路径约定：`MA/` = `MA_space/imcui/third_party/MatchAnything/`；`MAR/` = `MA/third_party/ROMA/roma/`；`MN/` = `MINIMA/`。

**通用坐标原则（后文反复用到）**：若两图同尺寸、同缩放，且一种实现对两侧坐标都加了同一个常量偏移 c（比如 RoMa 的 +0.5，或 LoFTR 系的 `x*s` 与 `(x+0.5)*s-0.5` 之差），那么对仿射 `x1 = A x0 + t` 而言，只会把 t 带偏 `(A−I)c`；A 接近单位阵时可以忽略。但如果两侧缩放不同，或者直接拿匹配点去和 GT 点比较误差，偏移就不能忽略。所以适配层里仍应统一改成「像素中心在整数」的约定。

---

## 1. MatchAnything-ELoFTR（MA-ELoFTR）

### 1.1 官方最小推理路径 / 权重 / 配置
- 官方的 visible–SAR 评测命令（`MA/scripts/evaluate/eval_visible_sar.sh:14`）：
  `python tools/evaluate_datasets.py configs/models/eloftr_model.py --ckpt_path weights/matchanything_eloftr.ckpt --method matchanything_eloftr@-@ransac_affine --imgresize 832 --thr 0.05 --npe ...`
- 构建流程：
  1. `config = get_cfg_defaults()`，返回 `_CN.clone()`（`MA/src/config/default.py:340-344`）。
  2. `config.merge_from_file(eloftr_model.py)`（`MA/tools/evaluate_datasets.py:104-107`）。
  3. `config.METHOD = 'matchanything_eloftr'`（`:110`）。
  4. `--npe` 时 `LOFTR.COARSE.NPE = [832, 832, imgresize, imgresize]`（`:114-115`）。
  5. **路径里含 `visible_sar` 时强制 `DATASET.RESIZE_BY_STRETCH = True`**（`:117-118`）。
  6. `LOFTR.MATCH_COARSE.THR = args.thr`（`:120-121`）。
  7. `matcher = PL_LoFTR(config, pretrained_ckpt, test_mode=True).matcher`（`:123`），`matcher.eval().cuda()`。
- 权重加载：`torch.load(ckpt)['state_dict']`，然后 `load_state_dict(strict=False)`（`MA/src/lightning/lightning_loftr.py:73-76`）；键名前缀 `matcher.` 会被剥掉（`MA/src/loftr/loftr.py:226-230`）。
  - **`strict=False` 且只打一条日志**：键名对不上时会静默跑随机权重，适配层必须检查 missing / unexpected keys。
- 模型选择：`METHOD == "matchanything_eloftr"` 时用 `LoFTR(config=_config['loftr'])`（`lightning_loftr.py:61-62`）。
  - RepVGG 的 `switch_to_deploy` 只在 `METHOD == 'loftr'` 时执行（`:80`），所以 MA-ELoFTR 以训练形态的 RepVGG 运行（数值上等价，推断）。
- 关键配置（`MA/configs/models/eloftr_model.py`）：
  - 结构：RepVGG 主干（`:62`），`RESOLUTION=(8,1)`（`:23`），`FINE_WINDOW_SIZE=8`（`:24`），RoPE（`:112`），`ALIGN_CORNER=False`（`:45`）。
  - 精度：`LOFTR.FP16=False`（`:49`）。
  - 匹配：config 默认 `MATCH_COARSE.THR=0.1`（`:96`），**SAR 评测时被脚本改成 0.05**；`FORCE_NEAREST=True`（`:95`），`MATCH_FINE.LOCAL_REGRESS=True`（`:105`），`TOPK=1`（`:125-126`）。
  - `BORDER_RM=2`（`MA/src/config/default.py:99`）。
- 评测里的 RANSAC：
  - `cv2.estimateAffine2D(..., ransacReprojThreshold=3.0, confidence=0.99999, method=cv2.RANSAC)`（`MA/src/utils/metrics.py:180-185`；阈值来自 `--rigid_ransac_thr` 默认 3.0，`evaluate_datasets.py:77-78`）。
  - SAR 集走 `gt_match` 分支、`ransac_mode='affine'`（`evaluate_datasets.py:152-157`）。
- 随机种子：`pl.seed_everything(66)`（`evaluate_datasets.py:109`；`default.py:337`）。

### 1.2 输入（最窄可调用边界 = `matcher(batch)`，batch 是 dict）
- 必需键：`image0` / `image1` 为 `(N,1,H,W)` float 张量（`MA/src/loftr/loftr.py:45-59`）。
- 可选键：
  - `scale0` / `scale1`：`(N,2)`，含义 `[w/w_new, h/h_new]`。用于把坐标映射回原图（`coarse_matching.py:245-252`，`fine_matching.py:432-440`）。
  - `mask0` / `mask1`：padding 时的 1/8 掩码。
- 官方预处理（`read_megadepth_gray`，`MA/src/utils/dataset.py:207-265`；数据集以 `read_gray=True, normalize_img=False, df=None, img_padding=False, resize_by_stretch=True` 调用，`evaluate_datasets.py:137`）：
  1. 读图：`cv2.imread(path, IMREAD_GRAYSCALE)`，得到 uint8（`dataset.py:122-128`）。
  2. 缩放：**拉伸到 832×832**（`:229-230`），`cv2.resize` 默认 INTER_LINEAR（`:241`）。对 512² 输入是**放大 1.625 倍**。
  3. `scale = [w/w_new, h/h_new]`（`:242`）。
  4. `torch.from_numpy(image).float()[None] / 255`（`:254-255`）。**只做 ÷255，没有 mean/std**。
- 对尺寸的要求：
  - 832 能被 32 整除；两图尺寸不同时走分支，分别过 backbone（`loftr.py:61-79`）。
  - RoPE 的 NPE 把位置按 `832 / test_res` 缩放（`MA/src/loftr/utils/position_encoding.py:77-80`）。如果不拉伸、直接用 512 原尺寸，应设 `NPE=[832,832,512,512]`（推断）。
- 通道：1 通道灰度。

### 1.3 非自然分布 / 超出 [0,1] 的 float
- 官方路径只从文件读 uint8，所以输入天然在 [0,1]。
- 绕过 loader、直接喂张量时，模型里没有 clamp，也没有逐图归一化：z-score 后带负值的 SAR 会原样进网络，属于分布外输入（推断）。
- 因此适配层应先自己把 SAR 映射成 uint8 或 [0,1]，再按官方方式 resize。
- 另外，Space 的 hloc 包装器用 `astype("uint8")`（`imcui/hloc/matchers/matchanything.py:59-64`），超出 [0,255] 的值会**回绕**，不会饱和截断。

### 1.4 输出
- 结果写回 batch：`mkpts0_f`、`mkpts1_f`（M×2），`mconf`（M）。评测脚本直接取这三个（`evaluate_datasets.py:180-182`）。
- 坐标系：
  - **已经乘了 scale，回到原图坐标系**。粗匹配点为 `[i%w_c, i//w_c] * (hw_i/hw_c) * scale`（`coarse_matching.py:244-252`），细化量同样乘 scale（`fine_matching.py:432-440`）。
  - 评测直接拿它们和原图坐标系下的 GT 比较（`evaluate_datasets.py:186-188`）。
- 像素中心约定：
  - 直接用 `x*scale`，没有 ±0.5 修正。
  - 细化网格偏移为 `meshgrid(W) − W//2 + 0.5`（`fine_matching.py:426`）。
  - 静态看不出绝对的中心约定；两侧处理对称，适用第 0 节的通用坐标原则（推断）。
- 稀疏程度：半稠密，数量取决于 THR；有置信度 `mconf`。

### 1.5 统一接口
- 见第 3 节：`PL_LoFTR(config).matcher` 同时覆盖 ELoFTR 和 RoMa 两种模型，但**两者的输入键不同**。

### 1.6 环境
- MA 自带 `environment.yaml`：python 3.8、pytorch 1.12.1、torchvision 0.13.1、pytorch-cuda 11.7（`MA/environment.yaml:8-11`）。
- MA `requirements.txt`：`kornia==0.4.1`、`pytorch-lightning==1.3.5`、`opencv_python==4.4.0.46`、`einops==0.3.0`、`timm==0.6.7`、`yacs`、`pynvml`（`MA/requirements.txt:1-22`）。
- Space 实际运行版本：torch 2.8.0、torchvision 0.23.0、pytorch-lightning 1.4.9、numpy~=1.24（`MA_space/requirements.txt:14,26,34-35`）。**说明新 torch 也能跑**。
- 没有编译算子：
  - ELoFTR 的注意力走 `XAttention`，`FLASH_AVAILABLE=False` 被写死（`linear_attention.py:13`）。
  - xformers 只在 RoMa 的 DINOv2 里以 try/except 方式可选导入（`MAR/models/transformer/layers/attention.py:20-25`）。
- 没有发现 `np.float` 这类已删除的 numpy 别名。
- `lightning_loftr.py` 顶部无条件导入 ROMA 和 pynvml（`:27-29`），所以**即使只跑 ELoFTR，也要装 RoMa 的依赖**。
- 配置陷阱：`configs/models/*.py` 通过 `from src.config.default import _CN as cfg` **直接修改全局 `_CN`**（`eloftr_model.py:1`、`roma_model.py:1`）。同一进程里先后加载两个配置会互相污染（例如 RoMa 配置把 `LOFTR.FP16` 设为 True，`roma_model.py:22`；ELoFTR 配置又显式设回 False，`eloftr_model.py:49`）。建议**每个模型单独进程**，或在加载后显式重设关键项。

### 1.7 适配草图（保持官方 SAR 评测预处理）
```python
# sys.path 需含 MA/（evaluate_datasets.py:17 的做法）
import cv2, torch, numpy as np
from src.config.default import get_cfg_defaults
from src.lightning.lightning_loftr import PL_LoFTR

cfg = get_cfg_defaults(); cfg.merge_from_file(f"{MA}/configs/models/eloftr_model.py")
cfg.METHOD = "matchanything_eloftr"
cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]
cfg.LOFTR.MATCH_COARSE.THR = 0.05
net = PL_LoFTR(cfg, pretrained_ckpt=CKPT, test_mode=True).matcher.eval().cuda()   # 检查日志里的 missing keys!

def prep(u8):                       # u8: HxW uint8（SAR 先自行转成 uint8）
    r = cv2.resize(u8, (832, 832))  # 拉伸 + INTER_LINEAR，同 dataset.py:229-241
    return torch.from_numpy(r).float()[None, None].cuda() / 255

def match(opt_u8, sar_u8):
    H, W = opt_u8.shape
    s = torch.tensor([[W / 832, H / 832]], device="cuda")
    b = {"image0": prep(opt_u8), "image1": prep(sar_u8), "scale0": s, "scale1": s}
    with torch.no_grad(), torch.autocast("cuda", enabled=cfg.LOFTR.FP16):
        net(b)
    return b["mkpts0_f"].cpu().numpy(), b["mkpts1_f"].cpu().numpy(), b["mconf"].cpu().numpy()
    # 已在原 512 网格；如要严格改成像素中心约定：(p + 0.5) * s' - 0.5 与 p*s 相差 0.5*(s-1)≈-0.19px（两侧相同）
```

---

## 2. MatchAnything-RoMa（MA-RoMa）

### 2.1 官方最小推理路径 / 权重 / 配置
- 评测命令：`evaluate_datasets.py configs/models/roma_model.py --ckpt_path weights/matchanything_roma.ckpt --method matchanything_roma@-@ransac_affine --imgresize 832 --npe`（`eval_visible_sar.sh:17`）。
  - **没有传 `--thr`**，argparse 默认 0.1 写进了 `LOFTR.MATCH_COARSE.THR`（`evaluate_datasets.py:36-37,120-121`），但 RoMa 不用这个值。
- 构建：`METHOD == "matchanything_roma"` 时用 `MatchAnything_Model(config=_config['roma'], test_mode=True)`（`lightning_loftr.py:63-64`）。
  - 它调用 `experiments/roma_outdoor.get_model(...)`，传入 `coarse_resolution=(560,560)`、`symmetric=True`、`upsample_preds=True`、`attenuate_cert=True`，并设 `upsample_res=(864,864)`（`MAR/matchanything_roma_model.py:21-25`；默认值见 `MA/src/config/default.py:22-27`）。
- 权重：
  - 同样是 `state_dict` + `strict=False` + 剥 `matcher.` 前缀（`matchanything_roma_model.py:100-104`）。
  - **DINOv2 ViT-L 不在 ckpt 里**：构建时从 `dl.fbaipublicfiles.com` 通过 `torch.hub.load_state_dict_from_url` 下载（`MAR/models/encoders.py:88-89`）。它被放进 list 以隐藏参数（`:105`），所以不进 state_dict。**服务器离线时要预先放好 `~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth`**。
  - VGG19 的 CNN 分支用 `pretrained=pretrained_backbone=True`（`experiments/roma_outdoor.py:160-165`；`MAR/models/encoders.py:63` 为 `tvm.vgg19_bn(pretrained=pretrained)`），构建时会再下载 torchvision 的 ImageNet 权重（推断：随后会被 ckpt 覆盖，但下载这一步仍需要网络或缓存）。
- 关键配置：
  - `ROMA.RESIZE_BY_STRETCH=True`（`roma_model.py:2`）；`NORMALIZE_IMG=False`（`default.py:7`）；`MATCH_THRESH=0.0`（`default.py:5`）。
  - 采样：`threshold_balanced`、`N_SAMPLE=5000`、`THRESH=0.05`（`default.py:17-20`）。
  - 精度：`MODEL.AMP=True`，fp16（`default.py:15`）。评测外层还有 `torch.autocast(enabled=LOFTR.FP16=True)`（`roma_model.py:22`；`evaluate_datasets.py:177`）。

### 2.2 输入（最窄边界 = `matcher(batch)` 或 `model.model.self_inference_time_match(imA, imB, ...)`）
- 用到的键：`forward_inference` **只读 `image0_rgb_origin[0]` / `image1_rgb_origin[0]`**（`matchanything_roma_model.py:63-71`），完全不用数据集缩放过的 `image0`。所以 `--imgresize 832` 对 RoMa 实际不起作用。另外只取 `[0]`，batch 只能为 1。
- `*_rgb_origin` 的来源：`Image.open(path).convert("RGB")` → `(3,H,W)` → `/255.`（`common_data_pair.py:211-212`）。即 **3 通道 float [0,1]、原始尺寸**；灰度图被复制成 3 通道。
- 模型内部处理（`MAR/models/matcher.py:649-767`）：
  - 路径输入同样是 `convert("RGB")/255`（`:657-659`）。
  - 粗阶段：`resize_by_longest_edge_and_stretch(im, 560)`，拉伸成方形（`:668-670`）；实现为 `transforms.Resize(..., BICUBIC)`（`MAR/utils/utils.py:11-19,32-46`）。
  - 上采样阶段：用原图重新拉伸到 864（`:702-708`）。
  - `norm_img=False`，**不做 ImageNet 归一化**（`:675-677,713-715`）。
- 尺寸要求：`h_resized == w_resized`（`:666`）。560 和 864 都满足「14 与 8 的倍数」（`default.py:23-25`）。两图不需要同尺寸（各自拉伸）。

### 2.3 非自然分布
- torchvision 的 BICUBIC 对 float 张量有过冲，**不 clamp**。
- 没有 ImageNet 归一化，所以网络直接看到 [0,1] 的值。z-score 后的负值会直接进 DINOv2（fp16）。必须先映射到 [0,1]（推断）。

### 2.4 输出
- `sample(warp, cert, num=5000)`（`matchanything_roma_model.py:83`）：
  - 超过 0.05 的 certainty 被**置为 1**（`MAR/models/matcher.py:477-480`）。
  - 先按 certainty 多项式抽 4×5000 个，再按 KDE 做 balanced 抽样（`:485-504`）。**结果是随机的**，需要固定 `torch.manual_seed`。
  - 返回的 `mconf` 是被截断后的值（只有 1 或 ≤0.05）。
- 对称 warp：A→B 和 B→A 两个方向拼接后一起采样（`matcher.py:755-761`）。
- 坐标：
  - `to_pixel_coordinates`：`W/2*(x+1)`（`matcher.py:539-546`）。拉伸模式下 H、W 取原图尺寸（`matchanything_roma_model.py:73-75`），**所以已经回到原图坐标**。
  - 网格为 `linspace(-1+1/h, 1-1/h)`（`matcher.py:741-746`），像素 i 的中心对应 **i+0.5**。要得到 0-based 像素中心坐标需 **−0.5**。
- 过滤：
  - `certainty > MATCH_THRESH(0.0)`（`matchanything_roma_model.py:86`）。
  - 丢掉坐标 `> W-1` 的点（`:88`）。在 +0.5 约定下，这会不对称地砍掉最右 / 最下的半个像素带。
  - warp 越界（|x|>1）的点 certainty 置 0（`matcher.py:751-753`）。
- 稀疏程度：稠密 warp，最多采样 5000 对。

### 2.5 环境
- 同 1.6。额外需要 DINOv2 和 VGG19 的预训练权重（在线或缓存）。xformers 可选。

### 2.6 适配草图
```python
cfg = get_cfg_defaults(); cfg.merge_from_file(f"{MA}/configs/models/roma_model.py")
cfg.METHOD = "matchanything_roma"
net = PL_LoFTR(cfg, pretrained_ckpt=CKPT, test_mode=True).matcher.eval().cuda()

def to_rgb01(u8):                          # 同 common_data_pair.py:212（灰度 -> RGB 复制）
    return torch.from_numpy(np.repeat(u8[..., None], 3, -1)).permute(2, 0, 1).float() / 255.

def match(opt_u8, sar_u8, seed=66):
    torch.manual_seed(seed)
    b = {"image0_rgb_origin": to_rgb01(opt_u8)[None].cuda(),
         "image1_rgb_origin": to_rgb01(sar_u8)[None].cuda()}
    with torch.no_grad(), torch.autocast("cuda", enabled=True):   # LOFTR.FP16=True（roma_model.py:22）
        net(b)
    k0 = b["mkpts0_f"].float().cpu().numpy() - 0.5              # RoMa +0.5 约定 -> 像素中心在整数
    k1 = b["mkpts1_f"].float().cpu().numpy() - 0.5
    return k0, k1, b["mconf"].float().cpu().numpy()
```

---

## 3. MatchAnything 的「统一接口」能否复用

- **评测入口**：`PL_LoFTR(config, pretrained_ckpt, test_mode=True).matcher`（`lightning_loftr.py:48-78`）按 `config.METHOD` 分发到 ELoFTR 或 RoMa。之后统一调用 `matcher(batch)`，并统一从 `batch['mkpts0_f'/'mkpts1_f'/'mconf']` 读结果（`evaluate_datasets.py:178-182`）。
- **两种模型要的 batch 键不一样**：
  - ELoFTR 要 `image0/1`：灰度、已缩放、(N,1,H,W)，再加上 `scale0/1`。
  - RoMa 只要 `image0_rgb_origin/1`：RGB、原尺寸、(1,3,H,W)。
  - 官方数据集会把两套键都填好（`common_data_pair.py:100-104,211-212`），所以脚本能对两者一视同仁。我们的适配层也可以照做：同时构造这两套键，然后统一调用 `matcher(batch)`。
- **Space 的 hloc 包装器 `imcui/hloc/matchers/matchanything.py` 不适合用作评测**：
  - ELoFTR 在这里**不缩放、不拉伸**，按 df=32 取整后补边成方形（`:67,100-108`）。
  - UI 端的预处理默认 `force_resize` 到 640×480（`imcui/hloc/match_dense.py:55-61,74-80`；`match()` 里 `:892-896`）。
  - 阈值与评测不同（ELoFTR 0.001、RoMa 0.1，`match_dense.py:53,72`）。
  - 这套是 demo 行为，与 SAR 评测约定（832 拉伸、thr 0.05）**不一致**。
- **结论**：可以复用 `PL_LoFTR(...).matcher` 这一个构造器加上「同时填两套键」的 batch 约定，写一个 MA 适配器同时覆盖两种模型。不要复用 hloc 包装器。

---

## 4. MINIMA 的共同外壳：`load_model` + `DataIOWrapper`

### 4.1 工厂函数
- `load_model(method, args, use_path=True, test_orginal_megadepth=False)`（`MN/load_model.py:158-164`）用 `eval(f"load_{method}")` 分发。
  - `use_path=True` 时返回 `matcher.from_paths(img0_pth, img1_pth)`。
  - `use_path=False` 时返回 `matcher.from_cv_imgs(img0, img1)`。
- CLI 支持的方法只有 `xoftr`、`sp_lg`、`loftr`、`roma`（`:167-170`）。**没有 eloftr**，全仓库里也找不到 ELoFTR 的加载代码（`grep -i eloftr` 只命中 `README.md:123,129`）。
- 各方法的 `args` 字段与默认值（`:173-191`）：
  - loftr：`ckpt=./weights/minima_loftr.ckpt`，`thr=0.2`。
  - xoftr：`match_threshold=0.3`，`fine_threshold=0.1`，`ckpt=./weights/weights_xoftr_640.ckpt`（**默认是 XoFTR 原版权重名，不是 `minima_xoftr.ckpt`**）。
  - sp_lg：`ckpt=./weights/minima_lightglue.pth`。
  - roma：`ckpt=./weights/minima_roma.pth`，`ckpt2='large'`。
- 所有 wrapper 返回同一种 dict：`{'matches': N×4, 'mkpts0', 'mkpts1', 'mconf', 'img0', 'img1', 'match_time'}`（例如 `MN/src/utils/data_io_loftr.py:88-96`）。**坐标已乘回原图尺度**（`:86-87`）。

### 4.2 已确认的坏点（代码级）
1. **`load_xoftr` 签名不匹配：确认**。
   - `load_model` 无条件调用 `load_{method}(args, test_orginal_megadepth=...)`（`load_model.py:160,163`），但 `def load_xoftr(args)` 没有这个参数（`:143`）。
   - 所以 `load_model('xoftr', ...)` 必然抛 `TypeError: unexpected keyword argument`。所有测试脚本和 demo 都经过 `load_model`（`demo.py:133`、`test_relative_homo_mmim.py:555` 等），**MINIMA 仓库里的 XoFTR 路径跑不通**。
   - 绕过办法：直接调用 `load_xoftr(args)`。
2. **`np.float`**：
   - 出现在 `data_io_loftr.py:65`（LoFTR）和 `data_io_roma.py:77`（RoMa）。numpy ≥ 1.24 会报 AttributeError（仓库锁定的是 `numpy==1.23.1`，`MN/requirements.txt:1`）。
   - `data_io.py:72`（XoFTR）和 `data_io_sp_lg.py` 已改成 `float`。
   - 子模块 `third_party/XoFTR/src/utils/data_io.py:67` 也有 `np.float`，但 MINIMA 用的是自己的 `MN/src/utils/data_io.py`，不经过它。
3. **RoMa_minima 在 py3.8 上的类型注解**：`tuple[int,int]`（`RoMa_minima/romatch/models/model_zoo/__init__.py:30,54`）在 python 3.8 导入时就会报错，需要 README 那条 `sed`（`README.md:253`）。python ≥ 3.9 不需要。
4. **依赖当前工作目录**：
   - `sys.path.append("./third_party/RoMa_minima/")`（`load_model.py:9`）。
   - `from src.config...` 和 `from third_party...` 这类包路径都依赖 cwd = MINIMA 根目录。
5. **只能在 CUDA 上跑**：`match_images` / `from_cv_imgs` 里写死了 `torch.cuda.synchronize()`（`data_io_loftr.py:121,125`；`data_io_roma.py:109,113`；`data_io.py:127,130`）。
6. **全局副作用**：`DataIOWrapper.__init__` 调用 `torch.set_grad_enabled(False)`（`data_io_loftr.py:31` 等），会作用于整个进程。
7. **`weights/download.sh` 只下 3 个文件**（lightglue、loftr、roma）。`minima_eloftr` 和 `minima_xoftr` 需要按 README 的链接手动下载（`README.md:126-130`）。

### 4.3 共同的预处理（LoFTR、XoFTR、SP+LG 相同；RoMa 见第 7 节）
- 配置（`MN/src/config/default.py:188-193`）：`TEST.IMG0_RESIZE = IMG1_RESIZE = 640`（按长边），`DF=8`，`PADDING=False`。
  - `default_for_megadepth_dense/_sparse.py` 只把长边改成 1200 / 1600，仅在 `test_relative_pose_mega_1500.py` 里使用。
- `preprocess_image`（`data_io_loftr.py:42-77`）：
  1. 若是 3 维，做 `cv2.cvtColor(BGR2GRAY)`（`:44-45`）。
  2. 按长边缩放到 640，向下取整到 8 的倍数（`:55-62`）。
  3. `cv2.resize`，默认 INTER_LINEAR（`:64`）。**512² 会被放大到 640²**（保持宽高比，两图各自独立缩放）。
  4. `scale = [w/w_new, h/h_new]`（`:65`）。
  5. 灰度：`torch.from_numpy(img)[None][None].float() / 255.0`（`:74`）。
- **对 float 输入（Q3）**：
  - `/255.0` 是无条件的（`:74,76`）。如果传入已经在 [0,1] 的 float，会再被除一次，变成 ≈0，**静默失败**。
  - `cv2.cvtColor` 不支持 float64（只支持 8U/16U/32F）。
  - 结论：`from_cv_imgs` 应当喂 **uint8（或 0–255 的 float32）**。
- MINIMA 官方的遥感跨模态评测（`test_relative_homo_mmim.py --choose_model 1`）：
  - 子集列表 `test_list_2.txt` 包括 `RemoteSensing/SAR_Optical`（6 对，`SO{k}a/b.png`）以及 CrossSeason、DayNight、DepthOptical、Infrared_Optical、Map_Optical、Optical_Optical（读自 `MN/data/Multi-modality-...zip`）。
  - 配对方式：**im0 = `files[2]`（b 图），im1 = `files[1]`（a 图）**（`test_relative_homo_mmim.py:144-156`）。
  - 调用：`from_paths(im0, im1)`（`:356`，`use_path=True`）。
  - 估计：`cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)`，即**默认 3px 阈值**。`--ransac_thres 1.5` 只被解析和记录，**没有实际使用**（`:382,498,604`）。
  - GT 的 `.mat` 做了 1-based 到 0-based 的平移换算（`:151-154`），说明 MINIMA 把输出视为 0-based。

### 4.4 复用建议
- **可以复用** `load_{loftr,sp_lg,roma}(args)` 和 `load_xoftr(args)`（直接调，不经过 `load_model`），再用它们的 `.from_cv_imgs(img0, img1)`。
- 需要提供一个 `SimpleNamespace` 形式的 args（字段见 4.1），并把 cwd 设为 MINIMA 根目录。
- numpy ≥ 1.24 时，可以在导入前 `np.float = float` 临时打补丁（推断：无副作用）。
- RoMa 另需两件事：输入 **3 通道 BGR**，输出坐标 **−0.5**（见第 7 节）。
- ELoFTR 不在这个外壳里（见第 9 节）。

---

## 5. MINIMA-LoFTR

### 5.1 推理路径 / 权重 / 配置
- `load_loftr(args)`（`MN/load_model.py:37-61`）：
  - 用 `LoFTR_minima` 子模块的 `default_cfg` 做 deepcopy。子模块 HEAD `9ba0871 "Update minima code"` 只改了训练和数据集文件，推理核心与上游 LoFTR 相同（`git show --stat`：`megadepth.py`、`lightning_*`、`train.py`、configs）。
  - 若 ckpt 文件名不是 `outdoor_ds.ckpt`，设 `coarse.temp_bug_fix=True`（`:48-51`）。
  - `match_coarse.thr = args.thr`，默认 0.2（`:53`；`:182`）。
  - `load_state_dict(torch.load(ckpt)['state_dict'], strict=True)`（`:56`）；`matcher.` 前缀会被剥掉（`LoFTR_minima/src/loftr/loftr.py:77-81`）。
- 结构（`LoFTR_minima/src/loftr/utils/cvpr_ds_config.py:11-48`）：ResNetFPN (8,2)、线性注意力、`dual_softmax`、`BORDER_RM=2`。

### 5.2 输入
- 最窄边界是 `LoFTR.forward(batch)`：`image0/1` 为 (1,1,H,W) float [0,1]，H、W 须为 8 的倍数。
- 经 wrapper 调用时，输入 2D uint8 灰度或 3 通道 BGR uint8 均可，然后按 4.3 处理（640 长边、÷255、无 mean/std）。
- 两图不要求同尺寸。

### 5.3 输出
- `batch['mkpts0_f','mkpts1_f','mconf']` 在**缩放后网格**里（wrapper 不传 `scale0`，所以 `coarse_matching.py:243-250` 里 scale 只是 `hw_i/hw_c`），随后由 wrapper 乘 `scale` 回到原图（`data_io_loftr.py:86-87`）。
- 像素约定：`i*8 → *scale`，没有 ±0.5 修正。对 512→640 的情况，偏差 0.5·(0.8−1) = −0.1 px，两侧相同，可忽略。
- 半稠密，`mconf` 是粗匹配置信度。

### 5.4 适配草图
```python
import os, sys, types, numpy as np; os.chdir(MINIMA); sys.path.insert(0, MINIMA)
np.float = float                                  # 仅当 numpy>=1.24
from load_model import load_loftr
m = load_loftr(types.SimpleNamespace(ckpt="weights/minima_loftr.ckpt", thr=0.2))
r = m.from_cv_imgs(opt_u8, sar_u8)                # 2D uint8 -> 内部 640 长边 / 255
k0, k1, conf = r["mkpts0"], r["mkpts1"], r["mconf"]   # 已在 512 原网格
```

---

## 6. MINIMA-XoFTR

### 6.1 推理路径 / 权重 / 配置
- `load_xoftr(args)`（`load_model.py:143-155`）：
  - `get_cfg_defaults(inference=True)` 把三处 `INFERENCE` 设为 True（`MN/src/config/default.py:195-203`）。注意它**修改的是全局 `_CN`**，再 clone。
  - `match_coarse.thr = args.match_threshold`（默认 0.3），`fine.thr = args.fine_threshold`（默认 0.1）（`:149-150`；`:175-176`；与 `default.py:31,43` 一致）。
  - 模型：`XoFTR(config["xoftr"])`。权重由 `DataIOWrapper(ckpt=...)` 执行 `load_state_dict(ckpt['state_dict'])`（strict）（`MN/src/utils/data_io.py:44-47`）。
- 模型来自子模块 `third_party/XoFTR@e9635d8`；`xoftr.py` 全部用相对导入（`:4-6`），不会和 MINIMA 的 `src` 包冲突。
- **入口是坏的**（见 4.2 第 1 条）：需要绕过 `load_model`，直接调用 `load_xoftr`。
- 权重：
  - 默认 ckpt 路径是原版 XoFTR 的 `weights_xoftr_640.ckpt`（`load_model.py:177`）。
  - MINIMA 版是 `minima_xoftr.ckpt`，只能从 GitHub release 手动下载（`README.md:130`）。仓库里没有 XoFTR 的训练脚本（`train_orders/` 只有 lightglue、loftr、roma）。

### 6.2 输入
- 与 4.3 相同：灰度，长边 640，df=8，÷255。
- 灰度张量在 `preprocess_image` 里就被 `.to(self.device)`；彩色分支不会（`data_io.py:80-83`）。

### 6.3 输出
- `mkpts0_f`、`mkpts1_f`，置信度用 **`mconf_f`**（细匹配置信度，`data_io.py:134-136`），而 LoFTR 用 `mconf`。
- 坐标系：模型里 `scale = hw_i/hw_c`（`XoFTR/src/xoftr/xoftr_module/coarse_matching.py:288-296`）；wrapper 再乘 `scale` 回到原图（`data_io.py:92-93`）。
- 半稠密。

### 6.4 适配草图
```python
from load_model import load_xoftr                 # 不要走 load_model('xoftr')——TypeError
m = load_xoftr(types.SimpleNamespace(ckpt="weights/minima_xoftr.ckpt",
                                     match_threshold=0.3, fine_threshold=0.1))
r = m.from_cv_imgs(opt_u8, sar_u8); k0, k1, conf = r["mkpts0"], r["mkpts1"], r["mconf"]
```


---

## 7. MINIMA-RoMa

### 7.1 推理路径 / 权重 / 配置
- `load_roma(args)`（`MN/load_model.py:7-34`）：
  - `sys.path.append("./third_party/RoMa_minima/")`，再 `from third_party.RoMa_minima.romatch import roma_outdoor`。
  - `ckpt2=='large'` 且给了 ckpt 时：`state_dict = torch.load(ckpt)`（整个文件就是 state_dict，没有 `['state_dict']` 这一层），然后 `roma_outdoor(device, weights=state_dict)`（`:20-26`）。
  - 最后 `DataIOWrapper(matcher, config["test"])`（`:32`）。
- `roma_outdoor` 的默认参数（`RoMa_minima/romatch/models/model_zoo/__init__.py:30-52`）：
  - `coarse_res=560`，`upsample_res=864`，`amp_dtype=fp16`（CPU 上改为 fp32，`:36-37`）。
  - DINOv2 权重在线下载（`:45-47`）。**离线环境要预先放好缓存**。
- `roma_model`（`romatch/models/model_zoo/roma_models.py:154-170`）：
  - VGG19 分支 `pretrained=False`（`:156`），不需要下载 torchvision 权重。这一点和 MA 不同。
  - `symmetric=True`、`attenuate_cert=True`、`sample_mode="threshold_balanced"`（`:164-168`）。
  - `load_state_dict(weights)` 是 strict 模式（`:169`）。
- 采样：`RegressionMatcher.sample(num=10000)`（默认值，`romatch/models/matcher.py:468-495`），`sample_thresh=0.05`（`:450`）。
  - wrapper 调用 `self.model.sample(warp, certainty)` 时没有传 num（`MN/src/utils/data_io_roma.py:117`），所以**一次采 10000 对**；MA 是 5000。
  - 与 MA 版不同：这里没有 `certainty.sum()==0` 的保护（对照 `MAR/models/matcher.py:487-494`）。

### 7.2 输入（重点：两次缩放 + 一次 uint8 重量化）
1. `from_paths` 默认用 `read_color=True`，即 `cv2.imread(IMREAD_COLOR)`，得到 BGR uint8（`data_io_roma.py:144-149`）。
2. `preprocess_image(gray_scale=False)` **无条件**执行 `cv2.cvtColor(img, BGR2RGB)`（`:56-57`）。所以 `from_cv_imgs` **必须传 3 通道**，传 2D 灰度会让 cv2 报错。
3. 按长边缩放到 640，df=8，INTER_LINEAR（`:67-76`），得到 512→640。
4. `/255.0`（`:88`）。
5. `ToPILImage()`（`:105-107`）：torchvision 对 float 张量执行 `mul(255).byte()`（推断，依 torchvision 实现），即**截断回 uint8 的 PIL RGB 图**。
6. `model.match(pilA, pilB, batched=False)`（`:112`）：
   - `check_rgb` 要求 PIL 的 mode 为 `"RGB"`（`romatch/models/matcher.py:629,637`；`utils/utils.py:660-662`）。**不接受 numpy 或张量**。
   - `TupleResize((560,560), BICUBIC)`（拉伸），然后 `/255`，然后 **ImageNet mean/std**（`matcher.py:651-654`；`utils/utils.py:165-174,234-239`）。
   - 上采样阶段：从同一个 PIL 图重新缩放到 864×864，再次归一化（`matcher.py:682-700`）。
- 最终效果：512 →（cv2 线性）640 →（量化）→（PIL bicubic 拉伸）560 / 864。**缩放了两次**，而且第一次放大后的结果会被 RoMa 缩小回去。

### 7.3 非自然分布
- `ToPILImage` 的 `.byte()` 对超出 [0,1] 的值会**回绕**（推断，依 torchvision 实现）。
- 输入必须是 uint8 BGR 3 通道。SAR 需先转 uint8，再复制成 3 通道。

### 7.4 输出
- `to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)`，其中 H、W 取的是 **640 网格**的尺寸（`data_io_roma.py:102-103,119`）；公式 `W/2*(x+1)`（`matcher.py:563-565`）。随后乘 `scale0/1`（512/640），得到原图坐标（`data_io_roma.py:124-126`）。
- 像素约定：**+0.5**（网格是 `linspace(-1+1/hs, 1-1/hs)`，`matcher.py:715-721`），经线性缩放后仍然是 +0.5，所以**需要 −0.5**。
- `mconf` 是截断后的 certainty（超过 0.05 的为 1）。
- 没有任何置信度过滤（wrapper 不过滤）。
- 采样是随机的，需要 `torch.manual_seed`。

### 7.5 适配草图
```python
from load_model import load_roma
torch.manual_seed(0)
m = load_roma(types.SimpleNamespace(ckpt="weights/minima_roma.pth", ckpt2="large"))
bgr = lambda u8: np.repeat(u8[..., None], 3, -1)          # 灰度 -> 3 通道（BGR 顺序无所谓）
r = m.from_cv_imgs(bgr(opt_u8), bgr(sar_u8))
k0, k1, conf = r["mkpts0"] - 0.5, r["mkpts1"] - 0.5, r["mconf"]   # RoMa +0.5 -> 像素中心在整数
```

---

## 8. MINIMA-SuperPoint+LightGlue

### 8.1 推理路径 / 权重 / 配置
- `load_sp_lg(args)`（`MN/load_model.py:64-140`）内部定义了一个 `Matching` 模块。
- SuperPoint：`SuperPoint(**sp_conf)`，参数 `max_num_keypoints=2048`、`detection_threshold=0.0005`、`nms_radius=4`、`remove_borders=4`（`:114-120`）。
  - 权重是**官方 SuperPoint v1**，构建时从 GitHub release 下载（`LightGlue/lightglue/superpoint.py:144-145`）。
  - **MINIMA 没有微调 SuperPoint**，只替换了 LightGlue 的权重。
- LightGlue：`LightGlue(features='superpoint', **lg_conf)`。
  - 构造时**先下载官方 `superpoint_lightglue` 权重**（`lightglue.py:408-414`，版本 `v0.1_arxiv`，`:341`）。
  - 然后由 MINIMA 重命名 `self_attn.{i}` / `cross_attn.{i}` 键，再 `load_state_dict(minima_lightglue.pth, strict=False)` 覆盖（`load_model.py:81-89`）。
  - **`strict=False` 是风险点**：键名改错时会静默沿用官方 LG 权重。适配层必须核对 missing / unexpected keys。
- LightGlue 配置（`:121-134`）：`filter_threshold=0.1`，`depth_confidence=0.95`，`width_confidence=0.99`（提前终止 / 剪枝，结果会有轻微非确定性），`flash=True`，`mp=False`。

### 8.2 输入（同样两次缩放）
1. DataIOWrapper（`data_io_sp_lg.py`，与 `data_io_loftr.py` 基本相同）：灰度，长边 640，df=8，÷255，得到 (1,1,640,640) 的 CUDA 张量。
2. `extractor.extract(image)` 使用 `SuperPoint.preprocess_conf = {"resize": 1024}`（`superpoint.py:115-117`）：
   - `kornia.geometry.transform.resize(side='long', antialias=True)`（`LightGlue/lightglue/utils.py:12-38,137-147`），**512 → 640 → 1024**。
   - 只有 `test_orginal_megadepth=True` 时才传 `resize=None`（`load_model.py:96-101`）。
3. SuperPoint 接受 1 或 3 通道（3 通道时 `rgb_to_grayscale`，`superpoint.py:155-156`），**不做 mean/std**，期望 [0,1]。

### 8.3 输出
- `extract` 把关键点换回 640 网格：`(kp+0.5)/scale-0.5`（`utils.py:146`），像素中心在整数。
- `rbd` 去掉 batch 维，按 `matches` 索引取点（`load_model.py:105-110`）。
- wrapper 再乘 `scale`（512/640）（`data_io_sp_lg.py`，与 `data_io_loftr.py:86-87` 相同）。这一步没有 ±0.5 修正，偏差 −0.1 px，两侧相同。
- `mconf` 为 `matching_scores0[matches[:,0]]`。
- 稀疏：每图最多 2048 个关键点。

### 8.4 环境注意
- `LightGlue/lightglue/utils.py:45,47` 用了 `collections.Mapping` / `collections.Sequence`，python ≥ 3.10 下已移除。只有 `map_tensor` / `batch_to_device` 会触发，MINIMA 的推理路径不走它们（`rbd` 不调用 `map_tensor`，`:64-67`）。
- 需要联网或缓存才能拿到 `superpoint_v1.pth` 和 `superpoint_lightglue_v0-1_arxiv.pth`。

### 8.5 适配草图
```python
from load_model import load_sp_lg
m = load_sp_lg(types.SimpleNamespace(ckpt="weights/minima_lightglue.pth"))
r = m.from_cv_imgs(opt_u8, sar_u8); k0, k1, conf = r["mkpts0"], r["mkpts1"], r["mconf"]
```

---

## 9. MINIMA-ELoFTR（`minima_eloftr.ckpt`）

- **仓库里没有推理代码**：README 发布了权重（`README.md:123,129`），但 `load_model.py` 没有 `load_eloftr`，CLI 也不接受 eloftr（`:167-170`）；没有 ELoFTR 子模块；`train_orders/` 里也没有 ELoFTR 的训练配置。
- **推断**：
  - 需要用上游 `zju3dv/EfficientLoFTR` 的代码来加载（该仓库**不在本次克隆范围内**，未核实）。结构和 cfg（`full_default_cfg` / `opt_default_cfg`，是否需要 `reparameter`）要在拿到权重后对照 state_dict 键名确认。
  - 按 MINIMA 的惯例，推理预处理应与其他 LoFTR 系一致：灰度，长边 640，÷255。ELoFTR 要求尺寸是 32 的倍数，640 满足。
  - 不能用 MA 的 LoFTR 代码加载它：MA-ELoFTR 是 RepVGG + RoPE + PAN 的定制配置（`MA/configs/models/eloftr_model.py:33-37,62,112`），与原版 ELoFTR 的 cfg 不同（推断，需要按键名核对）。
- **结论**：在 group-b 的范围内，这是唯一「只有权重、没有官方调用路径」的匹配器。如果要跑，需要另外克隆 EfficientLoFTR，并自己拼一个 DataIOWrapper 风格的调用（可以直接复用 `MN/src/utils/data_io_loftr.py` 的 `DataIOWrapper`：它只要求 `model(batch)` 写出 `mkpts0_f`、`mkpts1_f`、`mconf`，而 ELoFTR 正好也输出这三个键，推断）。

---

## 10. 对比总表

| 匹配器 | 官方入口（最窄边界） | 输入 dtype / 通道 | 归一化 | 内部缩放（512² 时） | 宽高比 | 阈值 / 采样（官方评测） | 输出坐标 | 像素中心修正 | 已知坏点 |
|---|---|---|---|---|---|---|---|---|---|
| MA-ELoFTR | `PL_LoFTR(cfg).matcher(batch)`，键 `image0/1`、`scale0/1` | (1,1,H,W) float，灰度 | ÷255 | cv2 线性，**拉伸到 832²**（SAR 集强制） | 拉伸 | coarse thr **0.05**（脚本）/ 0.1（配置）；NPE [832,832,832,832] | 模型内乘 scale，**已是原图坐标** | 无（两侧相同，可忽略） | `strict=False`；配置文件修改全局 `_CN` |
| MA-RoMa | 同上，键 `image0/1_rgb_origin` | (1,3,H,W) float [0,1] 原尺寸，RGB | ÷255，**不做 ImageNet 归一化** | torchvision bicubic 拉伸到 560 → 864 | 拉伸 | `threshold_balanced`，**5000**，thr 0.05，fp16 | 原图坐标（H、W 取原尺寸） | **−0.5** | `strict=False`；DINOv2 和 VGG19 需在线下载 |
| MINIMA-LoFTR | `load_loftr(args).from_cv_imgs(u8, u8)` | uint8 2D 或 BGR | ÷255 | cv2 线性，长边 640，df 8 | 保持 | coarse thr 0.2 | wrapper 乘 scale，原图坐标 | 无（−0.1px，可忽略） | `np.float`（`data_io_loftr.py:65`） |
| MINIMA-XoFTR | `load_xoftr(args).from_cv_imgs` | 同上 | ÷255 | 同上 | 保持 | coarse 0.3 / fine 0.1；置信度用 `mconf_f` | 原图坐标 | 无 | **`load_model('xoftr')` 会 TypeError**；默认 ckpt 是原版 XoFTR |
| MINIMA-RoMa | `load_roma(args).from_cv_imgs(bgr, bgr)` | **必须 BGR 3 通道 uint8** | ÷255，经 PIL 往返，**ImageNet mean/std** | 640（cv2）→ 560 / 864（PIL bicubic 拉伸） | 先保持，后拉伸 | `threshold_balanced`，**10000**，thr 0.05 | 原图坐标 | **−0.5** | `np.float`（`data_io_roma.py:77`）；py3.8 需 `tuple` 补丁；DINOv2 在线下载 |
| MINIMA-SP+LG | `load_sp_lg(args).from_cv_imgs` | uint8 2D 或 BGR | ÷255 | 640（cv2）→ **1024**（kornia antialias） | 保持 | SP 2048 点，thr 0.0005；LG filter 0.1 | 原图坐标 | 已在 1024→640 做过；640→512 未做（−0.1px） | LG 覆盖加载 `strict=False`；SP 是官方原版权重 |
| MINIMA-ELoFTR | **没有** | — | — | — | — | — | — | — | 需要上游 EfficientLoFTR 代码（未核实） |

**所有 MINIMA wrapper 共有**：写死 `torch.cuda.synchronize()`；依赖 cwd；`set_grad_enabled(False)` 是全局的；环境为 py3.8、torch 2.0.1、cu11.7、numpy 1.23.1、opencv 4.5.1.48、kornia 0.6.11（`MN/environment.yaml:9-13`，`MN/requirements.txt:1-6`）。
**MA 的环境**：自带 yaml 为 py3.8、torch 1.12.1、cu11.7、kornia 0.4.1、pytorch-lightning 1.3.5；Space 实际跑在 torch 2.8、PL 1.4.9、numpy~1.24 上；没有编译算子，xformers 可选。
