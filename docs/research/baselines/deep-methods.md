# 深度方法明细

> 行号都对应 [sources.md](sources.md) 里记下的 commit。「读到」= 直接读了代码或 README；「推断」= 我的判断。
> 服务器已有克隆的方法，如果服务器上的 commit 和我读的不同，会在对应条目里单独说明。

---

## 1. MatchAnything（ELoFTR 版 / RoMa 版）— 跨模态，推荐

- **出处**：He et al., *MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training*，TPAMI 2026（GitHub README 第 12 行）；arXiv 2501.07556。
- **代码**：GitHub `zju3dv/MatchAnything@8cd8c11` 只有 README，说明模型和代码都在 HF Space 上（README 第 22 行）。实际推理和评测代码在 HF Space `LittleFrog/MatchAnything@6a7bcb5` 的 `imcui/third_party/MatchAnything/` 下。许可证 Apache-2.0（HF 模型卡 tags）。
- **权重**：
  - 官方 zip 放在 Google Drive（Space 内 README 第 37 行）。
  - HF `zju-community/matchanything_eloftr@7bd52a4`：官方 transformers 版，只有 ELoFTR，不是 gated。
  - HF `vismatch/matchanything-eloftr`、`vismatch/matchanything-roma@5f0b730`：第三方镜像。
  - 服务器需要走 hf-mirror。
- **光学–SAR**：有官方评测脚本 `scripts/evaluate/eval_visible_sar.sh`。
  - 第 14 行（ELoFTR）：`--imgresize 832 --thr 0.05 --npe`，方法名为 `matchanything_eloftr@-@ransac_affine`。
  - 第 17 行（RoMa）：`--imgresize 832`，`ransac_affine`。
  - 论文数值（arXiv HTML，读到）：visible–SAR 1209 对，指标为「仿射变换估计后，landmark 平均 warp 误差的成功率」。MA-RoMa SR@10px = 93.3%，MA-ELoFTR = 72.5%，分别比各自原版相对提升 78.5% 和 207.5%。据此反推原版 RoMa ≈ 52%，ELoFTR ≈ 24%（推断，由相对提升换算）。
  - 论文正文写评测分辨率为「长边 840」，与脚本的 832 不一致。
- **输入约定（读到）**：
  - 数据集 loader `read_megadepth_gray`：`cv2.IMREAD_GRAYSCALE` → 按长边 resize → `/255`（`src/utils/dataset.py:117-129,207-257`）。评测时 `read_gray=True, normalize_img=False, df=None, img_padding=False, load_origin_rgb=True`（`tools/evaluate_datasets.py:137`）。
  - **ELoFTR 版**：直接用上面这个灰度 [0,1] 张量；`NPE = [832,832,imgresize,imgresize]`（`evaluate_datasets.py:115`）。coarse 阈值由脚本传入 0.05（config 默认 0.1，`configs/models/eloftr_model.py:96`）。
  - **RoMa 版**：用原图重新读入，`Image.open(...).convert("RGB")` → `/255.`，`RESIZE_BY_STRETCH=True`（拉伸成方形）；`norm_img=False`，**不做 ImageNet 归一化**（`third_party/ROMA/roma/models/matcher.py:649-677`；`configs/models/roma_model.py:2-3`；`src/config/default.py:5-27`）。coarse 560² / upsample 864²、symmetric、`threshold_balanced` 采样 5000 对，采样阈值 0.05（`default.py:17-27`）。
- **输出**：半稠密匹配（ELoFTR 版）或稠密 warp 采样得到的匹配（RoMa 版），坐标映射回原图。
- **环境**：Space 内 README 写「在 CUDA 11.7 上测试」；environment.yaml 未展开核对。
- **风险**：transformers 的 `AutoImageProcessor` 默认会 resize 到 480×640（ELoFTR README 第 170 行）；vismatch 包装的 MA-RoMa 是否复现了 `norm_img=False` **未核实**。

## 2. MINIMA — 跨模态，推荐 RoMa 版

- **出处**：Ren et al., CVPR 2025（README 第 40 行）；arXiv 2412.19412。代码 `LSXI7/MINIMA@796e772`（与服务器一致），Apache-2.0。
- **权重**：GitHub release `LSXI7/storage/releases/download/MINIMA/{minima_lightglue.pth, minima_loftr.ckpt, minima_roma.pth, minima_eloftr.ckpt, minima_xoftr.ckpt}`（README 第 125-130 行），也可以用 `weights/download.sh`。GitHub release 服务器应该能访问。
- **光学–SAR**：MMIM 遥感子集混合了 7 类跨模态（光学–SAR、光学–地图、光学–深度等），用单应性角点误差 AUC@3/5/10 评测。MINIMA-RoMa 32.55 / 44.68 / 64.38，原版 RoMa 29.24 / 40.50 / 57.84，XoFTR 27.35 / 39.58 / 56.63，测试时长边统一为 640（arXiv HTML 表 5，读到）。第三方 rsim：在 SRIF 光学–SAR 子集上是唯一 0% 失败的方法。
- **输入约定（读到）**：
  - 统一经过 `DataIOWrapper`：`cv2.imread`（RoMa 用 `IMREAD_COLOR`，其余用 `IMREAD_GRAYSCALE`，`data_io_roma.py:144-149`，`data_io_loftr.py:103-108`）→ 长边 resize 到 640、df=8（`src/config/default.py:189-193`）→ `/255.`。
  - RoMa 版先 `cv2.cvtColor(BGR2RGB)`，再 `ToPILImage` 后交给 RoMa 的 `match(..., batched=False)`（`data_io_roma.py:56-57,105-112`）。RoMa 内部再 resize 到 560/864，并做 ImageNet 归一化。**等于缩放了两次**：512 → 640 → 560/864。
  - 输出乘 `scale = w/w_new` 回到原图（`data_io_roma.py:124-126`）。
  - SP+LG：`SuperPoint(max_num_keypoints=2048, detection_threshold=0.0005)`，LightGlue `filter_threshold=0.1`（`load_model.py:114-134`），`extract()` 默认按长边 1024 重新缩放（LightGlue 行为）。
  - LoFTR：`thr` 默认 0.2（`load_model.py:182`）；非官方 ckpt 需要 `temp_bug_fix=True`（`load_model.py:49-51`）。
  - XoFTR：coarse 0.3 / fine 0.1（`load_model.py:175-176`）。
- **环境**：python 3.8、pytorch 2.0.1、pytorch-cuda 11.7、numpy 1.23.1、kornia 0.6.11（`environment.yaml:9-13`，`requirements.txt:1-6`）。
- **坑（读到 + 推断）**：
  - `np.float` 出现在 `data_io_roma.py:77`、`data_io_loftr.py:65`，numpy ≥ 1.24 会报错。
  - `load_model()` 把 `test_orginal_megadepth` 传给 `load_xoftr(args)`，但后者签名里没有这个参数（`load_model.py:143,160`），推断会 TypeError。
  - 虽然发布了 `minima_eloftr` 权重，但 `load_model.py` 里没有对应的 eloftr 加载器（`choices` 只有 xoftr / sp_lg / loftr / roma，第 169 行）。
  - RoMa_minima 需要 README 里那条 `sed` 补丁（第 253 行）。

## 3. XoFTR — 跨模态（可见–热红外），推荐

- **出处**：Tuzcuoğlu et al., CVPR 2024 Image Matching Workshop（README 第 5 行）。代码 `OnderT/XoFTR@e0fbea4`，Apache-2.0（LICENSE 读到）。
- **服务器版本**：`e9635d8`（2024-09-08）。上游 `e0fbea4`（2025-08-20）的 commit message 是「Update data_io.py np.float -> np.float32」，所以服务器版在新版 numpy 下大概率报错（推断）。
- **权重**：Google Drive（README「Pretrained models weights」，640 和 840 两个版本）。镜像 HF `vismatch/xoftr@d8ee7d8`，包含 `xoftr_640.safetensors`、`xoftr_840.safetensors`（第三方）。
- **光学–SAR**：官方没有评测。第三方 rsim 在 SpaceNet9 上 3.0 px（并列第一），约 0.4 s/对；最优归一化是 percentile。
- **输入约定（读到）**：
  - `DataIOWrapper.preprocess_image`：BGR→GRAY（第 46-47 行）→ 按长边 resize 到 640（`default.py:189-190`）并向下取整到 8 的倍数（第 57-66 行）→ `/255.0`（第 76 行）→ 输出乘 `scale` 回到原图（第 87-88 行）。
  - `PADDING=False`（`default.py:192`）。阈值 coarse 0.3、fine 0.1（`default.py:31,43`）；`get_cfg_defaults(inference=True)` 打开推理模式（第 199-202 行）。
  - 灰度分支里写死了 `.cuda()`（第 76 行）。
  - demo 里用的是 `USAC_MAGSAC`（notebook）。
- **环境**：python 3.8、pytorch 2.0.1、CUDA 11.8（`environment.yaml:9-11`）。

## 4. ReDFeat — 光学–SAR 专门训练，推荐作为 SAR 监督参照

- **出处**：Deng & Ma, *ReDFeat: Recoupling Detection and Description for Multimodal Feature Learning*，IEEE TIP 2023（arXiv 2205.07439；`Multimodal_Feature_Evaluation` README 链接 IEEE 9999700）。代码 `ACuOoOoO/ReDFeat@66df800`，**仓库里没有 LICENSE 文件**。
- **权重**：就在仓库里：`Pretrained/VIS_SAR.pth`、`VIS_IR.pth`、`VIS_NIR.pth`（文件树读到）。VIS_SAR 的训练数据是 OSdataset（Xiang et al. 2020，GF-3，1 m，512×512，训练/测试 2011/424 对，两侧都是 1 通道），见 `Multimodal_Feature_Evaluation` README 表格。
- **光学–SAR 结果**：MIFNet 论文的 OSdataset 400 对表中，ReDFeat SRR 41.4%、Herr 3.3、MS 9.1%，是表内最好（arXiv 2501.11299 HTML，读到）。
- **输入约定（读到，`match.py`）**：
  - `Image.open().convert('RGB')` → `TF.to_tensor`（[0,1]）→ **逐通道 z-score**（第 138-141 行、第 158-161 行）。
  - 可见光走 `net.forward1`，SAR 走 `net.forward2`（第 75-78 行），**不对称**。
  - 多尺度金字塔：`scale_f = 2^0.25`，`min_size = 256`，`max_size = 1000`（第 118-121 行）。
  - NMS 阈值：repeatability 0.4、reliability 0.5；边界 5 px。取 top 4096 个点（第 114、123-125 行）。
  - 匹配：BF + ratio 0.9（第 180-186 行）。
  - `extract_multiscale` 里用了全局变量 `args.border`（第 81 行），把它当库函数调用时需要改。
- **环境**：README 写 PyTorch 1.10、Kornia 0.6.2，「更新的版本也应兼容」。

## 5. FHReg（GUSO）— 光学–SAR 专门训练，条件推荐

- **出处**：Yan et al., ISPRS JPRS 235 (2026) 190-210（README bibtex）。代码 `vision-heng/GUSO@f3e5420`，MIT 许可。
- **权重**：只在 Google Drive（README「Weight Download」）。**没有镜像**。
- **光学–SAR**：在 GUSO 上训练（0.16–0.98 m 超高分辨率），声称能 zero-shot 迁移到 OSdataset 和 MSAW（README）。
- **输入约定（读到）**：
  - batch 的键名是 `image_opt` / `image_sar`（`demo.py:63`），两张图拼成一个 batch 共用 backbone（`wavelet_backbone.py:292-297`）。
  - stem 是 `Conv2d(3, 96, 4, stride 4)` + LayerNorm2d，**要求 3 通道**（`wavelet_backbone.py:247-250`）。
  - `demo.py` 用 rasterio 读成 float32，**不做 ÷255**（第 92-106、199-203 行）；而 `zero_shot_msaw_os.py` 用 `TF.to_tensor(uint8)`，得到的是 [0,1]（第 116-117 行）。两条官方路径不一致，建议按 zero-shot 脚本走（推断）。
  - coarse 阈值 0.1（`demo.py:31`）；`MGDPT_IMG_RESIZE = 512`（`cvpr_ds_config.py:74`）。
  - 输出先经 FSC（affine，3 px）过滤，再做 KDE 采样（`demo.py:209-214,239-243`）。为了统一评测，我们应该取 FSC 之前的原始匹配 `mkpts0_f / mkpts1_f`（推断）。

## 6. MIFNet — 跨模态（单模态训练），备选

- **出处**：Liu et al., IEEE TIP 34:3593-3608, 2025（README bibtex）。代码 `lyp-deeplearning/MIFNet@dd3bde3`（与服务器一致），MIT 许可。
- **权重**：Dropbox（README 第 36 行）；另外需要从 HF 下载 `stabilityai/stable-diffusion-2-1`（第 39-44 行）。XFeat 权重在仓库里（`checkpoints/xfeat.pt`）。
- **光学–SAR**：OSdataset 400 对（GF-3，512²）：XFeat 18.4% → XFeat+MIFNet 36.1%，SuperPoint+MIFNet 39.7%，ReDFeat 41.4%（论文，读到）。
- **输入约定（读到）**：
  - XFeat 分支：`cv2.imread(IMREAD_COLOR)`（BGR）→ PIL → `Resize((768,768))`（**拉伸**）→ `ToTensor` → `×255`（`xfeat_engine.py:295-307`）→ 模型内部取通道均值 + InstanceNorm（`xfeat_engine.py:173-174`）。
  - top 1024 个点，detection_threshold 0.01（`configs/xfeat.yaml`）。
  - SD（DIFT）分支：`PIL.convert('RGB')`，`img_size = 768`，`t = 0`，`up_ft_index = 1`，`ensemble = 8`（`test_xfeat_mifnet.py:73-84`；yaml）。
  - opt-sar 模式：`class_threshold = 0.2`（第 129 行）。
- **坑（读到）**：
  - 第 106 行 `if args.mode == "opt-nir" or "opt-sar":` 恒为真，所以总是加载 remote 权重。对我们来说这正好是需要的那个。
  - 输出关键点在 **768 网格**里，没有映射回原图（第 92-94 行）。
- **环境**：torch 2.0.1、diffusers 0.15.0、xformers 0.0.20、transformers 4.29.2、python 3.10（`requirements.txt`，README）。

## 7. VMGGA — 光学–SAR 有专用权重，备选（被百度网盘卡住）

- **出处**：Tang, Han, Peng, Chen, Ye, ISPRS JPRS 2026（doi 10.1016/j.isprsjprs.2026.05.005，README 第 9 行）。代码 `yeyuanxin110/VMGGA@5df54e8`，Apache-2.0。
- **权重**：`vmgga_optical_sar.pth` **只在百度网盘**，Google Drive 标注为 TBA（README 第 73-78 行）。README 说 checkpoint 自包含，不需要另外下载 DINOv3（第 70-71 行）。
- **输入约定（读到）**：
  - `cv.imread(IMREAD_COLOR)`（`src/utils/image.py:14`）→ BGR2GRAY → 右下补边到 16 的倍数 → `/255.0` → `{"image0","image1"}`（`demo_vmgga.py:318-328`）。
  - 光学–SAR 配置：`match_threshold = 2e-5`，`ransac_threshold = 2.0`（`configs/demo/optical_sar.yaml`）。
  - demo 用 pydegensac 估计单应（第 140-150 行）。
- **环境**：python 3.10、PyTorch 2.8、CUDA 12.6、kornia 0.8.1（README 第 106-107 行）。

## 8. CasP — 通用（另有 MINIMA 微调版），**权重拿不到**

- **出处**：Chen et al., ICCV 2025 highlight（README）。代码 `pq-chen/CasP@cee1dc3`（与服务器一致），Apache-2.0。
- **权重**：README 第 34 行明说「由于资助方的披露限制，模型权重只通过 demo 提供」。Space 通过 `HF_TOKEN` 从私有仓库 `pq-chen/CasP` 拉取（`app.py:296-303`）。**zero-shot 跑不了**，除非向作者申请。
- **输入约定（读到）**：`data_mode == "gray"` 时 `/255`（`demo.py:137-141`），`image_size` 默认 1152，步长 32（`demo.py:59`；`app.py:195-197`）。

## 9. MapGlue — 光学遥感跨模态，**代码和权重都拿不到**

- `PeihaoWu/MapGlue@81b06ec` 只有 README。HF Space `wupeihao/MapGlue@2ce03af` 的 `app.py` 从 `./weights/fastmapglue_model.pt` 加载 TorchScript，缺失时提示需要 HF_TOKEN（`app.py:90-103`）。
- 输入：uint8 RGB 张量，不做归一化（第 132-149 行）。

## 10. RoMa（v1）— 通用锚点 / 消融用

- **出处**：Edstedt et al., CVPR 2024。读的代码是 `Parskatt/RoMa@77f8d68`（2026-01-23）；**服务器是 `edd1b8b`（2025-02-16）**。MIT 许可（DINOv2 部分为 Apache，README 第 107-108 行）。
- **权重**：GitHub release `Parskatt/storage/releases/download/roma/roma_outdoor.pth`；DINOv2 从 `dl.fbaipublicfiles.com` 下载（`model_zoo/__init__.py:8-15`）。**服务器需要能访问 fbaipublicfiles**（推断；访问不了就要手动下载）。
- **输入约定（读到）**：
  - 路径输入：`Image.open` → 检查不是 16 位（`check_not_i16`）→ `convert("RGB")`（`matcher.py:535-539`）。
  - 然后 resize 到 (560,560)，`ToTensorScaled(/255)`，ImageNet Normalize（`utils.py:164-172`；`matcher.py:827-830`）。
  - upsample 阶段用 864²（`roma_models.py:35-36`）。
  - 张量输入：要求 3 通道、边长是 14 的倍数，**不做归一化**（`matcher.py:544-550,832-841`）。
  - 采样：`sample_thresh = 0.05`，`num = 10000`，KDE 平衡采样（`matcher.py:565,613-644`）。
  - 输出：[-1,1] 坐标 → `to_pixel_coordinates` 用 `W/2*(x+1)`（第 730 行），像素中心在 i+0.5。
  - 构造时要求 float32 matmul precision 为 highest，否则 RuntimeError（`roma_models.py:45`）。
- **环境**：README 写「在 Linux python 3.12 上测试」，用 uv 安装。

## 11. RoMa v2 — 通用 SOTA 锚点，推荐

- **出处**：Edstedt, Nordström et al., arXiv 2511.15706（2025-11）。代码 `Parskatt/romav2@95c9968`，MIT 许可（DINOv3 为自定义许可，README 第 117-119 行）。
- **权重**：GitHub release `Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt`，构造时自动下载（`romav2.py:98-101`），整个 state_dict 一起加载，没有看到单独下载 DINOv3（推断）。
- **输入约定（读到）**：
  - `_load_image`：路径输入 → `convert("RGB")` → uint8 张量；ndarray 或张量必须是 3 通道；**只有 uint8 才 ÷255**（`romav2.py:273-298`）。
  - bicubic + antialias 缩放到 `H_lr × W_lr`（默认 setting `precise` 为 800²，hr 为 1280²，`romav2.py:79,152-158,310-338`），拉伸。
  - ImageNet 归一化在特征提取器内部（`normalizers.py:4-9`，`features.py:5,32`）。
  - `sample(preds, num)` 从 warp 采样（`romav2.py:372-439`），坐标约定同 RoMa。
  - 仓库自带 `benchmarks/satast.py`（卫星–宇航员照片基准）。
- **环境**：python ≥ 3.10，torchvision ≥ 0.23（`pyproject.toml:9,15`）。

## 12. DKM v3 — 通用（服务器已有）

- `Parskatt/DKM@ef57565`（与服务器一致）。权重：GitHub release `Parskatt/storage/releases/download/dkmv3/DKMv3_outdoor.pth`（`model_zoo/__init__.py:3`）。
- 输入：resize 到 540×720，upsample 到 864×1152（`model_zoo/__init__.py:15-25`；`dkm.py:569`），ImageNet 归一化。**路径输入时不会 `convert("RGB")`**（`dkm.py:663`），灰度图需要先自己转成 RGB PIL 图。

## 13. ELoFTR / LoFTR — 通用（LoFTR 服务器已有，ELoFTR 缺）

- **ELoFTR**：Wang et al., CVPR 2024。代码 `zju3dv/EfficientLoFTR@07e9c14`，Apache-2.0。
  - 权重：Google Drive（README 第 33 行）；HF `zju-community/efficientloftr@face1a7`（transformers 版）、`vismatch/eloftr`。
  - 输入：`IMREAD_GRAYSCALE` → resize 到 32 的倍数 → `/255.`（README 第 57-66 行）。**必须调用 `reparameter(matcher)`**（第 53 行）。
  - `MATCH_COARSE.THR = 0.2`（full 模型）；`NPE` 建议设为长边（`full_config.py:33,37`）。
  - 环境：python 3.8、torch 2.0.0+cu118（`environment.yaml`、README 第 28 行）。
- **LoFTR**：`zju3dv/LoFTR@df7ca80`（与服务器一致）。
  - 权重：Google Drive。
  - 输入：灰度 `/255`，边长是 8 的倍数，demo 默认 640×480（`demo/demo_loftr.py:52`），`THR = 0.2`（`cvpr_ds_config.py:32`）。
  - 环境：python 3.8、pytorch 1.8.1、cudatoolkit 10.2（`environment.yaml`）。

## 14. SuperPoint + LightGlue — 稀疏锚点

- `cvg/LightGlue@eb42fee`，Apache-2.0（SuperPoint 部分另有限制性许可，README 第 183 行）。权重在 GitHub release（`superpoint.py:144`）。
- **输入（读到）**：
  - `load_image` → RGB → `/255.0`（`utils.py:72-94`）。
  - `extract()` 按长边**缩放到 1024**（`superpoint.py:115-117`、`disk.py:17-18`、`aliked.py:631-632`；`utils.py:14-18` 用 kornia antialias）。
  - SuperPoint 内部转灰度（`superpoint.py:155-156`），DISK/ALIKED 需要 RGB（灰度会被复制成 3 通道）。
  - 关键点用 `(kp+0.5)/scale-0.5` 映射回原图（`utils.py:146`）。
  - 默认 `max_num_keypoints = None`，SuperPoint 的 `detection_threshold = 0.0005`。
- 对 512² 输入：官方默认会先放大到 1024；是否用 `resize=None` 需要在评测协议里定死（推断）。

## 15. LoMa — 2026 稀疏锚点（LightGlue 的替代）

- Nordström, Edstedt et al., ECCV 2026 Oral（arXiv 2604.04931）。代码 `davnords/LoMa@8fb59c4`，MIT 许可。
- 权重：GitHub release `davnords/storage/releases/download/loma/loma_{B,B128,L,G,R}.pt(h)`（README 第 81-85 行）。
- 输入：DaD 检测器 `convert("RGB")` → 长边 1024 → `/255`（`detector/dad.py:36,167-180`）。`match()` 返回原图像素坐标（`loma.py:404-438`），默认 2048 个点（`loma.py:232`）。
- LoMa-R 旋转不变，作者说适合航拍（README 第 26 行）。

## 16. RIPE — 通用（服务器已有），不推荐进主表

- Künzel et al., ICCV 2025。`fraunhoferhhi/RIPE@b173418`（服务器 `e90c844`，比我读的旧）。
- 许可：Fraunhofer 学术许可，禁止商业使用（`LICENSE:11-13`）。
- 权重：`cvg.hhi.fraunhofer.de/RIPE/ripe_weights.pth`（`vgg_hyper.py:20`）。
- 输入：`decode_image(...)/255` → `resize_image`，每边夹到 [512,768]（README 第 62-63 行，`utils.py:57`）→ VGG + InstanceNorm（`vgg.py:11`）。`detectAndCompute(threshold=0.5, top_k=2048)`。

## 17. GIM — 通用（未深入）

- `xuelunshen/gim@f09105a`，ICLR 2024。提供 gim_roma / gim_dkm / gim_loftr / gim_lightglue；权重在 Google Drive / OneDrive（README 第 73 行），HF 镜像 `vismatch/gim-dkm`、`vismatch/gim-lightglue`。
- rsim 评测里包含 GIM–LightGlue 和 GIM–DKM，没有进入前列。

## 18. 看过但排除的深度方法

- **RMSO-ConvNeXt**（`yeyuanxin110/RMSO-ConvNeXt@19c8faa`）：模板匹配（只估平移）；README 说部分工具是 Win64 下的 `.pyd` 加密文件。
- **SOMA**（arXiv 2511.13168）、**PromptMID**（arXiv 2502.18104）：没有找到代码链接。
- **RRSI**（Ye et al., arXiv 2609.06343）：README 写「将开源」。
- **BIFT**（`yyxgiser/BIFT@3b8d170`）：目前只有数据集。
- **MASt3R / DUSt3R**：rsim 指出它们假设透视几何和景深变化，与正射影像的前提冲突。
