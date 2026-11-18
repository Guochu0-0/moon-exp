# 月球 Val 首批 32 张图均导出 4096 个 ReDFeat 检测点

**摘要**

为地球与月球的光学–SAR 检测对照准备逐图数据。沿用 B0 的官方 ReDFeat 权重和检测配置，月球 Val 的 16 对 patch 均成功导出匹配前坐标和分数，32 张图均达到每图 4096 个输出点的上限。WHU-OPT-SAR 已确定首批四景，影像下载受学院内网通道阻塞。SuperPoint 暂待检测分支经过跨模态微调的可核验公开权重。

**引言**

B0 记录了 ReDFeat 在月球上的匹配输出。ReDFeat 先从光学和 SAR 各自的图像中检测候选点，再按描述子筛选对应关系。为独立检查检测阶段，这批记录保留每张图在筛选对应关系前的坐标和重复性分数。 [B0 Notes](../B0/notes.md)

**方法**

*固定月球清单。* 从 gpfs 的 `/remote-home/xufang/YGC/dataset/Moon` 读取既有 Val 光学–SAR patch，包含有标注和无标注的同名双图。对配对后的 ROI 按名称升序排列，沿索引范围均匀取 4 个 ROI，本次为 035、070、071、080。每个 ROI 内按数字 patch ID 排序，取零基索引 `floor(n/8)`、`floor(3n/8)`、`floor(5n/8)`、`floor(7n/8)`。每张图读取整个 512×512 patch。清单在检测前确定，保留源路径和窗口。 [月球样本清单](extra/data/moon_manifest.json)

*检测器和输入。* 使用官方 `Pretrained/VIS_SAR.pth`，gpfs 权重位于 `/remote-home/xufang/YGC/weights/official/redfeat/VIS_SAR.pth`，SHA-256 为 `6294dcbd7b116c02bba4b7704ed8495ead1666a81494b14630945014ffaeea24`。月球光学除以 255，SAR 前两波段之和转 dB，以 −25 dB 为下限，再按各 patch 的 2% 和 98% 分位拉伸到 [0, 1]。光学走 `forward1`，SAR 走 `forward2`。模型按 B0 对输入做逐通道标准化，在 512、431、362、304、256 五个尺度检测，3×3 NMS、重复性阈值 0.4、每层边界 10 像素，按分数稳定排序保留至多 4096 点。 [ReDFeat 官方代码](https://github.com/ACuOoOoO/ReDFeat/tree/66df80027dcb649118259a91cadbff4b50adc880) · [启动记录与配置](launch/main.json)

*验收。* 在 160 的 V100 上用清单首尾各一对小跑，通过后运行全部 16 对，推理代码提交为 `30f70be4cf1b51c207726f51a3f67ec8b556d0fb`。正式归档的键形如 `ROI_035__patch_26__optical`，每个键对应一张图的 `N×3 float32` 数组，列为 `x, y, score`，整数坐标表示输入 patch 中的像素中心，左上像素中心为 (0, 0)。零检测时数组为 `0×3`，执行失败在逐图记录中单列。 [正式关键点 NPZ](extra/data/moon_redfeat.npz) · [逐图点数和状态](extra/data/moon_redfeat_counts.json) · [小跑点数和状态](extra/data/smoke/moon_redfeat_counts.json) · [小跑关键点 NPZ](extra/data/smoke/moon_redfeat.npz)

**结果**

*月球两种模态均导出了全部预定图像。* 16 对样本对应 32 个不同的归档键，所有数组可读、数值有限，坐标实测范围为 10 至 501 像素，分数均达到阈值并按降序排列。小跑和正式批次的同图数组逐值一致。正式 NPZ 的 SHA-256 为 `756ae903fccb2946f5a61c3c62af618ee0497254711a07109223f2a38133b20b`。

| 模态 | 图像 | 执行失败 | 零检测 | 每图输出点数 | 总输出点数 |
|---|---:|---:|---:|---:|---:|
| 光学 | 16 | 0 | 0 | 4096 | 65,536 |
| SAR | 16 | 0 | 0 | 4096 | 65,536 |

表 1。月球 Val 首批逐图导出覆盖量，两种模态的每张图均达到 B0 的输出上限。

*WHU 已登记四景，影像待下载。* 作者发布的两个 Drive 目录各有 100 个同名 GeoTIFF。按共同文件名升序取前四景，登记了 8 个文件的 Drive ID 和预期字节数。执行时 154 可通过 SSH 登录，11080 与 17890 两个下载代理端口均无监听，工位机 SSH 超时。gpfs 文件系统约有 52 TB 可用，154 容器内缺少配额查询工具，用户配额未能独立确认。计划目录为 `/remote-home/xufang/YGC/dataset/WHU-OPT-SAR`，验收读到 0 个文件、0 对。GeoTIFF 的波段、NoData、配准网格和地球 patch 窗口仍待影像到位后核验。 [作者发布页](https://github.com/AmberHen/WHU-OPT-SAR-dataset) · [四景下载清单](extra/data/whu_download_manifest.json)

*SuperPoint 暂待权重。* MINIMA 附录 B 和 MIFNet 官方训练配置均冻结 SuperPoint，Project2026 加载原版 v1，Thermal SuperPoint SLAM 的公开热红外权重使用单模态 FLIR 图像训练。本批未核实到检测分支经过跨模态微调的可获取公开 SuperPoint 权重。 [MINIMA 附录 B](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Ren_MINIMA_Modality_Invariant_CVPR_2025_supplemental.pdf) · [MIFNet 训练配置](https://github.com/lyp-deeplearning/MIFNet/blob/main/training/configs/sp_remote_homography.yaml) · [Project2026 加载方式](https://github.com/cangshengmo/Project2026/blob/main/lightglue/superpoint.py#L144-L145) · [Thermal SuperPoint SLAM 训练配置](https://github.com/specarmi/Thermal_SuperPoint_SLAM/blob/master/configs/training/superpoint_flir_train_heatmap.yaml)

**讨论**

月球首批样本的检测结果已齐备。逐图点数受 4096 点上限约束，当前数据只量化实际导出的点。地球影像和 SuperPoint 的两项缺口限制了对照范围。
