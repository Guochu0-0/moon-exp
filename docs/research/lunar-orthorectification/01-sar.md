# A. 月球 SAR / 雷达数据：存档产品几乎都没控制，大多只到参考球面

> 调研日期：2026-10-28。出处全称与链接见 [sources.md](sources.md)。
> 阅读程度：【全文】读过原文相关段落；【摘要】只读到摘要或检索摘录；【题录】只核了题名；【官方文档】PDS 数据集说明（.cat）、产品标签（.lbl/.txt）、SIS、机构网页。自己的换算和判断一律标【推断】。
> 术语（先用白话）：
> - **正射**：把影像里因地形高低造成的位置错动按 DEM 改正掉，使每个像素落在它真实的地面位置。只投到一个固定半径的球面上不算正射，地形位移还在。
> - **控制**：用控制点或参考数据（如 LOLA）把影像整体平移、旋转到正确位置。不控制，影像的位置只取决于航天器轨道和时间记录的精度。
> - **斜距几何**：SAR 按「离雷达多远」排像素，所以地面高处会被画得更靠近雷达一侧。地面距离向位移约等于高差 / tan(入射角)（公式见 Eke 2014【全文】、Kirk 2010【全文】）。

## A.0 一句话结论

- 月球轨道 SAR 只有三台：Chandrayaan-1 Mini-SAR、LRO Mini-RF、Chandrayaan-2 DFSAR。嫦娥七号轨道器带一台小型 SAR，截至本文写作没有找到它已经发射的官方报道。
- 三台的标准存档产品都**没有做控制**。Mini-SAR/Mini-RF 的 Level 2 写的是 "orthorectified"，但标签里的参考面是半径 1737.4 km 的球。Kirk 2010 和 Fassett 2024 都说这批产品「投到球面 / 没有严格正射」。DFSAR 的地理定位逐像素用了 LOLA DEM（约 118 m），但没有找到公开的定位精度数字。
- 做过控制并正射的 Mini-RF 产品只有几类：USGS 的极区控制镶嵌（Kirk 2013，误差 ≤30 m），Fassett 2024 正射到 SLDEM/LOLA 的全月镶嵌，Xu 2025 的配准数据集（约 110 m）。它们都不是 PDS 标准的逐条带 Level 2 产品。

## A.1 Chandrayaan-1 Mini-SAR：150 m 分辨率，极区镶嵌相对 LOLA 偏到约 5 km

| 项目 | 内容 | 出处 |
|---|---|---|
| 成像方式 | 侧视 SAR，混合极化：发圆极化，相干接收 H、V 两路，可算全部 4 个 Stokes 参数 | Kirk 2010【全文】；PDS `inst.cat`（MRFFR）【官方文档】 |
| 波段 | S 波段 12.6 cm | Kirk 2010【全文】 |
| 分辨率 | 150 m，地面处理成 75 m/像素，幅宽 13 km | Kirk 2010【全文】 |
| 入射角 | 名义约 33°，跨幅宽变化 ±2.5° | Kirk 2010【全文】；Eke 2014 写天底角约 33°【全文】 |
| 覆盖 | 2009 年 2–4 月，两极 80° 以上 >95% | Kirk 2010【全文】 |
| 产品等级 | PDS 有 PDR 原始、Level 1（CDR，β0）、Level 2（CDR-MAP）、Level 3 极区镶嵌，共用 Mini-RF 的数据格式 | PDS `mrffr_*_ds.cat`【官方文档】 |
| Level 2 几何 | 数据集说明原话："The Level 2 CDRs differ from the Level 1 products in that they are orthorectified and are re-sampled into oblique cylindrical map projections." 坐标系写的是 "Moon Principle Axis (PA) coordinate system" | `mrffr_5_cdr_map_ds.cat`【官方文档】 |
| Level 3 | 极区镶嵌，"ingesting the Level 2 stokes parameters into the ISIS software" | `mrffr_5_cdr_mosaic_ds.cat`【官方文档】 |
| 控制 | 无。Eke 2014："The currently available mosaics are neither controlled ... nor orthorectified to tie the images to an underlying base map such as that provided by the LOLA DEM." | Eke 2014【全文，arXiv 版】 |
| 定位精度 | 镶嵌图相对 LOLA DEM 基准可偏到约 5 km（Mini-RF 约 2 km） | Eke 2014【全文】 |
| | Xu 2025 配准时测到 Mini-SAR 偏移大多集中在 −2850 至 −3000 m，随时间稳定，个别 −2500 至 −3200 m | Xu 2025【全文】 |
| 发布 | PDS Geosciences Node（`/lunar/ch1-orb-l-mrffr-1-pdr-v1/`） | 【官方文档】 |

## A.2 LRO Mini-RF 单站模式：标准产品是「Vexcel 投到 1737.4 km 球面、未控制」

### A.2.1 仪器与模式

| 项目 | 内容 | 出处 |
|---|---|---|
| 工作期 | 单站（同一航天器收发）2009-07 至 2010-12，此后发射机失效 | `mrf_inst.cat`【官方文档】；errata："Since 26 December 2010, the Mini-RF instrument has not transmitted the RF energy required to create radar images" 【官方文档】 |
| 波段 | S 波段 2.38 GHz（12.6 cm）；X/C 波段 7.14 GHz（4.2 cm，历史上叫 X 波段） | `mrf_inst.cat`【官方文档】；Kirk 2010【全文】 |
| 极化 | 混合极化：发右旋圆极化，收 H、V 及其相位。每像素 4 个数：H 功率、V 功率、HV* 互功率实部、虚部 | `mrflro_5_cdr_map_v1_0_ds.cat`【官方文档】 |
| 分辨率 | baseline：150 m、16 视；zoom：15 m × 30 m、8 视 | `mrf_inst.cat`【官方文档】 |
| | zoom 「15-m (along-track) by 30-m (across-track) ... processed at a grid spacing of 7.5 m/pixel」；baseline 75 m/像素 | Kirk 2010【全文】 |
| | 实测一个 Level 2 标签：PixelResolution 31.77 m（距离向）、LineResolution 14.59 m（方位向），range_looks 1，azimuth_looks 8，SARPixelSpacing 7.5 m | `LSZ_06700_2CD_EKU_47N277_V1.TXT`【官方文档】 |
| 入射角 | 平坦区名义约 49°；因地形，局部入射角至少覆盖 20°–80° | Fassett 2024【全文，经网页抽取】 |
| | 标签例：`INCIDENCE_ANGLE = 49.77`、`50.47`、`50.75`、`53.61` | 几个 Level 2 标签 + SIS 样例【官方文档】 |
| | Jackson 坑定向立体：常规 44°–48°，降低视角的一轨 24°–29° | Kirk 2010【全文】 |
| X 波段 | errata："The X-Band data have not been calibrated." | errata.txt【官方文档】 |

### A.2.2 产品等级与每级的几何处理

| 等级 | 数据集 ID | 几何处理（按官方文档原话） | 出处 |
|---|---|---|---|
| Level 1 CDR | LRO-L-MRFLRO-4-CDR-V1.0 | "SAR images in range and azimuth orientation with pixel values in beta naught"；"Level 1 data products are not dependent on definition of a coordinate system." SIS 写 Level 1 处理含 "Slant Range to Ground Range Conversion" | `mrflro_4_cdr_v1_0_ds.cat`、SIS §4.2.3.2【官方文档】 |
| | | Kirk 2010："Pixel coordinates in a Level 1 image are not georeferenced, but instead reflect the geometry of the SAR imaging process" | Kirk 2010【全文】 |
| Level 2 CDR（逐条带） | LRO-L-MRFLRO-5-CDR-MAP-V1.0 | "The Level 2 CDRs differ from the Level 1 products in that they are orthorectified and are re-sampled into oblique cylindrical map projections."（数据集说明与 SIS §4.3.2.2 同文）。SIS §4.2.3.3："Image Formation, wherein the L1 image, and optionally an available DEM or GCP file(s) are used to project the image into desired projection." | `mrflro_5_cdr_map_v1_0_ds.cat`、`mrflro_dp_sis.pdf`【官方文档】 |
| Level 3 极区镶嵌 | LRO-L-MRFLRO-5-CDR-MOSAIC-V1.0（2012） | 数据集说明：用 ISIS 摄入 Level 1 S1、CPR 手工拼。标签说明："orthorectifying them using a combination of spacecraft spice data and spatially tying the data to the LOLA base map"。地图半径 1737.4 km，128 与 32 ppd（标签另有 256 ppd 版本） | `mrflro_5_cdr_mosaic_v1_0_ds.cat`、`lsz_06899_3s1_phu_90n000_e_v1.lbl`、`lsz_xxxxx_3s1_pfu_90n000_v1.lbl`【官方文档】 |
| Level 3 全月镶嵌 | LRO-L-MRFLRO-5-GLOBAL-MOSAIC-V1.0（2014 发布） | 用 ISIS 摄入 **Level 2** 文件，`map2map` 重投到 128 ppd 简单圆柱投影后按轨道号顺序拼接："Different viewing geometries are not differentiated"。标签：SIMPLE CYLINDRICAL，半径 1737.4 km，237 m/像素 | `mrflro_5_globalmosaic_ds.cat`、`global_cpr_128ppd_simp_0c.lbl`【官方文档】 |
| Level 3 InSAR/立体 DEM | — | SIS 附录规划由 Sandia 用连续模式数据做立体 DEM（"altitude in meters above the 1,737.4 km radius lunar sphere"）；PDS 有 InSAR 连续模式 Level 1 数据集，未找到已发布的 DEM 产品 | SIS §6.1、`mrflro_4_cdr_insar_v1_0_ds.cat`【官方文档】 |

### A.2.3 核心矛盾：「orthorectified」指的是用 Vexcel Ortho 处理器投到 1737.4 km 球面，没有可确认的地形改正，也没有控制

**三方说法原文对照**

| 出处 | 原话 | 阅读程度 |
|---|---|---|
| PDS 数据集说明 / SIS | "they are orthorectified and are re-sampled into oblique cylindrical map projections" | 【官方文档】 |
| Level 2 标签 DESCRIPTION（SIS 样例） | "The Level 2 data products are orthorectified and are re-sampled into oblique cylindrical map projected SAR images." | 【官方文档】 |
| Kirk 2010（USGS，Mini-RF 团队成员） | "The standard products at these levels are uncontrolled, and are (so far) projected onto the lunar reference sphere, although the Vexcel software is capable of orthorectifying images based on a DTM." 同文另一处："As noted, the standard Level 2 products are uncontrolled and unrectified." | 【全文】 |
| Kirk 2010 的实验 | 自制两套镶嵌，一套 "uncontrolled and unrectified (projected onto the lunar reference sphere)"，东视与西视之间净偏移近 10 km，"mainly as a result of uncorrected stereo parallax"；相邻同侧轨道间约 330 m | 【全文】 |
| Eke 2014 | PDS 镶嵌 "neither controlled ... nor orthorectified"（指 Mini-SAR 和 Mini-RF 的极区镶嵌） | 【全文，arXiv 版】 |
| Kirk 2013（LPSC） | 称早先 "uncontrolled but orthorectified (i.e., projected onto a detailed topographic model to remove parallax distortions) mosaics of the lunar poles"，"have recently been archived in the NASA PDS"，即上面 2012 年的 Level 3 极区镶嵌 | 【全文】 |
| Fassett 2024（Mini-RF 团队） | "As released, the data were reduced onto a lunar sphere and so lacked rigorous rectification to lunar surface topography"；结论："The Vexcel processor-derived data that were archived in the PDS were not fully orthorectified or correctly selenolocated" | 【全文，经网页抽取】 |
| Xu 2025（中科院空天院） | "The calibrated and mapped Mini-SAR/Mini-RF images (Level-2 data) ... have gone through a map projection and orthorectification process with DEM data. Though these procedures eliminate the majority of location errors from the terrain, the final processed SAR data ... exhibits significant geometric distortion." | 【全文】 |

**Level 2 标签和 Vexcel 参数文件里的事实**（2026-10-28 从 PDS 直接下载核对）

1. `.lbl` 的地图投影对象写的是球：
   ```
   MAP_PROJECTION_TYPE = "EQUIRECTANGULAR"     （低纬；极区为 "OBLIQUE CYLINDRICAL"）
   A_AXIS_RADIUS = 1737.4 <km>   B_AXIS_RADIUS = 1737.4 <km>   C_AXIS_RADIUS = 1737.4 <km>
   MAP_RESOLUTION = 4096.0 <pix/deg>   MAP_SCALE = 0.0074031617246459 <km/pix>
   ```
   （`lsz_06700_2cd_eku_47n277_v1.lbl`）【官方文档】
2. 每个 Level 2 有一份 Vexcel 处理参数文件（`.TXT`，由标签 `^TEXT` 指向）。其 `OrthoProduct` 段里：
   - `processor_name: Ortho`、`processing_agency: Vexcel`，`gcp_used: 0`、`dem_used: 1`；
   - `lunar_model { major: 1737400 minor: 1737400 terrain_height: 0.000000 }`；角点坐标的高程列均为 0；
   - 附一个 `dem { PROJECTION_NAME: EQA ... ROWS: 3000 COLS: 1000 ... PIXEL_WIDTH: 0.02 PIXEL_HEIGHT: 0.0067 UNITS: DEGREES }` 段，即一块 20° × 20° 的等经纬格网瓦片；**没有写 DEM 的名称或来源**。【官方文档】
3. 抽查 11 个 Level 2 参数文件（轨道 300–334、4600–4611、6700–6701，2010-02 至 2011-01 生成），**全部** `gcp_used: 0`、`dem_used: 1`，每个只引用一块 20° 瓦片。多例条带超出瓦片范围：轨道 6700 条带纬度 43.8–50.4°N，瓦片是 49.9–70°N；轨道 4600 条带 26.1–42.1°N，瓦片 9.9–30°N；轨道 4611 条带 27.0–42.9°S，瓦片 10–30.1°S。【官方文档；抽样是我做的】
4. 标签 `SOFTWARE_NAME` 有的写 `"Ortho"`（版本 9.0.320，2010 年初生成），有的写 `"ISIS"`（3.2.0，2011 年生成），但这些产品的参数文件都是 Vexcel Ortho 10.0.085 的输出。【官方文档】
5. 坐标系写法前后不一。Level 2 数据集说明写 PA（主轴系），SIS §4.4.2 和全月镶嵌数据集说明写 ME（平地球/极轴系）。两者在月面相差约 875 m（DE440）/ 约 1 km（LRO 白皮书），见 [03-geodesy.md](03-geodesy.md)。【官方文档】

**判断**【推断】：
- PDS 文档里的 "orthorectified" 是 Vexcel 第二级处理器（名字就叫 "Ortho"）这一步的名称。该处理器能吃 DEM（SIS："optionally an available DEM"），但存档产品的几何参考是 1737.4 km 球（标签半径、`terrain_height: 0`、角点高程 0），而且一律 `gcp_used: 0`，没有控制。
- `dem_used: 1` 说明处理时挂了一块 DEM 瓦片，但 DEM 来源没有记录。格网约 0.02° × 0.0067°，在赤道约 600 m × 200 m，是很粗的格网。瓦片又经常盖不住整条带。DEM 有没有真正参与改正、在哪些像素上参与，从标签判断不了。Mini-RF 团队自己（Kirk 2010、Fassett 2024）的结论是「投到球面 / 没有严格正射」，应以此为准。较稳妥的说法是：**PDS Level 2 是「投到参考球面、至多带有一块来源不明的粗 DEM、未控制」的地图投影产品；"orthorectified" 不能理解为按 LOLA/SLDEM 一级的 DEM 做了地形改正。**
- Kirk 2013 说 2012 年的 Level 3 **极区**镶嵌是「未控制但正射」，标签也写了 "tying the data to the LOLA base map"。这套镶嵌由 USGS ISIS 从 Level 1 重新投影，处理链与 Level 2 不同。Eke 2014 说它「既没控制也没正射」，与 Kirk 2013 冲突，可能因为 Eke 用的版本或语义不同，没有核实到底。

### A.2.4 Mini-RF 存档产品的位置误差：典型约 0.9 km，最大约 3 km，主因是时间记录误差

| 误差描述 | 数值 | 出处 |
|---|---|---|
| 与其他 LRO 数据中同一坑的平均偏移 | 约 900 m（Fa & Eke 2018、Nypaver 2021，经 Fassett 2024 转述；Fassett 2024 自己也得到中位约 0.9 km） | Fassett 2024【全文，经网页抽取】 |
| 偏移随轨道号在 0–3000 m 间线性变化，相邻轨道间有突变 | 0–3000 m | Xu 2025【全文】 |
| 极区单站数据（2009-07 至 2010-07）起始时间误差从 0 线性增长到 1.0–1.3 s 后归零，反复出现；按轨道号分段线性改正后，东、西视图之间的错位从约 3 km 降到约 30 m | 约 3 km → 约 30 m | Kirk 2016【全文】 |
| 1.0–1.3 s 换算成沿轨位移：LRO 50 km 轨道地速约 1.6 km/s，对应约 1.6–2.1 km | — | 【推断】 |
| 未控制镶嵌中东视与西视的错位（含视差） | 约 1–3 km（Kirk 2013）；近 10 km（Kirk 2010 LCROSS 区） | Kirk 2013【全文】；Kirk 2010【全文】 |
| 极区 PDS 镶嵌相对 LOLA | 可达约 2 km | Eke 2014【全文】 |
| 当时 SPICE 星历精度 | "on the order of 100 m" | Kirk 2010【全文，引 Archinal 2010】 |

## A.3 Mini-RF 双站观测（2012 起）：地面发射、LRO 接收，新版产品按 LOLA DEM 成像但未控制

| 项目 | 内容 | 出处 |
|---|---|---|
| 方式 | 地面站发、Mini-RF 收。2011 年 4 月用 Arecibo（S 波段 2380 MHz）试验，2012 年起转为常规双站；2016 年 2 月测试 Goldstone DSS-13（X 波段 7147 MHz），2017 年起加入 X/C 波段 | `mrf_inst.cat`【官方文档】；双站处理说明【官方文档】 |
| 极化 | 地面发左旋圆极化（LHCP），Mini-RF 收 H、V | `mrf_bistatic_ds.cat`【官方文档】 |
| 分辨率 | "averages approximately 100 m"；旧版（PDS3 v2）EDR 典型距离向约 40 m、方位向约 2 m，RDR 重采样到 100 m 方像素 | `mrf_inst.cat`、`mrf_bistatic_ds.cat`【官方文档】 |
| | 新版（PDS4，2024）：先在 2 m × 20 m 网格上做时域后向投影（TDBP），Stokes 产品降采样到 100 m × 100 m | 双站处理说明 v1.0（2024-09）【官方文档】 |
| 几何（旧版 PDS3） | 网格以航天器轨迹定义，另附逐像素经纬度、入射角等的 DDR 文件："Level 1 data products are not dependent on definition of a coordinate system. However, latitude and longitude information is provided in the level 1 DDR files" | `mrf_bistatic_ds.cat`【官方文档】；Kirk 2016【全文】 |
| 几何（新版 PDS4） | 用 SPICE 星历；网格是斜圆柱投影（以条带中线为赤道）；成像时的表面交点用 **16 ppd 全月 DEM**，"carefully constructed by resampling the LOLA LDEM256 product ... down to a resolution consistent with LDEM16"；坐标系 MOON_ME；背景层 Height 写 "Local height above the 1737.4 km reference sphere"，Slope 写 "...the DEM onto which the data are processed" | 双站处理说明 v1.0【官方文档】 |
| 控制 | 处理说明未提控制点或与 LOLA 配准。Kirk 2016 用时间改正 + 正射把 Cabeus 一景对到受控单站镶嵌：该景原始相对 LOLA 偏约 2.7 km（对应约 1.6 s 时间误差），改正后误差约 30 m，小于 100 m 像素 | 处理说明【官方文档】；Kirk 2016【全文】 |
| 结论 | 新版双站产品等于是「用 SPICE 定位、按约 1.9 km 格网的 LOLA DEM 成像、未控制」【推断：16 ppd 在赤道约 1.9 km，来自 ULCN 文档同样的换算】 | — |
| 发布 | PDS Geosciences：`lro-l-mrflro-2_3_5-bistatic-v2/v3`（PDS3）、`urn-nasa-pds-lro_minirf_bistatic`（PDS4） | 【官方文档】 |

## A.4 Chandrayaan-2 DFSAR：L/S 双频、可全极化，定位逐像素用 LOLA DEM，但没有公开精度

| 项目 | 内容 | 出处 |
|---|---|---|
| 波段 | L 波段 1.25 GHz（24 cm），S 波段 2.5 GHz（12 cm） | Putrevu 2020【全文】 |
| 极化 | 单 / 双 / 混合 / 全极化 | Putrevu 2020【全文】；Bhiravarasu 2021【全文，arXiv 版】 |
| 分辨率 | 2–75 m（可编程）；幅宽 10 km；轨道高度名义 100 km | Putrevu 2020 表 1【全文】 |
| 入射角 | 9.6°–36.9° | Putrevu 2020【全文】 |
| | 实例：Peary 坑 L 波段全极化 26°，像素 0.6 m（方位）× 9.6 m（距离），投到 25 m/像素；Manzinus C L 波段 CP 30°；Byrgius C 约 24°–26° | Bhiravarasu 2021【全文】 |
| 产品等级 | L0A/0B：未 / 已距离压缩原始数据；**L1A**："seleno-tagged single-look complex (SLC)"；**L1B**："seleno-tagged ground-range product"；**L2**："seleno-referenced product"；成像算法 Range-Doppler | Bhiravarasu 2021【全文】 |
| 地理定位 | "Each pixel of a given slant image is tagged to a particular seleno-location ... using the slant range grid generated from the orbit attitude information, a digital elevation model (DEM) available at ~118.5 meters spacing from the LRO ... LOLA ... and radar parameters." | Bhiravarasu 2021【全文】 |
| 地形改正 | 作者在分析中用斜距网格文件把数据投到月面坐标，并说明 "The terrain correction of DFSAR data is performed by utilizing Chandrayaan-2 spacecraft orbit information and (per pixel) topography ... from a global DEM at ~118 m spacing available from the LOLA instrument"。图 10 注明某坑东缘 "distorted due to improper terrain correction" | Bhiravarasu 2021【全文】 |
| 控制 | 未找到用控制点或 LOLA 做整体配准的官方说明 | — |
| 定位精度 | **未找到**公开数字。Chandrayaan-2 轨道器定轨精度也未找到 | — |
| 极区镶嵌 | ISRO 2025-11-08 发布 L 波段全极化极区（80°–90°）"Polar Mosaic products (Level 3C)"，约 1400 景，最高 25 m/像素；页面未说明 DEM、正射与精度 | ISRO 网页【官方文档】 |
| 工具 | SAC 的 MIDAS 软件 2022 年时："The module for geocoding and ortho-rectification of DFSAR data using LOLA DEM is under development" | Tarun 2022 LPSC【全文】 |
| 发布 | ISRO ISSDC PRADAN，PDS4 格式（需注册登录；产品 SIS 未读到） | ISRO 网页【官方文档】 |
| Current Science 2020 专刊 | 读了 Putrevu 2020（仪器）；专刊中是否有单独的数据处理与几何精度文章，**未找到** | — |

## A.5 地基雷达：按参考球体做时延-多普勒成像，70 cm 的 Level 2 再按 Clementine 光学底图几何校正

| 数据 | 成像方式 / 波段 | 分辨率 | 几何与定位 | 出处 |
|---|---|---|---|---|
| Arecibo→GBT 70 cm 全近侧图 | Arecibo 发圆极化脉冲，GBT 收两种圆极化；SAR 式「分块聚焦」时延-多普勒 | 最细约 320 m × 450 m，成图 400 m | 映射到经纬度时 "For most mapping, we assume a single mean radius for the Moon, but more detailed topography (if available) could be used to orthorectify the final image"。各视之间用互相关找偏移、橡皮拉伸后叠加；"Data being archived with the NASA PDS at 'Level 2' have also been geometrically corrected to match the Clementine 750-nm optical image base" | Campbell 2007（TGRS 投稿稿，PDS 文档目录）【全文】 |
| Arecibo→GBT 12.6 cm（S 波段）全近侧图 | 连续波 + 伪随机码，分块聚焦 | 约 40 m，4 视功率图 80 m 采样；CPR 100 视 400 m | 「聚焦成像与映射到经纬度格网沿用 70 cm 的步骤」；波束中心位置用理想波束图匹配首零点来校准指向；PDS 数据集说明：功率按 "a reference spherical shape" 归一化，INC 文件是 "incidence angle ... for a reference sphere"。未提控制或正射 | Campbell 2010 Icarus（修订稿）【全文】；`slrm_ds.cat`【官方文档】 |
| GBT 双站 | 即上面两套：Arecibo 发、GBT 收 | — | 同上 | 同上 |
| Haystack 3.8 cm | 早期全月雷达图 | 2–4 km | 未读到原文（Zisk 1974），只见 Campbell 2007 引用 | Campbell 2007 引言【全文】；Zisk 1974【题录】 |
| Goldstone 3.5 cm 干涉 | 地基雷达干涉测高（两极） | 150 m 平面，约 50 m 高程 | 产出 DEM，不是正射影像 | Margot 1999【摘要】 |
| 旧版 EB-MOON 标签 | 12.6 cm 等早期数据 | 100 像素/度 Mercator | 标签称经纬度只精确到度（检索摘录，未打开标签核对） | 检索结果【摘要】 |

【推断】地基雷达图的整体位置取决于地月几何、波束指向和星历，只有 70 cm 的 Level 2 明确按 Clementine 底图做了几何校正。Clementine 底图本身相对现行 LRO 框架就有数百米到数公里的误差（见 [02-optical.md](02-optical.md) B.6），所以这些图不宜当作 SAR 侧的几何基准。

## A.6 测深雷达和规划中的 SAR：Apollo 17 ALSE、Kaguya LRS 不是成像 SAR；嫦娥七号带一台小型 SAR

- **Apollo 17 ALSE**（Apollo Lunar Sounder Experiment）：轨道测深雷达，用于探测月表以下的层位，不是成像 SAR。Porcello 1974【题录】，频率等参数未核对。
- **Kaguya LRS**（Lunar Radar Sounder）：测深雷达，观测月海下层位。Ono 2009 Science【题录】，参数未核对。
- **嫦娥七号轨道器 MSAR**（微型合成孔径雷达）：多极化，分辨率优于 1 m，幅宽 5–20 km，视场角 15°–45°，用途是与高分辨率立体相机一起获取全月表与永久阴影区地形。未给波段与入射角。Wang 2024 NSR【全文，经网页抽取】。发射状态：截至 2026 年 5 月的报道说计划在 2026 年下半年发射，此后是否已经发射**未核实**。
- 中文综述 Lu 2025（遥感学报，空天院）有月球 SAR 对比表，含嫦娥七号 SAR。正文 PDF 的中文字体无法提取，只读到题录和页面要点【题录】。
- **KPLO（韩国 Danuri）**：6 台载荷是 LUTI、PolCam、KMAG、KGRS、DTNPL 和 NASA 的 ShadowCam，没有 SAR。出处是 KPLO 新闻资料包（ShadowCam 网站托管）及 KASI 页面，只读到检索摘录【摘要】。**NASA、ESA**：未找到已规划的月球轨道 SAR。
- **Mini-RF 自身**：Chan 2025 EPSC-DPS 摘要称，团队已用自研双站处理器手工重处理少量 X/C 波段单站数据，打算把串扰改正推广到整个存档。摘要未提几何与定位【摘要】。
