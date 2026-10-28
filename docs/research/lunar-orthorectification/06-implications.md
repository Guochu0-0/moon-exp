# F. 汇总与对我们的意义：SAR 侧即使正射，也还剩几十到上百米的误差

> 阅读程度标注同 [01-sar.md](01-sar.md)。出处见 [sources.md](sources.md)。有出处的数字和【推断】分开写。

## F.1 总表：各数据源的正射、DEM、控制与定位精度

| 数据源 / 产品 | 正射？ | 用的 DEM | 控制？ | 定位精度（相对 LOLA/LRO 框架，除非注明） | 出处 |
|---|---|---|---|---|---|
| **Mini-SAR** Level 2 / 极区镶嵌（PDS） | 文档写 "orthorectified"，实为投到 1737.4 km 球（同 Mini-RF） | 未说明 | 否 | 镶嵌偏到约 5 km；Xu 2025 测得 −2.85 至 −3.0 km 的稳定偏移 | `mrffr_5_cdr_map_ds.cat`【官方文档】；Eke 2014【全文】；Xu 2025【全文】 |
| **Mini-RF** Level 1 CDR | 否（距离-方位几何） | — | 否 | 不带地理坐标（靠 SPICE 和传感器模型） | `mrflro_4_cdr_v1_0_ds.cat`【官方文档】；Kirk 2010【全文】 |
| **Mini-RF** Level 2 CDR-MAP | 文档写 "orthorectified"；标签参考面是 1737.4 km 球，`terrain_height 0`，`dem_used 1` 但 DEM 来源不明、格网约 600 × 200 m、常盖不全；团队文献称「投到球面 / 未严格正射」 | 不明（见 A.2.3） | 否（`gcp_used 0`） | 平均约 0.9 km，最大约 3 km，随轨道号漂移 | 标签【官方文档】；Kirk 2010【全文】；Fassett 2024【全文，经网页抽取】；Xu 2025【全文】；Kirk 2016【全文】 |
| **Mini-RF** Level 3 极区镶嵌（2012） | 按 Kirk 2013 是「正射」；Eke 2014 说没有 | LOLA（标签："tying the data to the LOLA base map"） | 否（Kirk 2013："uncontrolled"） | 东西视之间 1–3 km；相对 LOLA 可达约 2 km | 标签【官方文档】；Kirk 2013【全文】；Eke 2014【全文】 |
| **Mini-RF** Level 3 全月镶嵌（2014） | 否，由 Level 2 拼接 | — | 否 | 继承 Level 2 | `mrflro_5_globalmosaic_ds.cat`【官方文档】 |
| **Mini-RF** 双站（PDS4，2024） | 成像时投到 DEM | LOLA LDEM256 重采样到 16 ppd（约 1.9 km） | 否 | 未报；Kirk 2016 测过一景原始偏移 2.7 km（旧版产品） | 双站处理说明【官方文档】；Kirk 2016【全文】 |
| **Mini-RF** USGS 受控镶嵌（Kirk） | 是 | LOLA、Kaguya | 是 | ≤30 m（北极）；12 m RMS（LCROSS 区） | Kirk 2010/2013【全文】 |
| **Mini-RF** Fassett 2024 | 是 | SLDEM 237 m / LOLA 100 m / LOLA 30 m | 是（对合成 S1 互相关，整景平移） | 未报米级数字 | Fassett 2024【全文，经网页抽取】 |
| **Mini-RF** Xu 2025 | 由 ISIS 生成 Level 2 | LOLA+SLDEM 60 m | 是（整景平移） | 约 110 m（小坑），185 m（全部） | Xu 2025【全文】 |
| **DFSAR** L1A/L1B/L2 | L1 "seleno-tagged"、L2 "seleno-referenced"；定位逐像素用 DEM | LOLA 约 118 m | 未找到 | **未找到** | Bhiravarasu 2021【全文】 |
| **DFSAR** L3C 极区镶嵌（2025） | 未说明 | 未说明 | 未说明 | 未说明 | ISRO 网页【官方文档】 |
| **Arecibo 70 cm** Level 2 | 否（单一平均半径） | — | 几何校正到 Clementine 750 nm 底图 | 继承 Clementine 底图误差 | Campbell 2007【全文】 |
| **Arecibo 12.6 cm** | 否（参考球） | — | 未提 | 未报 | Campbell 2010【全文】；`slrm_ds.cat`【官方文档】 |
| **CE-1 CCD** 全月图 | 是（几何校正） | CE-1 DEM | 未读到细节 | 平面 100–1500 m | Li 2010【摘要】 |
| **CE-2 CCD** DOM（7 m）/ DEM（20 m）| 是 | CE-2 自己的 DEM（RPC + DEM） | 5 个 LRRR 绝对控制 | 对 LRRR 21–97 m；对 SLDEM 183 m；对 LRO 354 m（σ 228 m）；有周期误差 | Ren 2014【全文】；Xin 2020【全文】；Liu 2019【全文】 |
| **CE-4/5/6 着陆区** | 是 | 降落相机 DEM | 控制到 CE-2 或 NAC | 局部 cm–m；两套底图之间差约 0.4 km | Liu 2019【全文】；CE-6 摘要 + 【推断】 |
| **LROC NAC** | 是（投影） | GLD100 / NAC DTM / LOLA | 指向用 LRRR 标定，定轨用 LOLA | 约 20 m（2012），18 m（2014），人造地标约 9–10 m | Speyerer 2012【全文】；Speyerer 2016【全文，二手】；Mazarico 2018【全文】 |
| **LROC WAC** 全月镶嵌 | 是 | GLD100 | 无控制点，靠定轨 + 标定 | 约 40–45 m | Speyerer 2016【全文，二手引 2014】 |
| **GLD100** | （DEM） | — | LRO 框架 | 垂直约 10–20 m | Scholten 2012【摘要】；LROC【官方文档】 |
| **NAC DTM** | （DEM + 正射图） | — | 控制到 LOLA | 约 10 m 水平、约 1 m 垂直 | Henriksen 2017【摘要】 |
| **Kaguya TC** TCO_MAP | 是 | TC DTM | 无（SLDEM2015 时才配到 LOLA） | DTM 对 LRRR 12–22 m；对 LOLA 平均差约 6 m | Sato & Haruyama 2021【全文】 |
| **SLDEM2015** | （DEM） | — | 逐块配到 LOLA | 垂直 3–4 m | Barker 2016【摘要】 |
| **Kaguya MI** | 未找到 | — | — | 未找到 | — |
| **Clementine** 底图 / **ULCN 2005** | 底图未正射；ULCN 2005 解了半径 | ULCN 2005 点 | 控制网本身 | 100 m 至几公里（CLCN 达 8–15 km） | Archinal 2006【全文】；Speyerer 2016【全文】 |
| **Chandrayaan-1 TMC** | DEM/正射存在 | TMC DEM | Clementine + ULCN2005（检索摘录） | 未找到 | ISSDC【官方文档，经检索】 |
| **Chandrayaan-2 TMC-2 / OHRC** | ISRO 提供正射与 DEM | TMC DTM | 未找到 | OHRC 实例沿轨约 4 km（对齐前）；TMC-2 未找到 | ASP 文档【官方文档】 |
| **SMART-1 AMIE** | 未找到 | — | — | 未找到 | — |
| **Apollo 量测相机**（Apollo Zone） | 是 | 自身立体 DEM | 光束法平差 | DEM 对 LOLA 垂直 40 m（σ 37 m） | SpaceRef 2011【摘要】 |
| **Lunar Orbiter** 全月镶嵌 | 是（约 60 m） | ULCN 2005 | 控制到 ULCN 2005 | 继承 ULCN 2005 | USGS 产品页【官方文档，经检索】 |
| *（地球）Sentinel-1 IW GRD* | 否（椭球，距离向单一高度） | — | — | 要求 7 m；开箱约 2–3 m 偏差；改正后亚米 | SentiWiki【官方文档】；Small & Schubert 2022【全文】 |
| *（地球）RTC / CEOS NRB* | 是（GTC + RTC） | Copernicus GLO-30（<4 m LE90 垂直、<6 m CE90 水平） | — | 要求报告偏差 + 标准差 | HyP3 指南【官方文档】；Copernicus DEM 手册【全文】；CARD4L【摘要】 |

## F.2 三方对比：月球 SAR 比月球光学差一到两个量级，比地球 SAR 差两到三个量级

| 类别 | 典型绝对定位 | 依据 |
|---|---|---|
| 地球 SAR（Sentinel-1） | 米级，改正后亚米 | Small & Schubert 2022【全文】 |
| 月球光学，LRO 系 | 10–45 m | Speyerer 2012/2016、Henriksen 2017、Mazarico 2018 |
| 月球光学，嫦娥二号 | 100–350 m（相对 LRO/SLDEM），对 LRRR 21–97 m | Xin 2020、Liu 2019、Ren 2014 |
| 月球 SAR，重处理且有控制 | 30 m（USGS 极区，人工控制）至约 110–230 m（全月自动） | Kirk 2013/2016；Xu 2025 |
| 月球 SAR，PDS 标准产品 | 约 0.9 km 典型，至 3 km | Fassett 2024；Xu 2025；Kirk 2016 |

【推断】按比例看，PDS Mini-RF Level 2 的绝对误差约为 Sentinel-1 GRD 的 100–1000 倍。即使是最好的重处理结果（30 m），也约为 LROC NAC 的 1.5–3 倍、Sentinel-1 的 10 倍以上。

## F.3 即使做了正射，SAR 侧还会剩下哪些几何误差

**有出处的数**

| 误差来源 | 量级 | 出处 |
|---|---|---|
| 整景定位残差（自动方法） | 约 110 m（小坑），185 m（平均），233 m（大坑） | Xu 2025【全文】 |
| 整景定位残差（人工控制） | ≤30 m；12 m RMS | Kirk 2013、Kirk 2010【全文】 |
| 时间改正后的残差 | 约 30 m | Kirk 2016【全文】 |
| 正射用的 DEM 格网 | 237 m（±60°，Fassett 用）、60 m（Xu 用）、118 m（DFSAR 用） | Fassett 2024、Xu 2025、Bhiravarasu 2021 |
| SAR 分辨率 | 15 × 30 m（多视后约 30 × 30 m） | `mrf_inst.cat`；Xu 2025【全文】 |
| SAR 立体匹配误差 | 约 1 像素（光学约 0.2） | Kirk 2010【全文】 |
| 影像与 DTM 不对齐时，正射会引入「额外的局部畸变」 | 未给数值 | Kirk 2010【全文】原话："the results would contain additional local distortions caused by the misalignment of the images with the DTM" |
| 对侧视向立体里小坑「浮起」 | 未给数值 | Kirk 2010【全文】 |
| 改正不当的坑缘畸变（DFSAR） | 未给数值 | Bhiravarasu 2021【全文】 |

**量级估计**【推断，全部是自算】：
1. **DEM 格网以下的地形不会被正射掉。** 地面距离向位移 ≈ Δh · cot θ。Mini-RF 名义 θ ≈ 49°，cot θ ≈ 0.87；DFSAR θ ≈ 26°，cot θ ≈ 2.05。假设小坑深径比约 0.2：
   - 直径 500 m 的坑深约 100 m。用 237 m 格网的 DEM 正射时基本表示不出来，坑底相对坑缘仍有约 87 m 的位移，约 12 个 7.5 m 像素。
   - 用 60 m 格网的 DEM 时，同样的坑只能部分改正。
   - 同样的坑在 DFSAR 26° 入射时约 200 m。
2. **DEM 的高程误差会变成平面误差。** SLDEM 3–4 m 的垂直误差 × 0.87 ≈ 3 m，可以忽略。
3. **SAR 与 DEM 没对齐时，正射本身会把像素推错。** 若整景还错 110 m，在 20° 的坡上高程取错约 40 m，平面再错约 35 m（≈ 110 · tan20° · cot49°）。这就是 Kirk 2010 所说的「额外局部畸变」，而且随坡度在影像内变化，是**非刚性**的。
4. **合计**：以 Xu 2025 级别的自动重处理为例，SAR 侧剩下：整景平移约 100–200 m；叠加按局部坡度和亚 DEM 地形变化的几十米非刚性位移（坑内最大，与视向方向有关）；再加上 15 × 30 m 分辨率和斑点带来的约 1 像素定位不确定度。用 PDS Level 2 时，再加上公里级的整景偏移和全部地形视差（坑尺度上几十到数百米）。
5. **对我们的 patch（512 × 7.5 m ≈ 3.8 km）**：公里级整景偏移在数据集构建时已经粗对齐掉（CONTEXT.md）。剩下的主要是 patch 内随地形变化的距离向位移：坑尺度几十到上百米，即十几个像素量级；以及 CE-2 一侧百米级、空间尺度较大的周期误差。这一条与前次调研（`lunar-data-properties`）的推断一致，但仍是推断。

## F.4 未解决 / 要看我们自己的数据才能回答的问题

1. **我们用的是哪个 Mini-RF 产品？** 判断方法（按优先顺序）：
   - **文件名**：`LSZ_ooooo_1CD_XKU_...` 是 Level 1（距离-方位几何）；`LSZ_ooooo_2CD_EKU_...` 或 `..._2CD_OKU_...` 是 Level 2（E 为等距圆柱投影，O 为斜圆柱投影）；`2S1`、`2CP` 等是 Level 2 的衍生波段；`LSZ_xxxxx_3S1_P..`、`GLOBAL_*_128PPD_SIMP` 是镶嵌。L = LRO，S = S 波段，Z = zoom。（errata.txt、ODE 帮助页【官方文档】）
   - **`.lbl` 的 `DATA_SET_ID`**：`LRO-L-MRFLRO-4-CDR-V1.0`（Level 1）、`-5-CDR-MAP-V1.0`（Level 2）、`-5-CDR-MOSAIC-V1.0`（极区镶嵌）、`-5-GLOBAL-MOSAIC-V1.0`（全月镶嵌）。
   - **`IMAGE_MAP_PROJECTION`**：Level 2 有 `MAP_PROJECTION_TYPE`、`A_AXIS_RADIUS = 1737.4`、`MAP_RESOLUTION = 4096 <pix/deg>`、`MAP_SCALE ≈ 0.0074 km/pix`。Level 1 没有地图投影对象。
   - **像素间距线索**【推断】：我们的数据是 7.5 m 网格。Level 1 zoom 的像素间距正好 7.5 m（`SARPixelSpacing: 7.5`），Level 2 的地图像素是 4096 ppd ≈ 7.40 m。若我们的 SAR patch 是 7.5 m 且未见 7.40 m，可能来自 Level 1，或者是重采样过的 Level 2。要看数据集构建脚本才能确定。
   - **`.TXT` 参数文件**：`dem_used`、`gcp_used`、`dem {}` 瓦片范围，用于判断这一景的 DEM 瓦片是否覆盖了我们的 ROI。
   - **是否来自 Fassett 2024（JHU-APL）或 Xu 2025 的重处理版本**：要看数据来源记录。
2. **CE-2 一侧是哪个版本？** CE2DOM2014（Ren 2014）还是 CE2TMap2015（7 m DOM），以及对应 DEM 瓦片号。版本不同，绝对偏移可能不同。CE-2 DOM 自身的平面精度没有找到直接评估，目前只能借用 DEM 的评估结果（183 m / 354 m）。
3. **数据集的「粗对齐」用的是什么？** 是经纬度直接裁切（等于接受两边各自的绝对误差），还是做过整景配准？这决定 patch 对的残余整体偏移会落在什么分布上。
4. **我们的 ROI 是否在极区？** 极区有 Kirk 的 USGS 受控镶嵌（若能获得）和 LOLA 30 m DEM，中低纬只有 SLDEM 60 m / 237 m 级的正射条件。
5. **视向（东视 / 西视，`LOOK_DIRECTION`、`LRO:LOOK_AZIMUTH_ANGLE`）和入射角（`INCIDENCE_ANGLE`）是否进了数据集元数据？** 用 cot θ 估地形位移需要它们，「坑内亮暗方向」也由它们决定。
6. **文献层面仍不确定的地方**：
   - Mini-RF Level 2 参数文件里那块 DEM 瓦片是什么数据、是否全为零：只靠标签判断不了，需要问 PDS Geosciences 或 Mini-RF 团队。
   - Level 2 实际用 PA 还是 ME：数据集说明与 SIS 不一致。
   - Kirk 2013 所说的受控极区镶嵌是否已存档：未找到。
   - DFSAR 的定位精度、PRADAN 产品 SIS：需注册登录，未读到。
   - Wang 2025（遥感学报）的数据与数值：正文付费，未读到。
   - Fassett 2024 正射后的米级残差：网页抽取未见；原文图表可能有。
