# 月球光学与 SAR 数据的正射与几何精度现状（文献调研）

> 调研日期：2026-10-28。只读文献、官方文档和 PDS 标签，没有在我们的数据上做实测。
> 阅读程度：**【全文】** 读过原文相关段落；**【全文，经网页抽取】** 经抓取工具抽取后阅读；**【摘要】** 只读到摘要或检索摘录；**【题录】** 只核了题名；**【官方文档】** PDS 数据集说明、标签、SIS、机构网页，均为直接下载阅读；**【推断】** 我自己的换算或判断，没有文献直接这么说。
> 前次调研 `origin/research/lunar-data-properties` 只作线索，本文的结论都重新核对过。
> 出处全表见 [sources.md](sources.md)。

## 要点

1. **PDS 的 Mini-RF Level 2 写的 "orthorectified"，实际是用 Vexcel 的 "Ortho" 处理器投到 1737.4 km 球面，没有控制。** 数据集说明和 SIS 的原话是 "they are orthorectified and are re-sampled into oblique cylindrical map projections"。但标签地图投影半径三轴都是 1737.4 km，Vexcel 参数文件写 `terrain_height: 0`、`gcp_used: 0`。所附 DEM 瓦片来源不明，格网约 600 × 200 m，常盖不全条带。团队文献也说是投到球面：Kirk 2010 "uncontrolled, and are (so far) projected onto the lunar reference sphere"；Fassett 2024 "reduced onto a lunar sphere ... not fully orthorectified or correctly selenolocated"。【官方文档；全文】 判断见 [01-sar.md](01-sar.md) A.2.3【推断】。
2. **Mini-RF 存档产品的整体位置误差典型约 0.9 km，最大约 3 km，主因是图像起始时间记录误差。** 时间误差在 0 到 1.0–1.3 s 之间线性漂移后归零，改正后东、西视图之间的错位从约 3 km 降到约 30 m（Kirk 2016【全文】）；Fassett 2024 约 0.9 km【全文，经网页抽取】；Xu 2025 0–3000 m 随轨道号漂移【全文】。
3. **做过控制并正射的 Mini-RF 产品不在 PDS 的逐条带产品里。** 现有三类：USGS 北极受控镶嵌，东西视之间 ≤30 m（Kirk 2013【全文】，未找到存档记录）；Fassett 2024 正射到 SLDEM/LOLA 的镶嵌（JHU-APL 仓库）；Xu 2025 的全月配准数据集，约 110 m（小坑）至 185 m（平均）【全文】。
4. **Chandrayaan-2 DFSAR 的定位逐像素用了 LOLA DEM（约 118 m），但没有公开的定位精度。** 产品分 L1A "seleno-tagged SLC"、L1B "seleno-tagged ground-range"、L2 "seleno-referenced"（Bhiravarasu 2021【全文】）；入射角 9.6°–36.9°、分辨率 2–75 m（Putrevu 2020【全文】）。ISRO 2025 年发布的 L 波段极区 Level 3C 镶嵌没有写 DEM 与精度【官方文档】。
5. **新版 Mini-RF 双站产品（PDS4，2024）按 LOLA DEM 成像，但没有控制。** 用 SPICE 星历和由 LDEM256 重采样到 16 ppd 的 DEM，在 MOON_ME 系里做时域后向投影，输出 100 m 像素（双站处理说明【官方文档】）。16 ppd 约 1.9 km【推断】。
6. **地基雷达图（Arecibo/GBT 70 cm、12.6 cm）按参考球体或平均半径成像。** 只有 70 cm 的 Level 2 "geometrically corrected to match the Clementine 750-nm optical image base"（Campbell 2007【全文】）。
7. **光学主产品都做了正射。LRO 系天然在 LOLA 框架里，绝对精度 10–45 m。** NAC 约 20 m（Speyerer 2012【全文】）；WAC 约 40–45 m（Speyerer 2016 引 2014【全文，二手】）；NAC DTM 约 10 m 水平、1 m 垂直（Henriksen 2017【摘要】）。
8. **嫦娥二号 7 m DOM / 20 m DEM 用自己的 DEM 正射，只以 5 个激光反射器为绝对控制。** 相对 SLDEM2015 平面平均 183 m，背面和南半球约为正面和北半球的 2 倍，CE-3/CE-4 区有沿轨、跨轨周期误差（Xin 2020【全文】）；相对 LRO 地形全局平均 354 m（σ 228 m）（Liu 2019【全文】）。
9. **当前事实上的几何基准是 LOLA 定义的 ME 系，精度 <10 m。** LRO 定轨在 GRAIL 重力场之后 <10 m（Mazarico 2018【全文】；Smith 2017【全文】）。ME 与 PA 两系相差约 875 m（NAIF DE440 FK【官方文档】）。Mini-RF Level 2 数据集说明写 PA、SIS 写 ME，前后不一【官方文档】。
10. **月球光学-SAR 配准只有整幅平移级别的结果。** 唯一的直接比较工作是 Wang 2025（遥感学报，特征匹配比较，HAPCG 最好；正文未读到）【题录 + 页面要点】。其余都是把 SAR 配到 DEM 或 DEM 模拟图，即 Xu 2025、Fassett 2024、Kirk 系列。没找到像素级、显式建模地形相关局部形变的光学-SAR 配准。
11. **即使用 LOLA/SLDEM 正射，SAR 侧仍剩整景 30–230 m 的平移残差，外加随坡度和亚 DEM 地形变化的几十米非刚性位移。** 平移残差见 Kirk 2013/2016、Xu 2025【全文】。非刚性位移是自算：直径 500 m 的坑在 237 m 格网 DEM 下约残留 87 m，约 12 个 7.5 m 像素【推断】。见 [06-implications.md](06-implications.md) F.3。
12. **量级对比：月球 SAR（存档）比月球光学（LRO）差一到两个量级，比地球 Sentinel-1（米级）差两到三个量级。** Sentinel-1 IW GRD 要求 7 m，开箱 2–3 m（Small & Schubert 2022【全文】）；GRD 本身也只投到椭球，类似 Mini-RF Level 2 投到球面。

## 汇总表（详表见 [06-implications.md](06-implications.md) F.1）

| 数据源 | 正射？ | DEM | 控制？ | 定位精度（相对 LOLA/LRO） | 主要出处 |
|---|---|---|---|---|---|
| Mini-SAR Level 2 / 镶嵌 | 名义正射，实为球面 | 未说明 | 否 | 约 3 km（Xu）；镶嵌至 5 km（Eke） | `mrffr_5_cdr_map_ds.cat`；Eke 2014；Xu 2025 |
| Mini-RF Level 1 | 否 | — | 否 | 不带地理坐标 | `mrflro_4_cdr_v1_0_ds.cat`；Kirk 2010 |
| Mini-RF Level 2（PDS） | 名义正射，实为 1737.4 km 球 | 来源不明粗瓦片 | 否 | 约 0.9 km，至 3 km | 标签；Kirk 2010；Fassett 2024；Xu 2025；Kirk 2016 |
| Mini-RF Level 3 极区镶嵌（2012） | 正射（Kirk 2013）/ 否（Eke 2014） | LOLA | 否 | 1–3 km（东西视之间） | 标签；Kirk 2013；Eke 2014 |
| Mini-RF 全月镶嵌（2014） | 否（由 Level 2 拼） | — | 否 | 同 Level 2 | `mrflro_5_globalmosaic_ds.cat` |
| Mini-RF 双站（PDS4） | 成像时投到 DEM | LOLA 16 ppd | 否 | 未报 | 双站处理说明 |
| Mini-RF USGS 受控（Kirk） | 是 | LOLA / Kaguya | 是 | ≤30 m | Kirk 2010/2013/2016 |
| Mini-RF Fassett 2024 | 是 | SLDEM / LOLA（237 / 100 / 30 m） | 是（整景） | 未报米级 | Fassett 2024 |
| Mini-RF Xu 2025 | 是（ISIS） | LOLA + SLDEM 60 m | 是（整景） | 约 110–185 m | Xu 2025 |
| DFSAR L1/L2 | 逐像素定位用 DEM | LOLA 约 118 m | 未找到 | 未找到 | Bhiravarasu 2021 |
| Arecibo 70 cm Level 2 | 否 | — | 校正到 Clementine 底图 | 继承 Clementine | Campbell 2007 |
| Arecibo 12.6 cm | 否（参考球） | — | 未提 | 未报 | Campbell 2010；`slrm_ds.cat` |
| CE-1 CCD 全月图 | 是 | CE-1 DEM | 未读到 | 100–1500 m | Li 2010 |
| CE-2 DOM 7 m / DEM 20 m | 是 | CE-2 DEM | 5 个 LRRR | 对 SLDEM 183 m；对 LRO 354 m | Ren 2014；Xin 2020；Liu 2019 |
| LROC NAC | 是 | GLD100 / NAC DTM | 定轨 + LRRR 标定 | 约 20 m | Speyerer 2012 |
| LROC WAC 镶嵌 | 是 | GLD100 | 无控制点 | 约 40–45 m | Speyerer 2016 |
| NAC DTM | — | — | 对 LOLA | 约 10 m / 1 m | Henriksen 2017 |
| Kaguya TC | 是 | TC DTM | 否（SLDEM2015 时配到 LOLA） | 对 LRRR 12–22 m | Sato & Haruyama 2021；Barker 2016 |
| Clementine / ULCN 2005 | 底图否 | ULCN 点 | 控制网 | 100 m 至几公里 | Archinal 2006 |
| Chandrayaan-2 TMC-2 / OHRC | ISRO 提供正射与 DEM | TMC DTM | 未找到 | OHRC 实例沿轨约 4 km（对齐前） | ASP 文档 |
| Apollo 量测相机 / Lunar Orbiter | 是 | 自身立体 / ULCN 2005 | 平差 / ULCN 2005 | DEM 对 LOLA 40 m / 继承 ULCN | SpaceRef 2011；USGS |
| *地球 Sentinel-1 GRD* | 否（椭球） | — | — | 7 m 要求；开箱 2–3 m | SentiWiki；Small & Schubert 2022 |

## 索引

| 文件 | 内容 |
|---|---|
| [01-sar.md](01-sar.md) | A. 月球 SAR / 雷达数据全清单：Mini-SAR、Mini-RF 单站（含 "orthorectified" 矛盾的标签核对）、双站、DFSAR、地基雷达、测深雷达与规划中的 SAR |
| [02-optical.md](02-optical.md) | B. 月球光学数据的正射现状：嫦娥 1/2/着陆区、LROC、Kaguya、Clementine/ULCN、印度相机、Apollo 与 Lunar Orbiter |
| [03-geodesy.md](03-geodesy.md) | C. 几何基准：ME/PA 与 DE421/DE440、控制网、定轨精度、月球 DEM 一览 |
| [04-methods-tools.md](04-methods-tools.md) | D. 处理方法与工具：USGS 雷达摄影测量、ISIS/SOCET/ASP 支持度、近年 Mini-RF 重处理、地球 SAR 对照 |
| [05-registration.md](05-registration.md) | E. 多源配准：光学-SAR、光学-光学、光学-LOLA 及残差 |
| [06-implications.md](06-implications.md) | F. 总表、三方精度对比、正射后 SAR 侧剩余误差、未解决问题与如何从文件标签判断产品 |
| [sources.md](sources.md) | 出处清单与阅读程度 |

## 仍不确定的地方（详见 [06-implications.md](06-implications.md) F.4）

- Mini-RF Level 2 参数文件里那块 DEM 瓦片是什么数据：标签没写名称，只能确认格网和范围。
- Mini-RF Level 2 的坐标系是 PA 还是 ME：数据集说明与 SIS 不一致。
- Kirk 2013 的受控极区镶嵌是否已存档：未找到。2012 年 PDS 极区镶嵌是否算「正射」：Kirk 2013 与 Eke 2014 说法相反。
- DFSAR 的定位精度、PRADAN 产品 SIS：未读到；Chandrayaan-2 定轨精度：未找到。
- Wang 2025（遥感学报）的数据与数值：正文付费，未读到。Fassett 2024 正射后的米级残差：网页抽取未见。
- CE-2 **DOM** 的平面精度没有直接评估，现有数字都是对 DEM 的。
- 我们的数据集用的是哪一级 Mini-RF 产品、哪一版 CE-2 DOM、粗对齐怎么做的：需要看数据集构建记录。判断方法见 F.4。
