# C. 几何基准：当前事实上的基准是 LOLA 定义的 ME 框架，精度约 10 m

> 阅读程度标注同 [01-sar.md](01-sar.md)。出处见 [sources.md](sources.md)。

## C.0 一句话结论

- PDS 存档统一用 **ME 系**（平地球/极轴系），由 JPL 星历（DE421，后续 DE440）实现。它和动力学用的 **PA 系**（主轴系）在月面相差约 0.9–1 km。
- 现行基准实际由 **LOLA** 定义："The LOLA-defined shape and reference frame enables accurate (<10m) geolocation globally"（Smith 2017）。LRO 定轨在 GRAIL 重力场之后好于 10 m。
- 其他任务的定轨精度：Kaguya 主任务约 10 m，嫦娥一号几百米，嫦娥三号约 20 m，Chandrayaan-1 约 500 m，Chandrayaan-2 未找到。差几个量级，所以各任务影像之间的绝对偏移大小不一。

## C.1 参考框架：ME 与 PA 相差约 875 m，标签里写错了系就是近 1 km 的偏移

| 内容 | 出处 |
|---|---|
| **ME 系**：+Z 是平均自转轴，本初子午线含平均地球方向，IAU/IAG 工作组报告用此系；**PA 系**：轴取月球主惯量轴 | NAIF `moon_de440_250416.tf` 注释【官方文档】 |
| 「The axes of the two systems differ by about 1 km at the lunar surface」；LRO 各仪器 "shall deliver data to the PDS with planetocentric coordinates in the ME reference system only"，LGCWG 建议其他数据也如此；选定 DE421，并给出 PA→ME 旋转 `M = Rx(-0.30") Ry(-78.56") Rz(-67.92") P`（仅适用于 DE421） | LRO Project / LGCWG 白皮书（2008，v4）【全文】 |
| DE440 下 MOON_ME 与 MOON_PA 夹角约 0.02886°，"equivalent to approximately 875 m" | NAIF FK 注释【官方文档】 |
| DE440 的 ME 实现（MOON_ME_DE440_ME421）与 DE421 的 ME 之间：2000–2040 年最大约 53 cm | NAIF FK 注释【官方文档】 |
| IAU 报告公式实现的 ME（SPICE 里叫 IAU_MOON）与 DE 实现的 ME 之间：振幅约 155 m、平均约 72 m（DE440，2000–2040；DE421 时为 76 m）。原因是 IAU 公式截断了天平动级数 | NAIF FK 注释（DE440 与 080317 两版）【官方文档】 |
| USGS 数据标准页：ME DE421 是现行通用框架，预计将改用 DE440 | USGS PSDI 数据标准页【摘要，经检索】 |
| 2024 年 IAU 决议 II 建立月球天球参考系（LCRS），属于惯性系层面 | arXiv 2510.15484【摘要，经检索】 |

【推断】对配准的意义有两条。
- 用 IAU_MOON 与用 DE 系 ME 处理的数据之间，最多会差约 155 m。
- 标签写 PA、实际按 ME 处理（或反过来）的数据会差约 875 m。Mini-RF Level 2 的数据集说明写 PA，SIS 写 ME（见 [01-sar.md](01-sar.md) A.2.3 第 5 条）。这只是**标签措辞**上的不一致，没有证据表明实际产品用了 PA。Kirk 2016 把 Mini-RF 的公里级误差归因于时间记录，不是框架。

## C.2 控制网：ULCN 2005 已被 LOLA 取代

| 内容 | 出处 |
|---|---|
| ULCN 2005：27 万点，水平约 100 m 至几公里，垂直数百米（详见 [02-optical.md](02-optical.md) B.6） | Archinal 2006【全文】 |
| LOLA："define the 'LOLA geodetic frame' to which the LRO datasets are now controlled"；"enables accurate (<10m) geolocation globally" | Smith 2017（Icarus，NTRS 接受稿）【全文】 |
| 激光反射器（Apollo 11/14/15、Lunokhod 1/2）的位置已知到亚米，被用来标定 NAC 指向，也是 CE-2 的 5 个绝对控制点 | Speyerer 2012【全文】；Ren 2014【全文】 |
| LRO 地形数据相对 5 个反射器偏差 <5.2 m；LRO 地形数据不确定度约 20 m | Liu 2019【全文，二手引 LRO 文献】 |

## C.3 定轨精度：LRO 从约 20 m 提高到 <10 m，其他任务在 10 m 到几百米之间

| 航天器 | 精度 | 出处 |
|---|---|---|
| LRO（任务要求） | 50 m 位置知识要求 | Mazarico 2018【全文】 |
| LRO（Mazarico 2012 阶段） | 无线电 + LOLA 交叉点满足 50 m；LRO 专用重力场使纯无线电定轨达到 20 m 级 | Mazarico 2018 摘要对早期工作的描述【全文】；Mazarico 2012【题录】 |
| LRO（GRAIL 之后） | "further improved the radio-only orbit reconstruction quality (<10 meters)"；重叠弧段 RMS：径向约 50 cm，总位置 <10 m；60 个人造地标检验约 9–10 m | Mazarico 2018（PSS，NTRS 预印本）【全文】 |
| LRO（2010 年 Mini-RF 处理时） | SPICE 星历 "on the order of 100 m" | Kirk 2010【全文】 |
| Kaguya | 主任务 Doppler + 测距几十米，加同波束 VLBI 约 10 m；延伸任务原为几公里，用 LALT 对 LOLA 改到几十米 | IEICE 2009 摘要、NASA PGDA、JAXA DARTS【摘要】 |
| 嫦娥一号 / 三号 | 几百米量级 → 20 m（嫦娥三号环月段） | 鹊桥二号定轨论文综述句、深空探测学报综述【摘要，二手】 |
| 嫦娥二号 | 绕月段的米级数字**未找到** | — |
| Chandrayaan-1 | 测图阶段约 500 m | IAC-06 摘要【摘要】 |
| Chandrayaan-2 | **未找到** | — |

## C.4 月球 DEM 一览

| DEM | 分辨率 | 垂直精度 | 平面精度 / 基准 | 出处 |
|---|---|---|---|---|
| LOLA 格网（LDEM） | 75°–90° 可到 30 m/像素（Xu 2025）；全月常用约 118 m 格网（DFSAR、Eke、双站处理都用到）；低纬两条轨迹之间有公里级空隙（Kirk 2012） | LOLA 剖面不确定度约 1 m 垂直、约 10 m 水平（Henriksen 2017 摘要） | 定义基准，<10 m（Smith 2017） | Smith 2017【全文】；Xu 2025【全文】；Kirk 2012【全文】；Henriksen 2017【摘要】 |
| SLDEM2013 | TC + LOLA 填补极区空隙 | — | — | Sato & Haruyama 2021【全文】 |
| SLDEM2015 | ±60°，512 ppd（赤道约 60 m） | 约 3–4 m | 每块 TC DEM 已配到 LOLA；Mazarico 2018 称其 "tied to LOLA at submeter-level" | Barker 2016【摘要】；Mazarico 2018【全文】 |
| GLD100 | 100 m，79°S–79°N | 约 10–20 m（LROC 网页） | LRO 框架 | Scholten 2012【摘要】；LROC【官方文档】 |
| Kaguya TC DTM | 约 10 m 影像，DTM 1° 瓦片 | 自身约 5 m，对 LRRR 4.7 m | 对 LRRR 12–22 m | Sato & Haruyama 2021【全文】 |
| TC 重处理（USGS/Ames） | 13 万多块 | 2–4 m | 水平在约 30 m 正射像素内 | Laura 2025【摘要】 |
| 嫦娥一号 DEM | 500 m | 120 m | 192 m | 图集书目描述【题录 / 二手】 |
| 嫦娥二号 CE2TMap2015 | 20 m（另有 7 m、50 m） | 对 SLDEM 均值 2.3 m（σ 15.4 m） | 对 SLDEM 183 m；对 LRO 354 m；对 LRRR 21–97 m；有周期性误差 | Xin 2020【全文】；Liu 2019【全文】；Ren 2014【全文】 |
| NAC DTM | 2–5 m | 绝对约 1 m（与 LOLA 相当） | 绝对约 10 m | Henriksen 2017【摘要】 |
| Apollo Zone DEM | 约 30 m | 对 LOLA 平均 40 m（σ 37 m） | — | SpaceRef 2011【摘要】 |
| ULCN 2005 DEM | 16 ppd（约 1.9 km） | 数百米 | 100 m 至几公里 | Archinal 2006【全文】 |
| Mini-RF 雷达立体 DTM（Kirk） | Cabeus 局部；Jackson 25 m 格网 | Cabeus：精度不差于几十米、精密度约 10 m | 控制后像素级 | Kirk 2010【全文】 |
| Mini-RF InSAR / Sandia 立体 DEM | SIS 规划过 | — | — | SIS【官方文档】；**未找到**已发布产品 |
| DFSAR 干涉 / 立体 DEM | — | — | — | **未找到** |
| Goldstone 雷达干涉（两极） | 150 m | 约 50 m | — | Margot 1999【摘要】 |
