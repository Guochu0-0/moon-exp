# D. 处理方法与工具：月球 SAR 正射的工具链早就有了，缺的是系统化的控制

> 阅读程度标注同 [01-sar.md](01-sar.md)。出处见 [sources.md](sources.md)。

## D.0 一句话结论

- 雷达摄影测量（用 SAR 影像做控制、平差、立体测图和正射）在行星上由 USGS 的 Kirk 等人建立。工具是 ISIS（开源）+ SOCET SET（商业），先用于 Magellan、Cassini，后用于 Mini-SAR/Mini-RF。
- 月球 SAR 的公里级误差主要来自时间记录和星历，不是地形。所以有效的做法分两步：先改正整体偏移（改起始时间、平移，或做带控制点的平差），再投到 LOLA/SLDEM 上去掉视差。
- 近年三类做法：Kirk 2013/2016 的严格控制 + 正射（≤30 m）；Fassett 2024 的「入射角合成 S1 + 互相关 + ISIS 正射」；Xu 2025 的「DEM 模拟 SAR + NCC 整幅平移」（约 110 m）。后两者都只估整幅平移，不处理条带内的局部形变。
- 地球上的同类标准：Sentinel-1 GRD 只投到椭球；地形校正（Range-Doppler 正射）+ 辐射地形校正（RTC，Small 2011）后才算 CEOS ARD；绝对定位米级。

## D.1 USGS 雷达摄影测量：从 Magellan、Cassini 到 Mini-RF

| 内容 | 出处 |
|---|---|
| Magellan、Cassini 的数据 "were produced only in map-projected form"，所以 USGS 在 SOCET SET 里写的传感器模型要先「撤销」地图投影，回到距离-多普勒坐标；没有为它们做 ISIS 传感器模型 | Kirk 2010【全文】 |
| Mini-RF 有 Level 1（未投影）数据，所以同时为 ISIS 和 SOCET SET 写了传感器模型。ISIS 流程：`mrf2isis` 导入 → `spiceinit`（可指定用于正射的 DTM）→ `fx` 算 S1/CPR、取对数 → `autoseed`/`pointreg` 自动连接点 → `qnet` 人工补点（尤其是对侧视向影像之间）→ 对 Kaguya 或 LOLA 量控制点 → `jigsaw` 平差 → `cam2map` 正射 → `automos` 镶嵌 | Kirk 2010【全文】 |
| SOCET SET 流程：`mrf4socet` 导出 → APM/IPM 量点 → MST 平差 → NGATE 自动生成 DTM → ITE 人工编辑 → 正射 → `ortho2isis3`/`dem2isis3` 回 ISIS | Kirk 2010【全文】 |
| 雷达立体特有的系统误差：在对侧视向像对里，小坑会显得「浮」在周围地面之上（亮暗坑壁的错配），人工也难分辨 | Kirk 2010【全文】 |
| 后续改进：SOCET SET 原先把稀疏轨迹点按折线插值，导致数公里误差，改为稠密样条点后消失；jigsaw 支持「固定先验轨道 + 任意阶多项式改正」；对侧视向自动匹配成功率低，靠人工点 | Kirk 2012、Kirk 2013【全文】 |
| 双站：重建每个像素的观测时间、距离、多普勒，用低阶多项式拟合（"semi-rigorous"），再在新轨迹和 DEM 上严格求交 | Kirk 2016【全文】 |
| 精度：LCROSS 区平差后连接点 RMS 从 150 m 降到 12 m（1.6 像素），最大误差从 900 m 降到 35 m；北极 85°–90° 受控镶嵌中东、西视之间 ≤30 m（未控制时 1–3 km）；时间改正后约 30 m | Kirk 2010、Kirk 2013、Kirk 2016【全文】 |
| 立体匹配误差：SAR 约 1 像素量级，光学约 0.2 像素（Kirk 2010 表 1 的参数假设） | Kirk 2010【全文】 |

## D.2 现有软件对 Mini-RF / DFSAR 的支持：Mini-RF 有，DFSAR 基本没有

| 软件 | Mini-RF | DFSAR | 出处 |
|---|---|---|---|
| ISIS | 有 `mrf2isis`、雷达传感器模型、jigsaw 平差（Kirk 实现）；ISIS 文档另有 `mrf2pds` | 未找到支持 | Kirk 2010、2013【全文】；ISIS 应用文档【题录】 |
| SOCET SET | USGS 写的 Mini-RF 传感器模型（商业软件，USGS 提供客座设施） | 未找到 | Kirk 2010【全文】 |
| NASA Ames Stereo Pipeline | 有实验性的 CSM SAR 模型：`mrf2isis` → `spiceinit shape=ellipsoid` → `isd_generate` → 立体。原话："ASP approximates the curved SAR imaging rays as straight lines"；"The results are plausible but likely not fully rigorous"；示例只演示立体，没有平差或正射到 LOLA | ASP 文档 §8.12.4【官方文档】 |
| ESA SNAP | 用户论坛报告 DFSAR SLC L1.1 无坐标属性、参考系为 GCS_Moon_Spheroid，SNAP 不支持其定标 | ESA STEP 论坛（用户帖）【摘要，非官方】 |
| ISRO MIDAS | 可做 DFSAR 极化分析；地理编码和用 LOLA DEM 正射的模块（2022 年时）"under development" | Tarun 2022 LPSC【全文】 |

## D.3 近年 Mini-RF 重处理：都先改整体偏移，再投到 LOLA/SLDEM

| 工作 | 输入 | 方法 | DEM | 报告的精度 | 产品 | 出处 |
|---|---|---|---|---|---|---|
| **Kirk 2013** 北极受控镶嵌 | Level 1，1140 景（70°–90°N）先做 85°–90°N | `coreg` 自动约 6000 点（同视向），人工 25 点 + 从 LOLA 控制 NAC 镶嵌转来 17 个控制点，jigsaw 平差，东、西视分开拼 | LOLA | 东西视之间 ≤30 m（未控制 1–3 km）；平差 RMS 因尺度问题当时未报 | 计划存 PDS；**未找到**已发布记录 | Kirk 2013【全文】 |
| **Kirk 2016** 双站 Cabeus | 双站 RDR | 半严格传感器模型 + 起始时间改正 | LOLA | 改正后约 30 m（<100 m 像素） | 计划把全部双站数据重新地理参考，**未找到**完成记录 | Kirk 2016【全文】 |
| **Fassett 2024** | 全部 6644 个 S 波段 Level 1 条带 | 由局部入射角预测合成 S1（函数参照 Thompson 2011，用南极数据拟合），与实测 S1 互相关，求行（方位）、列（距离）偏移；再用 ISIS `cam2map` 正射到形状模型，30 m/像素 | 60°S–60°N 用 SLDEM（237 m/像素）；60°–75° 用 LOLA 100 m；75° 以上用 LOLA 30 m | 只说配准后 i 与 S1 的 Spearman 相关中位数 −0.42，近似 "good enough for successful coregistration"；未见以米计的残差。另有一句："treats Mini-RF as a camera and does not distinguish between the radar illumination in the azimuth and range directions"（上下文未完全读到）| 图 6、7 的全月与极区镶嵌放在 JHU-APL 数据仓库（doi:10.48732/vzjt-1634），另有 90 m 版本可向作者索取 | Fassett 2024【全文，经网页抽取】；JHU-APL 页【摘要，经检索】 |
| **Xu 2024**（IGARSS） | S 波段 Level 1 | DEM 模拟 SAR 自动配准（Xu 2025 的前身） | — | — | — | Xu 2024【题录；前次调研记为摘要】 |
| **Xu 2025**（Remote Sensing） | 6584 景 Mini-RF S zoom Level 1、735 景 Mini-SAR | 用 Thompson 13 cm 散射模型由局部入射角模拟 OC 图像，去背景（图像金字塔低通后相减，相关峰提升约 66%）后做 NCC；每景分 5 段（Mini-SAR 3 段）各求偏移，>10 km 的视为无效，取中位数作整景平移，用 ISIS `translate` 改正，再生成 Level 2 | LOLA 60 m（>60°）+ SLDEM 60 m 拼接的晕渲图；不用 WAC/NAC，因为 PSR 内光学信息不足 | 南极 >75°S 人工标 240 个「标准坑」（3 个误配剔除）：平均 185.38 m（约 6 像素）；>10 km 大坑 232.86 m；<10 km 小坑 111.86 m（约 4 像素），作者取 ~110 m 为精度。摘要："reduced from approximately 3000 m to about 100 m" | 全月配准数据集（未找到公开下载地址） | Xu 2025【全文】 |
| **LRO QuickMap** | — | Xu 2025 称 QuickMap 提供了两极的已配准 Mini-RF 镶嵌，中低纬没有 | — | — | 在线地图 | Xu 2025【全文，二手】 |
| **Bhiravarasu 2021** 中的 Mini-RF | Level 1 | 照 Fa & Eke 2018、Virkki & Bhiravarasu 2019 的做法用 ISIS3 正射到 15 m | — | — | — | Bhiravarasu 2021【全文】 |

**有没有机构发布过受控并正射的 Mini-RF / DFSAR 逐条带或全月镶嵌？**
- PDS：2012 年 Level 3 **极区**镶嵌按 Kirk 2013 的说法是「未控制但正射」。2014 年全月镶嵌由 Level 2 拼成，未控制、未正射。
- 未找到 Kirk 2013 所说的受控极区镶嵌已存入 PDS 的记录。
- Fassett 2024 的镶嵌在 JHU-APL 仓库（不在 PDS）。
- DFSAR：ISRO 2025 年发布了 L 波段极区 Level 3C 镶嵌，处理说明未读到。

## D.4 地球对照：Sentinel-1 GRD 只到椭球，地形校正和 RTC 之后才是 ARD

| 内容 | 出处 |
|---|---|
| Sentinel-1 Level-1 GRD："projected to ground range using the Earth ellipsoid model WGS84"，"The terrain height used varies in azimuth and it is constant in range" | NASA Earthdata 的 Sentinel-1B GRD 条目（转述 ESA 定义）【官方文档，二手】 |
| ESA SentiWiki：Level-1 GRD 在地面距离坐标；NRT 绝对定位精度 SM GRD 2.5 m、IW GRD 7 m，EW 未规定 | SentiWiki S1 产品页【官方文档】 |
| 椭球地理编码（GEC，整景只用一个高度）与地形地理编码（GTC，逐像素用 DEM 高度）的区别；"Vertical errors in hypsometry ... induce horizontal errors in planimetry"，DEM 高估使像素取自过近的近距端，低估则过远 | Small & Schubert，*Guide to Sentinel-1 Geocoding*（UZH-S1-GC-AD，1.12 版，2022）§4.7.9【全文】 |
| 「开箱即用」的绝对定位误差（ALE）：方位约 −3 m、距离约 2 m 量级的偏差（瑞士角反射器），IW 要求 7 m；做大气延迟（距离向约 3 m）和时间偏差改正后到亚米 | 同上 表 12、§4.7.11【全文】 |
| RTC：Small 2011 提出用模拟的照射面积归一化得到地形平坦化的 γ0（"Flattening Gamma"） | Small 2011（IEEE TGRS 49(8)）【题录】 |
| CEOS ARD 的 NRB（归一化雷达后向散射）：要求做 RTC，给 γ0；要求给出单一数值的几何精度估计；后续版本要求以偏差 + 标准差报告绝对定位误差，并写明所用大地水准面 | CARD4L NRB 规范 v4–v5.5 检索摘录【摘要】 |
| ASF HyP3 RTC：用 GAMMA 软件，DEM 为 Copernicus GLO-30（2022 版，转到 EGM2008）；在斜距几何中用查找表做地形校正，只重采样一次；γ0 默认；DEM 匹配失败阈值为残差 2 像素、最大偏移 50 m；页面未给定位精度数字，也未提 CARD4L | ASF HyP3 RTC 产品指南【官方文档】 |
| Copernicus DEM GLO-30：绝对垂直精度 <4 m（90% 线性误差），绝对水平精度 <6 m（90% 圆误差） | Copernicus DEM 产品手册 I5.0【全文】 |

【推断】用地球的说法类比：PDS 的 Mini-RF Level 2 相当于 Sentinel-1 的「GEC / 椭球投影产品」，只是参考面换成球、几何定位还差两到三个量级。Fassett 2024 和 Xu 2025 相当于「先改定位偏差、再做 GTC」，但都没有做 RTC 那样的辐射地形改正。Fassett 2024 做了经验的入射角归一化，比较接近 RTC。
