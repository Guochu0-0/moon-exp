# 出处清单（lunar-orthorectification）

调研日期 2026-10-28。阅读程度：
- **全文**：读过原文相关段落，包括 arXiv、NTRS、Europe PMC 上的接受稿或预印本。
- **全文（经网页抽取）**：原文网页经抓取工具抽取后阅读，不能保证读到全部段落或公式。
- **摘要**：只读到摘要、检索摘录或页面要点。
- **题录**：只核了题名或出处。
- **官方文档**：PDS 数据集说明（.cat）、产品标签（.lbl/.txt）、SIS、机构网页，均为直接下载阅读。

## PDS / 官方文档（Mini-RF、Mini-SAR、地基雷达）

| 简称 | 内容 | 链接 | 阅读程度 |
|---|---|---|---|
| PDS Mini-RF 节点页 | LRO Mini-RF 数据集索引 | https://pds-geosciences.wustl.edu/missions/lro/mrf.htm | 官方文档 |
| `mrflro_4_cdr_v1_0_ds.cat` | Level 1 CDR 数据集说明 | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/catalog/mrflro_4_cdr_v1_0_ds.cat | 官方文档 |
| `mrflro_5_cdr_map_v1_0_ds.cat` | Level 2 CDR-MAP 数据集说明（"orthorectified"、PA） | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/catalog/mrflro_5_cdr_map_v1_0_ds.cat | 官方文档 |
| `mrflro_5_cdr_mosaic_v1_0_ds.cat` | Level 3 极区镶嵌数据集说明 | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/catalog/mrflro_5_cdr_mosaic_v1_0_ds.cat | 官方文档 |
| `mrflro_4_cdr_insar_v1_0_ds.cat` | InSAR 连续模式 Level 1 | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/catalog/mrflro_4_cdr_insar_v1_0_ds.cat | 官方文档 |
| `mrflro_5_globalmosaic_ds.cat` | 全月镶嵌数据集说明（ME） | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-global-mosaic-v1/lromrf_1001/catalog/mrflro_5_globalmosaic_ds.cat | 官方文档 |
| `mrf_inst.cat` | Mini-RF 仪器说明 | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/catalog/mrf_inst.cat | 官方文档 |
| `mrf_bistatic_ds.cat` | 双站 PDS3 v2 数据集说明 | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-2_3_5-bistatic-v2/lromrf_2xxx/catalog/mrf_bistatic_ds.cat | 官方文档 |
| Mini-RF SIS（MRF-4009） | *PDS Data Product Software Interface Specification*，2010-02-25 | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/document/dp_sis/mrflro_dp_sis.pdf | 官方文档 |
| errata.txt | Mini-RF 发布说明与勘误 | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/errata.txt | 官方文档 |
| Level 2 标签与参数文件 | `lsz_06700_2cd_eku_47n277_v1.lbl/.txt`、`lsz_06701_2cd_oku_88n108_v1.lbl/.txt`、`lsz_00300_2cd_oku_75s281_v1.lbl`；另抽查 lromrf_0001 的 00300_00399 与 lromrf_0003 的 04600_04699 目录下共 9 个 `.txt` | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0005/data/sar/06700_06799/level2/ 等 | 官方文档 |
| 镶嵌标签 | `lsz_06899_3s1_phu_90n000_e_v1.lbl`、`lsz_xxxxx_3s1_pfu_90n000_v1.lbl`、`global_cpr_128ppd_simp_0c.lbl` | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0005/data/sar/mosaics/ ；https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-global-mosaic-v1/lromrf_1001/data/128ppd/ | 官方文档 |
| 双站处理说明 | Dutton, Espiritu, O'Shea, Patterson 等，*LRO Mini-RF Bistatic Data Processing Description Document* Rev 1.0，2024-09-19 | https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_minirf_bistatic/document/minirf_bistatic_processing_description_v1.pdf | 官方文档 |
| Mini-SAR 数据集说明 | `mrffr_5_cdr_map_ds.cat`、`mrffr_5_cdr_mosaic_ds.cat`、`inst.cat` | https://pds-geosciences.wustl.edu/lunar/ch1-orb-l-mrffr-1-pdr-v1/ch1mrf_0xxx/catalog/ | 官方文档 |
| `slrm_ds.cat` | Arecibo/GBT 12.6 cm 数据集说明 | https://pds-geosciences.wustl.edu/Lunar/arcb_nrao-l-rtls_gbt-5-12.6cm-v1/slrm_90xx/catalog/slrm_ds.cat | 官方文档 |
| ODE Mini-RF MAPCDR 帮助页 | 产品命名 | https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/LRO/Mini-RF/MAPCDR.htm | 官方文档 |

## 月球 SAR 论文

| 简称 | 文献 | 刊 / 年 | DOI / URL | 阅读程度 |
|---|---|---|---|---|
| Kirk 2010 | Kirk R.L. et al., Radargrammetry with Chandrayaan-1 and LRO Mini-RF Images of the Moon | ISPRS TC IV & AutoCarto 2010, IAPRS XXXVIII-4 | https://www.isprs.org/proceedings/XXXVIII/part4/files/Kirk.pdf | 全文 |
| Kirk 2012 | Kirk R.L. et al., Progress in Radargrammetric Analysis of Mini-RF Lunar Images | LPSC 43, #2772 | https://www.lpi.usra.edu/meetings/lpsc2012/pdf/2772.pdf | 全文 |
| Kirk 2013 | Kirk R.L. et al., A Radargrammetric Control Network and Controlled Mini-RF Mosaics of the Moon's North Pole... at Last! | LPSC 44, #2920 | https://www.lpi.usra.edu/meetings/lpsc2013/pdf/2920.pdf | 全文 |
| Kirk 2016 | Kirk R.L. et al., A Semi-Rigorous Sensor Model for Precision Geometric Processing of Mini-RF Bistatic Radar Images of the Moon | ISPRS Archives XLI-B4, 425–432, 2016 | 10.5194/isprs-archives-XLI-B4-425-2016 | 全文 |
| Eke 2014 | Eke V.R. et al., Lunar polar craters – icy, rough or just sloping? | Icarus 241, 2014 | 10.1016/j.icarus.2014.06.021；arXiv:1312.4749 | 全文（arXiv 版） |
| Fassett 2024 | Fassett C.I. et al., Improved Orthorectification and Empirical Reduction of Topographic Effects in Monostatic Mini-RF S-band Observations of the Moon | Planetary Science Journal 5, 2024 | 10.3847/PSJ/ad0a61；数据 doi:10.48732/vzjt-1634 | 全文（经网页抽取） |
| Xu 2025 | Xu Z., Zhao F., Lu P., Gao Y., Meng T., Dang Y., Li M., Wang R., A Robust Digital Elevation Model-Based Registration Method for Mini-RF/Mini-SAR Images | Remote Sensing 17(4):613, 2025 | 10.3390/rs17040613 | 全文 |
| Xu 2024 | Automatic Registration of Mini-RF S-Band Level-1 Data | IGARSS 2024, 6142–6145 | 10.1109/IGARSS53475.2024.10642532 | 题录（前次调研记为摘要） |
| Bhiravarasu 2021 | Bhiravarasu S.S. et al., Chandrayaan-2 Dual-frequency Synthetic Aperture Radar (DFSAR): Performance Characterization and Initial Results | Planetary Science Journal 2:134, 2021 | 10.3847/PSJ/abfdbf；arXiv:2104.14259 | 全文（arXiv 版） |
| Putrevu 2020 | Putrevu D. et al., L- and S-band Polarimetric Synthetic Aperture Radar on Chandrayaan-2 mission | Current Science 118(2):226–233, 2020 | 10.18520/cs/v118/i2/226-233 | 全文 |
| Tarun 2022 | Tarun M. et al., A Software Tool to Process & Analyse Chandrayaan-2 Polarimetric DFSAR Data | LPSC 53, #2232 | https://www.hou.usra.edu/meetings/lpsc2022/pdf/2232.pdf | 全文 |
| ISRO 2025 | ISRO Offers Advanced Data Products for Deeper Understanding of the Lunar Polar Region（2025-11-08） | ISRO 网页 | https://www.isro.gov.in/ISRO_EN/Lunar_Polar_Region.html | 官方文档 |
| Campbell 2007 | Campbell B.A. et al., Focused 70-cm Wavelength Radar Mapping of the Moon | IEEE TGRS, 2007（PDS 文档目录中的投稿稿，卷期未核对） | https://pds-geosciences.wustl.edu/Lunar/arcb_nrao-l-rtls_gbt-4_5-70cm-v1/lrm_90xx/document/lunar_radar.pdf | 全文 |
| Campbell 2010 | Campbell B.A. et al., Earth-Based 12.6-cm Wavelength Radar Mapping of the Moon | Icarus, 2010（修订稿，卷期未核对） | https://pds-geosciences.wustl.edu/Lunar/arcb_nrao-l-rtls_gbt-5-12.6cm-v1/slrm_90xx/document/sband_lunar_radar.pdf | 全文 |
| Margot 1999 | Margot J.L. et al., Topography of the lunar poles from radar interferometry | Science 284:1658, 1999 | 10.1126/science.284.5420.1658 | 摘要 |
| Zisk 1974 | Zisk S.H. et al., High-resolution radar maps of the lunar surface at 3.8-cm wavelength | The Moon 10, 1974 | — | 题录（经 Campbell 2007 引用） |
| Porcello 1974 | Porcello L.J. et al., The Apollo lunar sounder radar system | Proc. IEEE 62, 1974 | — | 题录 |
| Ono 2009 | Ono T. et al., Lunar Radar Sounder Observations of Subsurface Layers Under the Nearside Maria of the Moon | Science 323, 2009 | — | 题录 |
| Wang 2024 NSR | Wang C. et al., Scientific objectives and payload configuration of the Chang'E-7 mission | National Science Review 11(2):nwad329, 2024 | https://academic.oup.com/nsr/article/11/2/nwad329/7503932 | 全文（经网页抽取，前 10 万字符） |
| Lu 2025 | Lu P.P., Gao Y., Meng T.Y., Zhao F., Dang Y.N., … Wang R., Wu Y.R.，合成孔径雷达月球探测研究进展 | 遥感学报 29(2):342–361, 2025 | 10.11834/jrs.20254003 | 题录（PDF 字体无法提取） |
| Chan 2025 | Chan K. et al., Enhancements to Mini-RF X/C-band data quality through cross-channel calibration and reprocessing strategies | EPSC-DPS 2025-875 | https://meetingorganizer.copernicus.org/EPSC-DPS2025/EPSC-DPS2025-875.html | 摘要 |
| 魏东旭 2026 | 魏东旭 等，月球轨道高分辨 SAR 特性分析与成像校正方法 | 电子技术应用 52(8):148–154, 2026 | 10.16157/j.issn.0258-7998.267760 | 摘要 |
| Marshall 2011 | Locating the LCROSS Impact Craters | arXiv:1103.1687 | https://arxiv.org/abs/1103.1687 | 摘要 |
| Fa & Eke 2018；Nypaver 2021 | 约 900 m 偏移的原始出处 | JGR Planets 123；— | 10.1029/2018JE005668 | 题录（数字经 Fassett 2024 转述） |
| KPLO 载荷 | KPLO 新闻资料包、KASI 页面 | — | https://shadowcam.im-ldi.com/KPLO_english_press_kit.pdf | 摘要（检索摘录） |
| 嫦娥七号发射状态 | 中国日报、IT 之家等 2026 年 3–5 月报道 | — | 检索 | 摘要 |

## 月球光学与 DEM

| 简称 | 文献 | 刊 / 年 | DOI / URL | 阅读程度 |
|---|---|---|---|---|
| Li 2010 | Li C.L. et al., The global image of the Moon obtained by the Chang'E-1: Data processing and lunar cartography | Sci China Earth Sci 53(8):1091–1102, 2010 | 10.1007/s11430-010-4016-x | 摘要 |
| Ye 2011 | Automatic seamless stitching method for CCD images of Chang'E-1 | J. Earth Science, 2011 | https://journal.hep.com.cn/jes/EN/1160322974387593935 | 摘要（检索摘录） |
| Ouyang 2010 | Chang'E-1 Lunar Mission: An Overview and Primary Science Results | Chin J Space Sci 30, 2010 | — | 摘要（检索摘录） |
| CE-1 图集 | Li C. 等，The Chang'E-1 Topographic Atlas of the Moon | Springer, 2016 | 978-3-662-48439-5 | 题录（精度数字来自书目描述） |
| Ren 2014 | Ren X. et al., A New lunar global topographic map products from Chang'E-2 Stereo Camera Image Data | EPSC 2014-344 | https://meetingorganizer.copernicus.org/EPSC2014/EPSC2014-344.pdf | 全文 |
| Li 2018 | 李春来 等，基于嫦娥二号立体影像的全月高精度地形重建 | 武汉大学学报（信息科学版）43(4):485–495 | 10.13203/j.whugis20170400 | 摘要 |
| Xin 2020 | Xin X., Liu B., Di K., Yue Z., Gou S., Geometric Quality Assessment of Chang'E-2 Global DEM Product | Remote Sensing 12(3):526, 2020 | 10.3390/rs12030526 | 全文 |
| Liu 2019 | Liu J. et al., Descent trajectory reconstruction and landing site positioning of Chang'E-4 on the lunar farside | Nat Commun 10:4229, 2019 | 10.1038/s41467-019-12278-3（PMC6760200） | 全文（Europe PMC） |
| CE-5 定位 | 嫦娥五号着陆定位（SSPMA 2021；Remote Sensing 2021） | — | 10.1360/SSPMA-2021-0069 | 摘要（检索摘录） |
| CE-6 定位 | 嫦娥六号着陆器高精度视觉定位 | 遥感学报, 2024 | 10.11834/jrs.20244229 | 摘要（检索摘录） |
| LROC CE-6 | First Look: Chang'e 6 | LROC 网页 | https://lroc.im-ldi.com/images/1374 | 摘要（检索摘录） |
| Speyerer 2012 | Speyerer E.J. et al., In-flight geometric calibration of the Lunar Reconnaissance Orbiter Camera | ISPRS Archives XXXIX-B4:511, 2012 | https://isprs-archives.copernicus.org/articles/XXXIX-B4/511/2012/ | 全文 |
| Speyerer 2016 | Speyerer E.J. et al., Geometric calibration of the Clementine UVVIS camera using images acquired by LRO | ISPRS Archives XLI-B4:497, 2016 | https://isprs-archives.copernicus.org/articles/XLI-B4/497/2016/ | 全文 |
| Speyerer 2014/2016 SSR | Pre-flight and On-orbit Geometric Calibration of the LROC | Space Sci Rev, 2016 | — | 题录（数字经 Speyerer 2016 ISPRS 转述） |
| Scholten 2012 | Scholten F. et al., GLD100: The near-global lunar 100 m raster DTM from LROC WAC stereo image data | JGR 117, 2012 | 10.1029/2011JE003926 | 摘要 |
| LROC GLD100 页 | WAC_GLD100 数据页 | LROC | https://data.lroc.im-ldi.com/lroc/view_rdr/WAC_GLD100 | 官方文档（检索摘录） |
| Henriksen 2017 | Henriksen M.R. et al., Extracting accurate and precise topography from LROC narrow angle camera stereo observations | Icarus 283:122–137, 2017 | 10.1016/j.icarus.2016.05.012 | 摘要 |
| Lee 2012 | Controlled Polar Mosaics of the Moon for LMMP by USGS | LPSC 43, #2507 | — | 题录 |
| Barker 2016 | Barker M.K. et al., A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera | Icarus 273:346–355, 2016 | https://dspace.mit.edu/handle/1721.1/118411 | 摘要 |
| Sato & Haruyama 2021 | Kaguya TC 报告幻灯片（LSIC，2021-03-02） | — | https://lsic.jhuapl.edu/uploadedDocs/presentations/763-1.07_Sato_TC_20210302_novoice.pdf | 全文（幻灯片） |
| Laura 2025 | Laura J.R. et al., Medium Resolution Lunar Topography: An Updated View ... from Stereophotogrammetrically Derived DTMs | ISPRS Archives XLVIII-M-6-2025:197 | https://isprs-archives.copernicus.org/articles/XLVIII-M-6-2025/197/2025/ | 摘要 |
| ODE TC 帮助 | KAGUYA TC Derived Data | ODE | https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/KAGUYA%20(SELENE)/TC/Derived_Data.htm | 官方文档（检索摘录） |
| USGS TC 镶嵌 | Moon SELENE Kaguya TC Global Orthomosaic 474m | USGS | https://astrogeology.usgs.gov/search/map/moon_selene_kaguya_tc_global_orthomosaic_474m | 官方文档（检索摘录） |
| Archinal 2006 | Archinal B.A. et al., The Unified Lunar Control Network 2005 | USGS OFR 2006-1367 | https://pubs.usgs.gov/of/2006/1367/ | 全文 |
| ISSDC TMC | Chandrayaan-1 TMC DEM 说明 | ISSDC CHBrowse | https://webapps.issdc.gov.in/CHBrowse/ | 官方文档（检索摘录） |
| Krishna 2012 | Comparison of DEMs from Terrain Mapping Camera images with LOLA | LPSC 43, #2026 | — | 题录 |
| ASP Chandrayaan-2 | Ames Stereo Pipeline 文档 §8.15 | — | https://stereopipeline.readthedocs.io/en/stable/examples/chandrayaan2.html | 官方文档 |
| OHRC 预印本 | Sub-metre lunar DEM generation and validation from OHRC | arXiv:2604.01032, 2026 | https://arxiv.org/abs/2604.01032 | 摘要（检索摘录） |
| Apollo Zone | Apollo Zone Digital Image Mosaic and DEM Released（2011 新闻稿） | SpaceRef / moonviews | https://moonviews.com/?p=351 | 摘要（检索摘录） |
| Edmundson 2012 | ISIS jigsaw 与 Apollo 量测相机 | ISPRS Annals, 2012 | — | 题录 |
| USGS Lunar Orbiter 镶嵌 | Moon Lunar Orbiter Digital Photographic Global Mosaic 59m（Becker 2008） | USGS | https://astrogeology.usgs.gov/search/map/moon_lunar_orbiter_digital_photographic_global_mosaic_59m | 官方文档（检索摘录） |
| Guo & Wu 2012 | Guo J., Wu B., Comparison of Lunar Topographic Models Derived from Multiple Sources Based on Least Squares Matching | ISPRS Archives XXXIX-B4:313, 2012 | https://isprs-archives.copernicus.org/articles/XXXIX-B4/313/2012/ | 全文 |
| Wu 2011 | Wu B. et al., Integration of Chang'E-1 imagery and laser altimeter data for precision lunar topographic modeling | IEEE TGRS, 2011 | https://research.polyu.edu.hk/en/publications/integration-of-change-1-imagery-and-laser-altimeter-data-for-prec/ | 摘要（检索摘录） |
| Wang 2025 | 王晨旭、彭嫚、谢彬、邸凯昌、芶盛，月球可见光和SAR影像特征匹配方法比较研究 | 遥感学报 29(2):389–401, 2025 | 10.11834/jrs.20243520 | 题录 + 页面要点（正文付费） |

## 几何基准与定轨

| 简称 | 文献 | 刊 / 年 | DOI / URL | 阅读程度 |
|---|---|---|---|---|
| LRO 白皮书 | LRO Project & LGCWG, A Standardized Lunar Coordinate System for the Lunar Reconnaissance Orbiter and Lunar Datasets, v4/5, 2008 | NASA | https://science.nasa.gov/wp-content/uploads/2024/01/luncoordwhitepaper-10-08.pdf | 全文 |
| NAIF FK | `moon_de440_250416.tf`、`moon_080317.tf` 注释 | NAIF | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/ | 官方文档 |
| NAIF 教程 | Special PCK and FK for Earth and Moon | NAIF | https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/23_lunar-earth_pck-fk.pdf | 官方文档（多为图，文字少） |
| USGS 数据标准 | PSDI Moon data standards | USGS | https://psdi.astrogeology.usgs.gov/moon/standards/data_standards/ | 摘要（检索摘录） |
| LCRS | Lunar References Systems ... ESA Moonlight；IAU 2024 决议 II | arXiv:2409.10043；arXiv:2510.15484 | — | 摘要（检索摘录） |
| Smith 2017 | Smith D.E. et al., Summary of the Results from the Lunar Orbiter Laser Altimeter after Seven Years in Lunar Orbit | Icarus 283:70–91, 2017 | 10.1016/j.icarus.2016.06.006（NTRS 20160008718） | 全文（接受稿） |
| Mazarico 2012 | Orbit determination of the Lunar Reconnaissance Orbiter | J. Geodesy 86:193–207, 2012 | 10.1007/s00190-011-0509-4 | 题录（经 Mazarico 2018 转述） |
| Mazarico 2018 | Mazarico E. et al., Orbit determination of the Lunar Reconnaissance Orbiter: Status after Seven Years | Planet. Space Sci. 162, 2018 | NTRS 20180007208 | 全文（预印本） |
| Kaguya 定轨 | IEICE 2009 技术报告；NASA PGDA 产品 74；JAXA DARTS 2019 | — | https://pgda.gsfc.nasa.gov/products/74 | 摘要（检索摘录） |
| 嫦娥定轨 | 鹊桥二号环月大椭圆轨道定轨精度分析（上海天文台）；深空探测学报综述 | — | 检索 | 摘要（二手综述句） |
| Chandrayaan-1 定轨 | IAC-06 C1.5 摘要 | — | https://iafastro.directory/iac/archive/browse/IAC-06/C1/5/4122/ | 摘要 |

## 地球对照

| 简称 | 文献 | DOI / URL | 阅读程度 |
|---|---|---|---|
| SentiWiki S1 产品 | ESA，Sentinel-1 Products | https://sentiwiki.copernicus.eu/web/s1-products | 官方文档 |
| Earthdata S1B GRD | NASA Earthdata Sentinel-1B GRD 条目 | https://www.earthdata.nasa.gov/data/catalog/asf-sentinel-1b-dp-grd-high-1 | 官方文档（二手转述 ESA 定义） |
| Small & Schubert 2022 | *Guide to Sentinel-1 Geocoding*, UZH-S1-GC-AD Issue 1.12 | https://sentinels.copernicus.eu/documents/247904/1653442/Guide-to-Sentinel-1-Geocoding.pdf | 全文 |
| Schubert 2017 | Sentinel-1A/B Combined Product Geolocation Accuracy, Remote Sensing 9(6):607 | 10.3390/rs9060607 | 题录 |
| Small 2011 | Small D., Flattening Gamma: Radiometric Terrain Correction for SAR Imagery, IEEE TGRS 49(8):3081–3093 | DOI 未核对 | 题录 |
| CARD4L NRB | CEOS-ARD Product Family Specification: Normalised Radar Backscatter（v4.1–v5.5） | https://ceos.org/ard/ | 摘要（PDF 下载失败，只读到检索摘录） |
| HyP3 RTC | ASF HyP3 RTC Product Guide | https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/ | 官方文档 |
| Copernicus DEM | Copernicus DEM Product Handbook, GEO.1988, Issue 5.0 | https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf | 全文（精度表） |

## 软件

| 简称 | 内容 | URL | 阅读程度 |
|---|---|---|---|
| ASP CSM Mini-RF | Ames Stereo Pipeline §8.12.4 "The USGS CSM SAR sensor for LRO Mini-RF" | https://stereopipeline.readthedocs.io/en/latest/examples/csm.html | 官方文档 |
| ISIS mrf2pds 等 | ISIS 应用文档 | https://isis.astrogeology.usgs.gov/ | 题录 |
| STEP 论坛 | How to Geocode and georeference Chandrayaan-2 DFSAR SLC L1.1 Data? | https://forum.step.esa.int/t/how-to-geocode-and-georeference-chandrayaan-2-dfsar-slc-l1-1-data/29507 | 摘要（用户帖，非官方） |
