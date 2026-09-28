# 源码清单（eval-metrics 调研合并版）

本表合并了 01–05 分文件的源码清单，并做了去重。
- 源码克隆在仓库外的 `D:\Code\refs\`，**不进 git**。
- 本表的 commit 用于复现：`git clone <url> && git checkout <commit>`。分文件里所有 permalink 都锁定在这些 commit 上。
- 核对方式：2026-09-23 逐个运行 `git -C D:\Code\refs\<dir> rev-parse HEAD`，结果与分文件记录的 commit **全部一致**。
- 与磁盘的对应：`ls D:\Code\refs` 共 42 个目录，全部出现在下表中；分文件里没有登记的多余目录为 0 个。与磁盘不一致或需要注意的地方，在「备注」列和文末「与磁盘不一致 / 需注意」一节里写明。

| 工作 | repo URL | commit | 本地路径 | 评测入口文件 | 引用分文件 |
|---|---|---|---|---|---|
| MultiResSAR 综述数据集 | https://github.com/betterlll/Multi-Resolution-SAR-dataset- | 6f664903bc9662a66ac7c2c88a39f5e2d9da1ec4 | D:\Code\refs\Multi-Resolution-SAR-dataset- | 无（只有 README） | 01 |
| SRIF（含 8 种方法的对比脚本和 GT） | https://github.com/LJY-RS/SRIF | 88881a3a8789d0bed6a8df91b943e02f2593a0cd | D:\Code\refs\SRIF | `demo_SRIF.m`、`demo_RIFT.m` 等 `demo_*.m`；GT 在 `dataset/<类别>/gt_N.txt` | 01, 04 |
| RIFT | https://github.com/LJY-RS/RIFT-multimodal-image-matching | 7ea830e2f13cc3c226f975fe9e98b7666a8f26fb | D:\Code\refs\RIFT-multimodal-image-matching | `RIFT_demo.m`、`FSC.m`（无 GT 评测） | 01, 02, 03, 04 |
| RIFT2 | https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation | 0e980ce4124d2abd62727dcf040198db0a18b269 | D:\Code\refs\RIFT2-multimodal-matching-rotation | 未审评测（只作为对比方法登记） | 03 |
| LNIFT（官方 exe） | https://github.com/LJY-RS/LNIFT_exe | 6349a8c47b4fc3c58c28c8aa526096831de06bdd | D:\Code\refs\LNIFT_exe | 可执行文件发布，无评测 | 03 |
| LNIFT（第三方复现） | https://github.com/arunsahu159/LNIFT-Locally-Normalized-Image-for-Rotation-Invariant-Multimodal-Feature-Matching | 1ffa3528cd88a655d9fb7e99dafb1a27ebe8d1ca | D:\Code\refs\LNIFT | 无（非官方 notebook） | 01, 03 |
| OS-Eval | https://github.com/xym2009/OS-Eval | 4df7d5a780d4c2d5b913ac7b7efa83ef44c43850 | D:\Code\refs\OS-Eval | `CalME/src/Main.cpp`、`CalME/src/matchFunc.cpp`、`CalME/config.txt` | 01 |
| OS-SIFT | https://github.com/xym2009/OS-SIFT | 631a8060042ed89ea11e43d97e44b387443b1eb6 | D:\Code\refs\OS-SIFT | 未审评测（只作为对比方法登记） | 02, 03 |
| CFOG | https://github.com/yeyuanxin110/CFOG | bdae6ade539ad8de0951d384e08c97694756a5a0 | D:\Code\refs\CFOG | 未审评测（只作为对比方法登记） | 02 |
| HOPC / HOPC_ncc | https://github.com/yeyuanxin110/HOPC | bb0aa72e9cbaf70a9892a4bb709fd8ae435beb8f | D:\Code\refs\HOPC | 未审评测 | 03 |
| HAPCG | https://github.com/yyxgiser/HAPCG-Multimodal-matching | 79aa1efee0039a9220897dba6f65faeb6b49bba6 | D:\Code\refs\HAPCG-Multimodal-matching | 未审评测 | 03 |
| SAR-SIFT（第三方） | https://github.com/yishiliuhuasheng/sar_sift | 6601368b5bf9cff418068721e2f2de1aa3ecc81d | D:\Code\refs\sar_sift | `ransac.py`、`sar_sift.py`（无 GT 评测） | 01, 03 |
| MOSS 数据 | https://github.com/betterlll/MOSS_data | 25c1cfcd1f0afc34a78af27dccd2637adf139a6e | D:\Code\refs\MOSS_data | 无（只有数据和 GT 点 `moss_datasets/*.txt`） | 01 |
| SOMA-1M | https://github.com/PeihaoWu/SOMA-1M | b86137c92b225402a4706eaee4bb7a61f11baabf | D:\Code\refs\SOMA-1M | 无（只有 README） | 01 |
| MapGlue（SOMA-1M 引用的协议） | https://github.com/PeihaoWu/MapGlue | 81b06ec0f95c6f635aa6486a66147fdf974f6e2c | D:\Code\refs\MapGlue | 无（只有 README） | 01 |
| ArePretrainedMatchers（rsim） | https://github.com/isaaccorley/rsim | 9950822cd7c9eaa23c233cc94099c1882ec6d2cc | D:\Code\refs\rsim | `src/rsim/spacenet9_matcher_benchmark.py`、`srif_matcher_benchmark.py`、`sarptical_pair_eval.py`；`scripts/` | 01（04 末尾提到过这篇） |
| vismatch（rsim 的匹配器接口） | https://github.com/gmberton/vismatch | 9d49b892ed21625cee00bc7797a45d352bf659a7 | D:\Code\refs\vismatch | `vismatch/base_matcher.py` | 01 |
| XoFTR | https://github.com/OnderT/XoFTR | e0fbea431b30be9742effbf5577c90aa8eb938f9 | D:\Code\refs\XoFTR | `test_relative_pose.py`、`src/utils/metrics.py` | 01, 04 |
| RoMa | https://github.com/Parskatt/RoMa | 77f8d68803526dcddfd9b7a46bc76125bdc25f15 | D:\Code\refs\RoMa | `romatch/benchmarks/hpatches_sequences_homog_benchmark.py` | 01 |
| DKM | https://github.com/Parskatt/DKM | ef57565db52684e661052ba82eb361329c63af3d | D:\Code\refs\DKM | `dkm/benchmarks/hpatches_sequences_homog_benchmark.py` | 01 |
| MINIMA | https://github.com/LSXI7/MINIMA | 796e7721174f9f829b79b3702bf8c2ae9a3d447a | D:\Code\refs\MINIMA | `test_relative_homo_mmim.py --choose_model 1`、`src/utils/metrics.py` | 01, 04 |
| GDROS | https://github.com/Zi-Xuan-Sun/GDROS | ee6b6216fe780bc03bfb4759ae094daa0cfddc60 | D:\Code\refs\GDROS | `test.py`、`core/LSRnet/LSmodel.py` | 04 |
| SOMA（traslauc，**不是** SOMA-1M） | https://github.com/traslauc/SOMA | 481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3 | D:\Code\refs\SOMA | `test.py`、`datasets/dataloader.py` | 01（只说明与 SOMA-1M 无关）, 04 |
| Shared Modality | https://github.com/BorisovAN/shmod | e3fa67f2bb2596b4046f5c67944cb9809bf9756f | D:\Code\refs\shmod | `scripts/matching_scripts/compute_matching_metrics.py`、`matching/homography.py`、`matching/detector.py` | 04 |
| RRSI | https://github.com/yeyuanxin110/RRSI | **无（空仓库，`rev-parse HEAD` 报错）** | D:\Code\refs\RRSI | 无 | 04 |
| MatchAnything（GitHub） | https://github.com/zju3dv/MatchAnything | 8cd8c1129a6d22dabea9405a869e4fad6ff8b630 | D:\Code\refs\MatchAnything | 只有 README | 04 |
| MatchAnything（HF Space，评测代码所在处） | https://huggingface.co/spaces/LittleFrog/MatchAnything | 6a7bcb589ec8da3a9e861e799122beaa5eba2193 | D:\Code\refs\MatchAnything-hf-space | `imcui/third_party/MatchAnything/tools/evaluate_datasets.py`、`scripts/evaluate/eval_visible_sar.sh` | 04 |
| HOMO-Feature | https://github.com/MrPingQi/HOMO_Feature_ImgMatching | 5f64b3536aae16154c7831936508dd67f97e2be2 | D:\Code\refs\HOMO_Feature_ImgMatching | 无评测（只有 `HOMO_image_matching_demo/A_HOMO_demo.m`） | 04 |
| Deep Image Analogy | https://github.com/msracver/Deep-Image-Analogy | 632b9287b42552e32dad64922967c8c9ec7fc4d3 | D:\Code\refs\Deep-Image-Analogy | 未审评测（张 ch3 迁移模块） | 03 |
| HardNet | https://github.com/DagnyT/hardnet | b1e9967299e36741bcc01d27c2312e91633d8443 | D:\Code\refs\hardnet | 未审评测 | 03 |
| Pix2Pix | https://github.com/phillipi/pix2pix | 89ff2a81ce441fbe1f1b13eca463b87f1e539df8 | D:\Code\refs\pix2pix | 未审评测 | 03 |
| Pix2PixHD | https://github.com/NVIDIA/pix2pixHD | 14b3b3c7fff413086e3b58df52096f16b6891172 | D:\Code\refs\pix2pixHD | 未审评测 | 03 |
| RIPE | https://github.com/fraunhoferhhi/RIPE | b173418008f4cb77a2ebffb570a7cab69e87cd08 | D:\Code\refs\RIPE | 训练验证用 `ripe/benchmarks/imw_2020.py`、`ripe/train.py`；论文表格的评测见 glue-factory fork | 05 |
| RIPE++ | https://github.com/fraunhoferhhi/RIPEpp | 0666e62bf00569870bbcc1fe4b2e03976fc563f8 | D:\Code\refs\RIPEpp | `ripepp/train.py`（选 best）、`ripepp/benchmarks/imw_2020.py`、`scared.py` | 05 |
| glue-factory（RIPE 作者 fork） | https://github.com/JohannesK14/glue-factory | 192baa367afc800614b224093b733433e012592b | D:\Code\refs\glue-factory-JohannesK14 | `gluefactory/eval/hpatches.py`、`eval/utils.py`、`eval/megadepth1500.py`；`configs/ripe+NN.yaml`、`ripepp+NN.yaml` | 05 |
| glue-factory（上游） | https://github.com/cvg/glue-factory | 2d17e3b3bd7d30f0c828d4c4d3eac4ecefbf283d | D:\Code\refs\glue-factory | 只用来确认里面没有 RaCo 评测 | 05 |
| SiLK | https://github.com/facebookresearch/silk | 7b9614b4a66361a0003aaa6fe9298ccdb267714b | D:\Code\refs\silk | `etc/mode/run-hpatches-tests-silk.yaml` → `lib/metrics/hpatches_metrics.py`、`lib/matching/mnn.py` | 05 |
| RaCo | https://github.com/cvg/RaCo | 35790eb48074ed14839d0fb496b8806caa4e766b | D:\Code\refs\RaCo | 无评测（只有推理代码） | 05 |
| CAPS | https://github.com/qianqianwang68/caps | 1cb601a2b77f505c4ad1bcc172568badffa9b86b | D:\Code\refs\caps | 只有 `extract_features.py` 和 `test/eval_pose_megadepth.py`，无单应评测 | 05 |
| GIM | https://github.com/xuelunshen/gim | f09105a0555eef5c93db032d97b9e3d8a4cccb20 | D:\Code\refs\gim | `test.py`（ZEB），无 HPatches 评测 | 05 |
| CA-Unsupervised（DeepHomography） | https://github.com/JirongZhang/DeepHomography | 3e811b7d06f84de34eae056baaf90ca886ae3908 | D:\Code\refs\DeepHomography | `Oneline-DLTv1/test.py` | 05 |
| GeoFormer | https://github.com/ruc-aimc-lab/GeoFormer | 8b9506e6e9c0e61848724955fb514824039b1ff7 | D:\Code\refs\GeoFormer | `eval_Hpatches.py`、`eval_FIRE.py`、`eval_ISC.py` → `eval_tool/immatch/utils/{hpatches,fire}_helper.py` | 05 |

## 与磁盘不一致 / 需注意

- **RRSI**：目录存在，但仓库没有任何 commit，无法锁定版本，也无法复现（见 04 §7）。
- **SOMA 与 SOMA-1M** 是两个无关的仓库。`D:\Code\refs\SOMA` 是 traslauc/SOMA（04 §4），SOMA-1M 论文的仓库是 `D:\Code\refs\SOMA-1M`（01 §8）。
- **LNIFT 与 LNIFT_exe**：`LNIFT` 是第三方复现（arunsahu159），`LNIFT_exe` 才是官方发布（01 §7、03 §4）。
- **glue-factory 有两份**：RIPE / RIPE++ 的论文表格用的是 JohannesK14 的 fork；上游 cvg 版只用来确认里面没有 RaCo。
- **MatchAnything-hf-space** 不是 GitHub 仓库，是 HuggingFace Space 的 git 克隆，permalink 也是 HF 的 blob 链接。
- 这些目录**只登记了、没有审评测代码**（只作为对比方法出现）：RIFT2、LNIFT_exe、OS-SIFT、CFOG、HOPC、HAPCG、Deep-Image-Analogy、hardnet、pix2pix、pix2pixHD。
- 分文件标为「已克隆」，但磁盘上找不到的仓库：**没有**。

## 未找到源码的工作

| 工作 | 情况 | 出处 |
|---|---|---|
| MultiResSAR 综述的指标脚本 | 没有公开 | 01 §1 |
| SOMA-1M / MapGlue 的评测代码 | 仓库只有 README | 01 §8 |
| HOWP（项目网页）、ASS（gitee，03 也没找到） | 未获取 | 01 §7、03 §4 |
| ROS-PC（王丽娜 ch5） | 没有找到公开源码 | 02 §4 |
| RDLNet、TSRM、张 ch5 方法、PSGF、EC-RIFT、吕 ch6 方法 | 论文没有给链接 | 03 §4 |
| PSO-SIFT、SAR-SIFT、FSC | 只找到第三方实现：ZeLianWen/Image-Registration、ShowStopperTheSecond/FSC，**没有克隆** | 03 §4 |
| RSCJ、PSO 模板匹配、WOA、Powell | 未找到 | 03 §4 |
| TAR、AnyMatch | 论文和 arXiv 页面都没有代码链接 | 04 §6、§10 |
| RRSI | 仓库为空 | 04 §7 |
| LoRetta（只有项目页）、MatchAnyEvents（有 repo，没有克隆）、MatchGS（没有查） | 不涉及光-SAR | 04 §0 |
| S2M2-SAR、XCP-Match | 未找到 | 05 §8、§9 |
| RaCo 评测代码 | 声称会发布到 glue-factory，目前还没有 | 05 §5 |
| CAPS、GIM 的 HPatches 单应评测代码 | 官方仓库里没有 | 05 §6、§7 |
| R2D2、DISK、PoSFeat、DeDoDe、MuM、nexus2、H-ViT | 有 repo，但没有克隆（不在细看范围内） | 05 §1 |
| SuperGlue、LightGlue、LoFTR、Efficient LoFTR、SGM-Net、XFeat | 评测都是自然图像位姿任务，没有克隆 | 01 §7 |

## 一键恢复克隆

```bash
# 按本表恢复 D:/Code/refs 下的全部克隆（已存在的目录会跳过）。RRSI 是空仓库，不在列表里。
cd /d/Code/refs
while read -r dir url commit; do
  [ -d "$dir" ] && { echo "skip $dir"; continue; }
  git clone "$url" "$dir" && git -C "$dir" checkout "$commit"
done <<'EOF'
Multi-Resolution-SAR-dataset- https://github.com/betterlll/Multi-Resolution-SAR-dataset- 6f664903bc9662a66ac7c2c88a39f5e2d9da1ec4
SRIF https://github.com/LJY-RS/SRIF 88881a3a8789d0bed6a8df91b943e02f2593a0cd
RIFT-multimodal-image-matching https://github.com/LJY-RS/RIFT-multimodal-image-matching 7ea830e2f13cc3c226f975fe9e98b7666a8f26fb
RIFT2-multimodal-matching-rotation https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation 0e980ce4124d2abd62727dcf040198db0a18b269
LNIFT_exe https://github.com/LJY-RS/LNIFT_exe 6349a8c47b4fc3c58c28c8aa526096831de06bdd
LNIFT https://github.com/arunsahu159/LNIFT-Locally-Normalized-Image-for-Rotation-Invariant-Multimodal-Feature-Matching 1ffa3528cd88a655d9fb7e99dafb1a27ebe8d1ca
OS-Eval https://github.com/xym2009/OS-Eval 4df7d5a780d4c2d5b913ac7b7efa83ef44c43850
OS-SIFT https://github.com/xym2009/OS-SIFT 631a8060042ed89ea11e43d97e44b387443b1eb6
CFOG https://github.com/yeyuanxin110/CFOG bdae6ade539ad8de0951d384e08c97694756a5a0
HOPC https://github.com/yeyuanxin110/HOPC bb0aa72e9cbaf70a9892a4bb709fd8ae435beb8f
HAPCG-Multimodal-matching https://github.com/yyxgiser/HAPCG-Multimodal-matching 79aa1efee0039a9220897dba6f65faeb6b49bba6
sar_sift https://github.com/yishiliuhuasheng/sar_sift 6601368b5bf9cff418068721e2f2de1aa3ecc81d
MOSS_data https://github.com/betterlll/MOSS_data 25c1cfcd1f0afc34a78af27dccd2637adf139a6e
SOMA-1M https://github.com/PeihaoWu/SOMA-1M b86137c92b225402a4706eaee4bb7a61f11baabf
MapGlue https://github.com/PeihaoWu/MapGlue 81b06ec0f95c6f635aa6486a66147fdf974f6e2c
rsim https://github.com/isaaccorley/rsim 9950822cd7c9eaa23c233cc94099c1882ec6d2cc
vismatch https://github.com/gmberton/vismatch 9d49b892ed21625cee00bc7797a45d352bf659a7
XoFTR https://github.com/OnderT/XoFTR e0fbea431b30be9742effbf5577c90aa8eb938f9
RoMa https://github.com/Parskatt/RoMa 77f8d68803526dcddfd9b7a46bc76125bdc25f15
DKM https://github.com/Parskatt/DKM ef57565db52684e661052ba82eb361329c63af3d
MINIMA https://github.com/LSXI7/MINIMA 796e7721174f9f829b79b3702bf8c2ae9a3d447a
GDROS https://github.com/Zi-Xuan-Sun/GDROS ee6b6216fe780bc03bfb4759ae094daa0cfddc60
SOMA https://github.com/traslauc/SOMA 481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3
shmod https://github.com/BorisovAN/shmod e3fa67f2bb2596b4046f5c67944cb9809bf9756f
MatchAnything https://github.com/zju3dv/MatchAnything 8cd8c1129a6d22dabea9405a869e4fad6ff8b630
MatchAnything-hf-space https://huggingface.co/spaces/LittleFrog/MatchAnything 6a7bcb589ec8da3a9e861e799122beaa5eba2193
HOMO_Feature_ImgMatching https://github.com/MrPingQi/HOMO_Feature_ImgMatching 5f64b3536aae16154c7831936508dd67f97e2be2
Deep-Image-Analogy https://github.com/msracver/Deep-Image-Analogy 632b9287b42552e32dad64922967c8c9ec7fc4d3
hardnet https://github.com/DagnyT/hardnet b1e9967299e36741bcc01d27c2312e91633d8443
pix2pix https://github.com/phillipi/pix2pix 89ff2a81ce441fbe1f1b13eca463b87f1e539df8
pix2pixHD https://github.com/NVIDIA/pix2pixHD 14b3b3c7fff413086e3b58df52096f16b6891172
RIPE https://github.com/fraunhoferhhi/RIPE b173418008f4cb77a2ebffb570a7cab69e87cd08
RIPEpp https://github.com/fraunhoferhhi/RIPEpp 0666e62bf00569870bbcc1fe4b2e03976fc563f8
glue-factory-JohannesK14 https://github.com/JohannesK14/glue-factory 192baa367afc800614b224093b733433e012592b
glue-factory https://github.com/cvg/glue-factory 2d17e3b3bd7d30f0c828d4c4d3eac4ecefbf283d
silk https://github.com/facebookresearch/silk 7b9614b4a66361a0003aaa6fe9298ccdb267714b
RaCo https://github.com/cvg/RaCo 35790eb48074ed14839d0fb496b8806caa4e766b
caps https://github.com/qianqianwang68/caps 1cb601a2b77f505c4ad1bcc172568badffa9b86b
gim https://github.com/xuelunshen/gim f09105a0555eef5c93db032d97b9e3d8a4cccb20
DeepHomography https://github.com/JirongZhang/DeepHomography 3e811b7d06f84de34eae056baaf90ca886ae3908
GeoFormer https://github.com/ruc-aimc-lab/GeoFormer 8b9506e6e9c0e61848724955fb514824039b1ff7
EOF
```
