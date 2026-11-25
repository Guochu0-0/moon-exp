# #270 传统方法接入与执行记录

代码准备于 2026-11-17。原版 RIFT、OS-SIFT 和 HIMO 已在 154 完成正式推理，160 完成拟合和评测。结论见 [B0 Notes](../notes.md#himo-的-10-px-成功率在三种传统方法中最高)。

## 收集边界验证

Spec 预审发现，旧收集端会将全部 MATLAB 异常或 `--limit` 子集发布为正式 raw，随后被统一拟合误算成模型失败。用真实 SciPy MAT 文件与模拟 split/manifest 复现，修复前 8 项测试有 4 项失败。收尾在 160 的工作台环境复核了 14 项回归，全部通过。

正式prep/collect现在禁止非零limit；任何MATLAB异常、缺文件、缺全量manifest、坏坐标均在发布前拒绝整批。即使只有一对报错也不猜测其原因，先解决异常再发布；正常计算出的零匹配仍是合法模型失败。写文件先在同盘临时目录完成序列化，再替换正式raw，避免序列化失败破坏旧文件。自检仅允许Val与正limit，存入独立目录。

验证命令（源码`e68c658`，服务器worktree`/remote-home/xufang/YGC/wt/baseline-270`）：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/envs/wb/bin/python runs/B0/code/test_traditional270.py
```

## 环境状态

154 使用已获授权的 R2024b Update 10 MATLAB 会话，通过 Engine 运行匹配。共享盘写入错误后，RIFT Test 与 OS-SIFT Val 从已写好的 MAT 恢复。HIMO Test 在更新在线授权后重跑，HIMO Val 收集的 Git 超时通过延长等待恢复。推理尝试的结束状态与执行环境见 `runs/B0/launch/`。

## 已准备的方法

- RIFT固定到`7ea830e2f13cc3c226f975fe9e98b7666a8f26fb`，严格调用官方`RIFT_no_rotation_invariance(...,4,6,96)`和demo的最近邻/唯一化，输出FSC之前点对，置信度为 `1/(1+matchmetric)`。上游说明这是不含旋转不变性与角点检测的最简公开实现。保留作者对边界描述子的原始处理，其索引错误可能留下零描述子。沿B0使用浮点映射，不额外uint8量化。
- OS-SIFT固定到`631a8060042ed89ea11e43d97e44b387443b1eb6`，官方光学/SAR分支、尺度与描述子。SAR前两波段相加，有限非负值除以patch最大值，然后执行官方`imadjust`及+0.001，这是线性SAR必要适配。`CSC2.m`在描述子ratio=0.95后第一次FSC之前截断，输出光学→SAR点对，置信度为 `1-ratio`。上游demo虽计算最强5000点子集，但实际将全体检测点传入描述子。
- HIMO 固定到 `1a525611a22e15c0862d2af6e3f53857f09a7e70`。用户确认保留参与后续匹配的逐尺度内部几何筛选，在全局去重后、最后一次全局去外点前导出 `MMatches`，再用 B0 的 3 px RANSAC。B0 负责读图和输入映射，调用作者的 `Deal_Extreme` 和 `Preproscessing` P-code，并核对倍率为 1 且输出尺寸与原 patch 一致。算法核心为 `func_HIMO` 与 `func_Math` 的 MATLAB 源码。两份补丁将原始距离与去重、参考匹配和内点掩码对齐，置信度为 `1/(1+matchmetric)`。空候选返回空数组，最佳尺度票数并列时按 MATLAB 列主序取首项。

## 排除与条件项

- ASS：作者Gitee核心`ASS_detector.p`/`ASS_descriptor.p`/`ASS_map.p`未公开源码，按已确认规则排除。
- MOSS：综述Table6所指作者仓库`betterlll/MOSS_data`只有数据与说明，无算法实现，排除。
- SRIF：作者`LJY-RS/SRIF`的`algorithms/SRIF`只有exe/DLL，排除。
- HOWP：2026-11-18 静态核验[作者发布包](https://skyearth.org/publication/project/HOWP/data/HOWP_Code.rar)，大小 23,392 字节，SHA-256 为 `3298f4b2640580764ce9077e0745ae67d714579eb50a845494a407dd32eb0a3c`。`HOWP_Func/` 的 15 个文件全为 MATLAB P-code，包含 `HOWP.p`、特征提取、加权相位与对数极坐标描述子。可读的 `HOWP_main.m` 在匹配前调用 `HOWP`。按核心源码未公开排除，本轮未测。

官方来源：

- https://github.com/LJY-RS/RIFT-multimodal-image-matching
- https://github.com/xym2009/OS-SIFT
- https://github.com/MrPingQi/HIMO_ImgMatching
- https://gitee.com/xxin08/ASS
- https://github.com/betterlll/MOSS_data
- https://github.com/LJY-RS/SRIF
- https://skyearth.org/publication/project/HOWP/

## 恢复环境后的执行

先按server技能在可连GitHub的服务器建`baseline-270`独立worktree，初始化本票三个固定submodule。不改变主checkout及共享环境。准备/输出中间mat目录放容器本地盘，例如`/opt/tmp/baseline270`，一次只运行一个MATLAB进程、2线程、关闭自动parpool。

1. 先做服务器语法/收集边界检查，再每法取Val前3对做(7,-5)光学平移自检。OS-SIFT的不同分支可能导致匹配精度差异，需要分别核对MATLAB原始点为1-based xy、收集端减1、最终拟合阶段两侧+0.5。自检不通过就不能产出正式结果。
2. 用 `traditional270.py prep --method <rift|ossift|himo> --split val --io <scratch> --limit 3 --selfcheck` 准备自检，批次目录为 `<scratch>/<method>/val_selfcheck`。输入准备与 Engine 分属两台主机时，scratch 须两台主机都可读写。
3. MATLAB 会话获授权并共享后，用 `traditional270_engine.py <session> <batch>` 调用固定两个线程的 MATLAB 入口并写启动记录。HIMO 的补丁由入口在临时目录应用。用 `traditional270.py collect` 检查四个角像素中心和图像中心经 B0 的 3 px RANSAC 后与已知平移的最大差值，须不超过 1 px。
4. 自检通过后Val/Test各做prep→MATLAB→collect，正式运行不带limit/selfcheck。每次准备输入都用新 `--io` 路径，原启动中断时可续用同一批次。输入manifest核对完整pair列表与代码提交。缺mat时collect直接退出，不生成“算法失败”。
5. 收集的原始结果在`runs/B0/raw/<method>`；`python -m baselines.fit runs/B0/raw/<method> --split <val|test> --run runs/B0 --name <method>`生成标准预测。任何方法全体环境报错时先修环境，不能把整组报错作为模型结果。
6. 每次正式运行记录参数、MATLAB版本/工具箱/线程/seed0和真实命令；收尾补齐end/status。完整结果后再追加exp方法名称、B0 Notes数字表并运行eval/check。新worktree未必带旧方法全部预测，不可直接用局部结果覆盖B0旧metrics：仅合并本票新指标，或在明确防止收尾搬移的情况下只读借用权威旧预测，生成完整指标。多票B0增量由主代理串行合并。

三种方法各收集 652 对 Val 和 953 对 Test。完整 B0 评测重新计算了 25 个原有方法，其存档指标和协议逐项一致。
