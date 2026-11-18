# #270 传统方法接入与执行前检查

代码准备于 2026-11-17；尚未在 MATLAB 中执行，不是已完成补测。输入/收集边界已在160服务器CPU上验证。

## 收集边界验证

Spec预审发现：旧收集端会将全部MATLAB异常或`--limit`子集发布为正式raw，随后被统一拟合误算成模型失败。修复前在160的`/opt/envs/wb/bin/python`用真实SciPy写读MAT与假split/manifest复现，8项测试有4项失败；修复后增加坏MAT、输入准备limit、自检manifest混用、序列化失败保护等检查，12项全部通过（0.078秒）。正式MATLAB推理尚未发生，B0旧结果不受该修复影响。

正式prep/collect现在禁止非零limit；任何MATLAB异常、缺文件、缺全量manifest、坏坐标均在发布前拒绝整批。即使只有一对报错也不猜测其原因，先解决异常再发布；正常计算出的零匹配仍是合法模型失败。写文件先在同盘临时目录完成序列化，再替换正式raw，避免序列化失败破坏旧文件。自检仅允许Val与正limit，存入独立目录。

验证命令（源码`e68c658`，服务器worktree`/remote-home/xufang/YGC/wt/baseline-270`）：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/envs/wb/bin/python runs/B0/code/test_traditional270.py
```

## 环境状态

154 外网 SSH 被网关关闭；经可达160探查，154的内网20020端口拒绝连接。150/160上只读检查`command -v matlab`、`/opt/matlab*`、`/opt/MATLAB*`、`/usr/local/MATLAB*`与`/root/matlab-batch.sh`，均未发现已有环境。需先恢复正式MATLAB环境，不能把这种环境阻碍计为算法失败，不在用户电脑运行正式测试。

## 已准备的方法

- RIFT固定到`7ea830e2f13cc3c226f975fe9e98b7666a8f26fb`，严格调用官方`RIFT_no_rotation_invariance(...,4,6,96)`和demo的最近邻/唯一化，输出FSC之前点对。上游说明这是不含旋转不变性与角点检测的最简公开实现，不能声称覆盖论文完整实现。沿B0使用浮点映射，不额外uint8量化。
- OS-SIFT固定到`631a8060042ed89ea11e43d97e44b387443b1eb6`，官方光学/SAR分支、尺度与描述子。SAR前两波段相加，有限非负值除以patch最大值，然后执行官方`imadjust`及+0.001；这是线性SAR必要适配，不是输入搜参。`CSC2.m`在描述子ratio=0.95后第一次FSC之前截断，输出光学→SAR点对；不执行几何引导重匹配。上游demo虽计算最强5000点子集，但传入描述子的是全体检测点；本接入跟实际执行路径，不擅自改变数量。
- HIMO固定到`1a525611a22e15c0862d2af6e3f53857f09a7e70`，尚未决定正式输出协议。逐尺度内部几何筛选参与后续匹配策略，不能把其最终点对称为完全未经过几何筛选的原始匹配。可能的最小适配是保留内部策略，只在末尾“One last outlier removal”之前导出MMatches，需用户先确认例外；当前没有为HIMO生成正式配置/预测。

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
2. 用`traditional270.py prep --method <rift|ossift> --split val --io /opt/tmp/baseline270 --limit 3 --selfcheck`准备自检；`traditional270_split`的输入输出目录分别是`<io>/<method>/val_selfcheck/in`和`out`。
3. 恢复并确认MATLAB有效授权后，记录`workbench.launch.begin`，在脚本固定线程下调用`traditional270_split(method,repo,in_dir,out_dir,2)`，收集并用统一RANSAC检查已知平移。输入/输出要给绝对路径。
4. 自检通过后Val/Test各做prep→MATLAB→collect，正式运行不带limit/selfcheck。输入manifest核对完整pair列表；缺mat时collect直接退出，不生成“算法失败”。
5. 收集的原始结果在`runs/B0/raw/<method>`；`python -m baselines.fit runs/B0/raw/<method> --split <val|test> --run runs/B0 --name <method>`生成标准预测。任何方法全体环境报错时先修环境，不能把整组报错作为模型结果。
6. 每次正式运行记录参数、MATLAB版本/工具箱/线程/seed0和真实命令；收尾补齐end/status。完整结果后再追加exp方法名称、B0 Notes数字表并运行eval/check。新worktree未必带旧方法全部预测，不可直接用局部结果覆盖B0旧metrics：仅合并本票新指标，或在明确防止收尾搬移的情况下只读借用权威旧预测，生成完整指标。多票B0增量由主代理串行合并。

本次准备已提交代码并验证输入/收集边界；未启动MATLAB/GPU、未生成新方法预测、未声称实验验收。PR保持草稿，不自动关票。
