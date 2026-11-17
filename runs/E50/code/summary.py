"""最后存档、完整 Val 曲线、全精度逐对配对 bootstrap 与自包含 Notes。"""
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from workbench import launch
from workbench.dataset import Dataset
from workbench.evaluate import jsonable, pair_errors
from workbench.protocol import summarize
from workbench.records import load_runs
from workbench.tbwriter import ScalarWriter

R = ROOT / "runs/E50"


def paired(a, b):
    score = np.clip(1 - a / 5, 0, None) - np.clip(1 - b / 5, 0, None)
    idx = np.random.default_rng(0).integers(0, len(score), size=(2000, len(score)))
    return {"delta_auc5": float(score.mean()), "ci95": np.percentile(score[idx].mean(1), [2.5, 97.5]).tolist(),
            "resamples": 2000, "seed": 0, "n": len(score)}


def duration(rows):
    return sum((datetime.strptime(r["end"], "%Y-%m-%d %H:%M:%S %z") -
                datetime.strptime(r["start"], "%Y-%m-%d %H:%M:%S %z")).total_seconds()
               for r in rows if r["end"])


def main():
    ds = Dataset("/remote-home/xufang/YGC/dataset/Moon")
    old = load_runs(Path("/remote-home/xufang/YGC/moon-exp/runs"))
    own = load_runs(ROOT / "runs")["E50"]
    refs = [("B0m", "anymatch_roma__minmax", "发布起点", "zero-shot"),
            ("T13", "main__step8000", "已有伪标签自训练", "最后8000"),
            ("E41", "main__screen0.8", "E41主方法", "最后7907"),
            ("E50", "main", "本实验", "最后7907")]
    rows, errors, comparisons = [], {}, []
    for split, expected in (("val", 652), ("test", 953)):
        assert len(ds.labelled(split)) == expected
        for node, method, name, rule in refs:
            run = own if node == "E50" else old[node]
            assert (run.dir / "preds" / method / f"{split}.jsonl").exists(), (node, method, split)
            preds = run.preds(method, split)
            assert set(ds.labelled(split)) <= set(preds), (node, split, "missing predictions")
            err = pair_errors(ds, split, preds)
            errors[node, split] = err
            rows.append({"node": node, "method": method, "name": name, "rule": rule, "split": split,
                         "summary": summarize(err)})
        for node, method, name, rule in refs[:-1]:
            comparisons.append({"reference": f"{node}/{method}", "split": split,
                                **paired(errors["E50", split], errors[node, split])})
    sweep = json.loads((R / "sweep/main/S/metrics.json").read_text())["methods"]
    curve = sorted((int(k[4:]), v["val"]["summary"]) for k, v in sweep.items() if "val" in v)
    assert [s for s, _ in curve] == list(range(250, 7751, 250)) + [7907]
    assert [k for k, v in sweep.items() if "test" in v] == ["step7907"]
    peak = json.loads((R / "sweep/main/peak.json").read_text())
    assert peak["step"] == 7907 and peak["metric"] == "last"
    best_step, best = max(curve, key=lambda x: x[1]["auc@5"])
    train = [json.loads(x) for x in (R / "ckpt/main/log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in train] == list(range(1, 7908))
    assert sum(len(r["pairs"]) for r in train) == 31628
    assert train[7115]["lr"] == 5e-5 and train[7116]["lr"] == 5e-6
    labels = json.loads((R / "extra/labels_meta.json").read_text())
    overflow = R / "ckpt/main/overflow.jsonl"
    skipped = len(overflow.read_text().splitlines()) if overflow.exists() else 0
    stages = json.loads((R / "launch/main_train.json").read_text())
    assert stages[-1]["status"] == "ok" and stages[-1]["end"]
    evaluation_envs = []
    for step, _ in curve:
        meta = json.loads((R / f"sweep/main/match/step{step}/val.meta.json").read_text())
        evaluation_envs.append({"step": step, "split": "val", "host": meta["env"]["host"],
                                "gpu": meta["env"]["gpu"], "env": meta["env"]})
    meta = json.loads((R / "sweep/main/match/step7907/test.meta.json").read_text())
    evaluation_envs.append({"step": 7907, "split": "test", "host": meta["env"]["host"],
                            "gpu": meta["env"]["gpu"], "env": meta["env"]})
    data = R / "extra/data"
    data.mkdir(exist_ok=True)
    report = {"results": rows, "paired": comparisons, "best_val": {"step": best_step, "summary": best},
              "updates": 7907, "gradient_pairs": 31628, "amp_skips": skipped,
              "train_process_sec": duration(stages), "train_unfinished_attempts": sum(not r["end"] for r in stages),
              "train_failed_attempts": sum(r["status"] == "fail" for r in stages),
              "teacher_process_sec": labels["teacher_process_sec"],
              "teacher_unfinished_attempts": labels["teacher_unfinished_attempts"],
              "evaluation_environments": evaluation_envs}
    (data / "summary.json").write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=1) + "\n")
    with open(data / "val_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", *curve[0][1].keys()])
        w.writeheader()
        w.writerows({"step": s, **v} for s, v in curve)
    with ScalarWriter(R / "tb/main/validation") as writer:
        for step, s in curve:
            writer.add("val/auc@5", s["auc@5"], step)
    # wb 环境不装绘图库；用已有 matplotlib 的训练环境只画图，不改变指标计算环境。
    subprocess.run(["/opt/envs/loftr/bin/python", str(R / "code/plot.py")], cwd=ROOT, check=True)
    result = lambda node, split: next(r["summary"] for r in rows if r["node"] == node and r["split"] == split)
    val, test = result("E50", "val"), result("E50", "test")
    metrics = []
    for r in rows:
        s = r["summary"]
        values = [f'{s[f"auc@{k}"]:.4f}' for k in (3, 5, 10, 20)]
        values += [f'{100*s[f"sr@{k}"]:.1f}%' for k in (3, 5, 10, 20)]
        median = f'{s["median"]:.3f}' if np.isfinite(s["median"]) else "∞"
        values += [f'{100*s["fail_rate"]:.2f}%', median]
        metrics.append(f'| {r["node"]}/{r["method"]}（{r["rule"]}） | {r["split"]} | ' + " | ".join(values) + " |")
    paired_table = [f'| {r["reference"]} | {r["split"]} | {r["delta_auc5"]:+.4f} | '
                    f'[{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}] |' for r in comparisons]
    notes = f'''# 固定伪仿射适配在原始 Val/Test 上的 AUC@5 为 {val["auc@5"]:.4f}/{test["auc@5"]:.4f}

**摘要**

本实验为月球光学–SAR 仿射配准增加 SCENES bootstrapping 的 RoMa／仿射任务适配对照。只用发布起点在无标注 Train 上生成一次伪仿射，再从同一起点训练 VGG 和完整 decoder；不使用目标域标注，也不混入有标注源域数据。固定种子 0 完成 7907 次有效更新，最后存档在原始 Val/Test 的 AUC@5 分别为 {val["auc@5"]:.4f}/{test["auc@5"]:.4f}。全部逐存档 Val 读数与最后存档 Test 已保存，比较的是完整训练配方，不能单独归因于损失形式。

**引言**

RoMa 先用图像特征给出粗对应位置，再用多级细化逐步调整位置和匹配置信度。SCENES bootstrapping 用起点模型的匹配估计几何关系，再以这一固定几何为监督微调模型。本任务要估计光学到 SAR 的仿射，因此以每个 patch 对一个伪仿射代替极线几何，并保留 RoMa 原生的锚点分类、鲁棒位置回归和 certainty 损失。这里检验这套适配在既定比较预算下的结果，不把它称作原始极线损失的严格复现。

**方法**

*来源与适配。* [SCENES §3.4、§4](https://arxiv.org/html/2401.10886v1#S3.SS4) 给出 bootstrapping 和论文的 100 匹配／20 内点门槛；[官方骨架 a6962bef](https://github.com/dakloepfer/scenes-finetuning/tree/a6962befb169dc5a2f06ca8fee075b1ea289ea30) 没有完整训练配置。本实验按 [AnyMatch 公开训练入口 259ad343](https://github.com/MnYangs/AnyMatch/blob/259ad343ffa9ca0703965f56880b3b5d2ecbc9e1/third_party/RoMa_AnyMatch/train_roma_outdoor.py) 设定底模范围、优化器和共同增强。公开脚本不等于发布权重实际使用命令已被确认。更换模型及几何/损失、不混有标注 MegaDepth、缩短训练预算并映射学习率下降点，均属于本次适配和比较协议。

*数据与固定几何。* 起点为 B0m/anymatch_roma__minmax 的发布权重 anymatch/RoMa_AnyMatch.pth，不接 E41 或其他已微调模型。光学 div255、SAR 逐 patch minmax；统一推理为长边 640、粗级 560、上采样 864、双向预测、threshold_balanced 采样 10000 点，种子 0 逐对重置。全部 7907 个 Train patch 对的原始点对保存在 raw/teacher_*/train.npz，推理实际参数、权重 SHA256 和环境在各 train.meta.json 及 [标签身份记录](extra/labels_meta.json)。以原始 512 网格 RANSAC 3 px 拟合仿射；阈值是本任务口径，不是 SCENES 的 F 矩阵阈值。只要求拟合成功、匹配至少 100、内点至少 20。固定伪仿射见 [逐对标签](extra/labels.jsonl)，名单见 [保留名单](extra/kept.json)。不做前一半筛选、内点率阈值、加权或训练中重估。

| Train 总数 | 保留数 | 内点数最小值／25%／中位数／75%／最大值 |
|---:|---:|---|
| 7907 | {labels["kept"]} | {" / ".join(f"{x:g}" for x in labels["n_inliers_quantiles"])} |

表 1：只按预定的匹配数和内点数门槛确定训练集合，分位数按全部 Train 统计。

*训练配方。* VGG 和完整 decoder（粗级及五级细化）参与训练，BN 沿用上游训练模式；DINOv2 按上游冻结提特征，无 LoRA。训练输入 560；两图共同做 x/y 各 [-32,32] 整数平移，在 ImageNet 归一化后补零，再以 0.5 概率共同水平翻转。固定仿射精确变为 G A G⁻¹，查询和目标两侧的补零区都纳入有效区计算。保留最近锚点分类 CE、多尺度鲁棒回归、certainty BCE，ce_weight=0.01、alpha=0.5、c=1e-4、local_dist={{1:4,2:4,4:8,8:8}}、local_largest_scale=8 和上游尺度权重；不带入软锚点、残差剔除、额外容差、稠密老师或可匹配检查。

*优化与预算。* AdamW，默认 betas=(0.9,0.999)、eps=1e-8，weight decay=0.01；单卡 batch=4，无累积。VGG lr=2.5e-6、decoder 各部分 lr=5e-5；梯度范数裁剪 0.01，无 warmup。MultiStepLR 在完成第 7116 次更新后乘 0.1，即第 7117 次更新起使用衰减学习率。AMP GradScaler 的增长间隔与公开脚本一致为 1000000；溢出被跳过的尝试不计入有效更新预算，记录 {skipped} 次。实际完成 7907 次更新、31628 个参与更新的 patch 对次；筛选后的集合按种子 0 重复打乱遍历。每 250 次存档，另存最后 7907。每 500 次更新退出训练子进程、释放 GPU，再恢复模型、优化器、缩放器、数据顺序和全部随机流；不根据 Val 早停。训练进程累计 {report["train_process_sec"]/3600:.2f} 小时（包含已结束的失败尝试）；伪标签匹配进程累计 {labels["teacher_process_sec"]/3600:.2f} 小时（含模型加载、I/O 及已结束的失败尝试），拟合 {labels["fit_sec"]:.1f} 秒，二者分开记录。未确认结束的尝试数：训练 {report["train_unfinished_attempts"]}、匹配 {labels["teacher_unfinished_attempts"]}，若非零则这些尝试的完整耗时未知。

*评价与统计。* 原始 Val 652 对，原始 Test 953 对。每个存档都评 Val，正式结果固定取最后 7907；Test 只评这一存档。仿射按统一 RANSAC 3 px 拟合，逐 patch 对误差为光学检查点经预测仿射到 SAR 检查点的平均距离，失败记 ∞，全部 patch 对等权。AUC@T 是 [0,T] 误差累积曲线的归一化面积；SR@T 是误差不超过 T 的比例，主要比较 AUC@5。配对比较按同一 patch 对的全精度误差做 2000 次重抽、种子 0，给差值的 95% 区间。区间只反映该数据集按 patch 对重抽的波动，不是训练种子波动。

资源允许时，不同存档的 Val 评测并行；空闲卡不足时用可用空卡继续。每个 patch 对仍独立按相同参数推理，全部 Val 结束后才汇总和评最后存档 Test。每份完整预测的实际主机、GPU 和环境版本均保存在原始匹配 meta，并汇入 summary.json 的 evaluation_environments；曲线包含的 GPU 型号为 {"、".join(sorted({e["gpu"] for e in evaluation_envs if e["split"] == "val"}))}，最后存档 Test 使用 {evaluation_envs[-1]["gpu"]}。调度没有修改单对推理参数，但跨 GPU 的浮点结果不保证逐位相同。

*历史对照身份。* T13 的训练记录写明学生 init 为空、使用 anymatch/RoMa_AnyMatch.pth，输入为光学 div255、SAR minmax；这里按这一记录比较，不将图中的 M15 关系解释为学生从 M15 权重续训。历史记录未含发布权重 SHA256，因此只能核对发布权重路径和输入配置，不能逐字节确认历史权重与本实验相同。

**结果**

*最后存档与起点、既有自训练和 E41 的指标已按同一原始 split 口径列出。* 表 2 均用已有最后存档或 zero-shot。T13/main__step8000 是已有自训练的最后存档；E41/main__screen0.8 的 Val/Test 都是最后 7907，不将历史 Val 峰值的缺失存档当作 Test 对照。AUC 在 [0,1]，SR 和失败率为百分比，误差中位数单位为原始网格 px。

| 方法（选取规则） | split | AUC@3 | AUC@5 | AUC@10 | AUC@20 | SR@3 | SR@5 | SR@10 | SR@20 | 失败率 | 中位误差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

表 2：所有正式比较均以最后存档或发布起点为依据，失败不从分母剔除。

*完整 Val 曲线的最高读数出现在第 {best_step} 次更新，AUC@5 为 {best["auc@5"]:.4f}。* 这项补充读数不替换正式最后存档，也未用该存档补评 Test。全部 32 个存档的数值见 [Val 曲线数据](extra/data/val_curve.csv)。

![](extra/val_curve.png)
图 1：固定种子 0 的全部 Val 存档读数；黑点是正式结果采用的最后存档，水平虚线是发布起点和 E41 最后存档。

*同一 patch 对上的 AUC@5 差及重抽区间见表 3。* 差值定义为本实验最后存档减参考方法，失败也保留在配对样本中。

| 参考方法 | split | ΔAUC@5 | 配对 bootstrap 95% 区间 |
|---|---|---:|---|
{chr(10).join(paired_table)}

表 3：每行 2000 次按 patch 对配对重抽，不代表跨训练种子的稳定性。

真实模型检查核对了共同增强的图像接线、双侧有效区、原生 BN 统计更新、VGG/粗级/细化梯度路径和参数组；同一真实输出上的原生损失数值及输出梯度与固定上游版本对照，详细值见 [诊断记录](extra/probe.json)。首轮诊断观测到默认初始 AMP 缩放下粗级溢出，后续诊断验证自动降低缩放后梯度有限；正式预算按成功更新计数。实现测试及启动依据见 launch/ 和 PR。

启动前的代码洁净检查曾因共享盘 Git status 超过 120 秒而退出，当时没有生成伪标签或开始训练；随后显式将 Git 等待预算设为 600 秒，仍逐项检查代码洁净，不使用 dirty 放行。原始错误见 [启动前错误记录](extra/startup_failure.txt)。这一处理只改变等待时限，不改变训练或评价配方。

训练完成后的 Val 评测中，已完成 16 个存档时检测到两张卡出现额外 CUDA 进程，按共享用卡规则撤下了本方两个评测进程。一次单卡恢复又遇到同卡任务，再次撤下本方进程；随后迁至 150 的空闲 RTX 3090，补齐四份 Val。第 20 份完整落盘后，160 的两张 V100 已重新持续空闲，因此停止刚开始的下一份评测并迁回双卡，以缩短等待。始终保留训练和完整 Val 记录，重算中断的存档，没有重新训练或重估伪仿射。中断和恢复均保留在 main、pipeline 与 pipeline_resume 的启动记录中，两次单卡阶段日志另存 extra/eval_resume_interrupted_160.txt 与 extra/eval_resume_migration_150.txt。

**讨论**

本实验补齐了预定预算下 SCENES bootstrapping 的 RoMa／仿射任务适配读数，完整数值与配对区间见 [汇总数据](extra/data/summary.json)。它与 E41 的可训练范围、BN 模式、扰动和优化配方不同，因此差异属于整套训练配方的比较，不能单独归因于 RL 或伪标签损失。目标域训练不使用标注，不否认起点含有预训练监督；没有源域混训和缩短预算也限制了对原论文的外推。

只运行种子 0，不能据此判断训练随机性下的稳定性；按 patch 对重抽的区间也未处理同 ROI 内的相关性。固定伪仿射本身可能有误，本实验没有用目标域标注改写它。曲线和分布提供了观察依据，未进一步检验产生差异的原因。没有重训 E41，也没有为缺失的 E41 峰值存档补训练。
'''
    (R / "notes.md").write_text(notes, encoding="utf-8", newline="\n")
    print(json.dumps(jsonable(report), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    launch.begin(R, "summary", [sys.executable, *sys.argv], entry="runs/E50/code/summary.py")
    try:
        main()
    except BaseException:
        launch.end(R, "summary", "fail")
        raise
    launch.end(R, "summary", "ok")
