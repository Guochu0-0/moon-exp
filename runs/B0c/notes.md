#70「改进来自哪一级」的粗级评估。B0c、S1c、Cc、Qc、Pc 五个实验由同一组脚本产出，脚本放在本实验的 `code/`；分析见 #70。

## 记录来源（事后重建，#77）

用 `git archive` 副本 `YGC/moon-exp-cf70` 跑，代码与 53d9bc8 一致。`code/` 里的 `common.sh`（ckpt 与配置）、`run.sh`（粗级匹配）、`fit.sh`（建五个实验、按 RANSAC 3 / 8 px 拟合）、`smoke.sh` 当时只在副本里，未进过 git，原样提交。
