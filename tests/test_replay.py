"""旧配置重放：用当前的配置展开逻辑重新展开每个已跑训练的配置文件，与启动记录里存的完整配置逐项对比。

守的是「新行为只能通过新增参数开启，默认保持旧行为」：改了某个默认值、改了补算规则，旧实验的配置在今天展开出来
就和当年不一样，这里变红。纳入的是 `scripts/finetune/run.py` 写的启动记录（RECORDS.md「启动记录」）：
入口为训练驱动、记了 `config_file` 与 `args`；补记（backfilled）的不纳入。配置文件按记录里的 commit 取出，
这样事后改配置文件不算差异，只比较代码的展开逻辑。

历史上合法的差异逐条列在 ADDED 里，每条写明原因；其余任何差异（已有键的值变了、表多了或少了）都变红。
"""
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENTRY = "scripts/finetune/run.py"

# 历史上合法的差异：启动记录写下之后才新增的参数。旧记录里没有这个键，今天展开出它的默认值，而这个默认值
# 就是新增之前的行为。每条：键路径 → (默认值, 原因)。只在「记录里缺这个键、今天展开出的正好是这里写的默认值」时放行，
# 所以改了某个参数的默认值照样变红。驱动每次都写展开后的完整配置，记录里缺键本身就说明它早于这个参数。
# 以后新增参数时在这里加一条，默认值必须等于新增之前的行为。
ADDED = {
    "run.select": ("val_peak", "#98 新增；之前驱动一律按 Val AUC@5 峰值选 ckpt"),
    "optim.ema": (0.0, "#120 新增权重 EMA；0 = 不用"),
    "model.train_decoder": (True, "#120 新增（RoMa）；之前总是训 decoder"),
    "model.vgg_lr": (1.0, "#120 新增（RoMa）；VGG 学习率 = 1 倍 [optim] lr，即之前所有参数同一学习率"),
    "model.vgg_dropout": (0.0, "#120 新增（RoMa）；0 = 不做 VGG 特征 dropout"),
    "pseudo.cls_soft": (False, "#120 新增（RoMa）；false = 粗级锚点一热目标，即原损失"),
    "pseudo.drop_top": (0.0, "#120 新增（RoMa）；0 = 不去掉残差最大的像素"),
    "model.train_res": ("560", "#122 新增（RoMa）；560 = 只训 560 一遍，即之前的训法"),
    "model.lora_r": (0, "#122 新增（RoMa）；0 = DINOv2 冻结、不挂 LoRA，即之前的做法"),
    "model.lora_alpha": (16.0, "#122 与 lora_r 同批新增；只在 lora_r > 0 时起作用"),
    "model.lora_lr": (1e-4, "#122 与 lora_r 同批新增；只在 lora_r > 0 时起作用"),
    "model.dino_ckpt": (False, "#122 新增（RoMa）；梯度检查点只影响显存，不影响结果"),
    "pseudo.dense": ("", "#122 新增（RoMa）；空 = 只用伪仿射，即之前的目标"),
    "pseudo.dense_r": (8.0, "#122 与 dense 同批新增；只在 dense 非空时起作用"),
    "pseudo.dense_cert": (0.5, "#122 与 dense 同批新增；只在 dense 非空时起作用"),
    "model.lora_targets": ("qkv", "#124 新增（RoMa）；qkv = 只挂 attn.qkv，即 #122 时 LoRA 的挂法"),
    "model.dino_unfreeze": (0, "#124 新增（RoMa）；0 = 不解冻 DINOv2 的块"),
    "aug.blockmask": (False, "#127 新增；false = 不做分块遮挡"),
    "aug.mask_block": (32, "#127 与 blockmask 同批新增；只在 blockmask 打开时起作用"),
    "aug.mask_ratio": (0.3, "#127 与 blockmask 同批新增；只在 blockmask 打开时起作用"),
    "pseudo.pair_weight": ("", "#127 新增；空 = 各对不加权"),
    "pseudo.reg_tol": (0.0, "#127 新增（RoMa）；0 = 回归误差不减容差，即原损失"),
    "pseudo.fine_ramp": ([], "#127 新增（RoMa）；空 = 不用尺度课程"),
    "model.unc_head": (False, "#127 新增（RoMa）；false = 不接不确定度头"),
    "pseudo.match_dir": ("", "#127 新增（RoMa）；空 = certainty 照旧学「是否在重叠区」"),
    "pseudo.match_r": (3.0, "#127 与 match_dir 同批新增；只在 match_dir 非空时起作用"),
    "pseudo.reg_loss": ("robust", "#127 新增（RoMa）；robust = RoMa 原鲁棒回归项"),
    "pseudo.mixlap_w": (1.0, "#127 与 reg_loss 同批新增；只在 reg_loss = mixlap 时起作用"),
    "pseudo.ema_target": (0.0, "#127 新增（RoMa）；0 = 回归目标不向 EMA 老师过渡"),
    "pseudo.ema_alpha": (0.5, "#127 与 ema_target 同批新增；只在 ema_target > 0 时起作用"),
    "pseudo.ema_ramp": ([2000, 4000], "#127 与 ema_target 同批新增；只在 ema_target > 0 时起作用"),
    "model.lora_dropout": (0.0, "#130 新增（RoMa）；0 = LoRA 支路不加 dropout"),
    "model.drop_path": (0.0, "#130 新增（RoMa）；0 = DINOv2 不加 stochastic depth"),
    "aug.shift_disk": (False, "#157 新增；false = 平移照旧在方形范围内均匀取"),
    "pseudo.mask_fill": (False, "#157 新增；false = 补零区照常监督"),
}


def allowed(k, old_new) -> bool:
    return k in ADDED and old_new[0] == "<缺>" and old_new[1] == ADDED[k][0]


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def records():
    """所有纳入重放的启动记录：(记录文件, 序号, 记录)。"""
    out = []
    for f in sorted((REPO / "runs").glob("*/launch/*.json")):
        for i, r in enumerate(json.loads(f.read_text(encoding="utf-8"))):
            if r.get("entry") == ENTRY and r.get("config_file") and r.get("args") and not r.get("backfilled"):
                out.append((f.relative_to(REPO).as_posix(), i, r))
    return out


def _configs_at(commit: str, config_file: str, dst: Path) -> Path:
    """把 commit 时 config_file 所在目录的全部 .toml 写到 dst（base 只在同目录内继承）。"""
    d = config_file.rsplit("/", 1)[0]
    for p in _git("ls-tree", "--name-only", commit, "--", d + "/").split():
        if p.endswith(".toml"):
            (dst / p.rsplit("/", 1)[1]).write_text(_git("show", f"{commit}:{p}"), encoding="utf-8")
    return dst / config_file.rsplit("/", 1)[1]


def _flat(cfg: dict) -> dict:
    return {f"{t}.{k}": v for t, tab in cfg.items() for k, v in tab.items()}


def diff(old: dict, new: dict) -> dict:
    """逐项差异：键路径 → (记录里的值, 今天展开的值)；缺失写 "<缺>"。整张表多出或缺失按表报一条。"""
    out = {}
    for t in sorted(set(old) | set(new)):
        if t not in old or t not in new:
            out[t] = ("<缺>" if t not in old else "<有>", "<缺>" if t not in new else "<有>")
    a = _flat({t: v for t, v in old.items() if t in new})
    b = _flat({t: v for t, v in new.items() if t in old})
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<缺>"), b.get(k, "<缺>")
        if va != vb:
            out[k] = (va, vb)
    return out


RECORDS = records()


def test_replay_has_records():
    assert len(RECORDS) >= 80          # 2026-10-20 时有 89 条；变少说明筛选条件或记录格式坏了


@pytest.mark.parametrize("src,i,rec", RECORDS, ids=[f"{s}#{i}" for s, i, _ in RECORDS])
def test_replay_old_config(src, i, rec, tmp_path):
    from finetune import config as C

    cfg = C.load(_configs_at(rec["commit"], rec["config_file"], tmp_path))
    d = diff(rec["args"], cfg)
    bad = {k: v for k, v in d.items() if not allowed(k, v)}
    assert not bad, f"{rec['config_file']}（commit {rec['commit'][:9]}）今天展开与启动记录不同：{bad}"


def test_added_entries_all_used(tmp_path):
    """白名单的每条都至少放行了一处差异；用不上的（写错了键、或不再需要）删掉，不留空条目。"""
    from finetune import config as C

    used = set()
    for n, (_, _, rec) in enumerate(RECORDS):
        d = tmp_path / str(n)
        d.mkdir()
        cfg = C.load(_configs_at(rec["commit"], rec["config_file"], d))
        used |= {k for k, v in diff(rec["args"], cfg).items() if allowed(k, v)}
    assert not set(ADDED) - used, sorted(set(ADDED) - used)


def test_allowed_only_missing_key_with_listed_default():
    assert allowed("pseudo.mask_fill", ("<缺>", False))
    assert not allowed("pseudo.mask_fill", ("<缺>", True))         # 默认值被改了
    assert not allowed("pseudo.mask_fill", (False, True))          # 记录里有这个键，值变了
    assert not allowed("optim.lr", ("<缺>", 1e-5))                 # 不在白名单里的新增键


def test_diff_reports_changed_missing_and_extra():
    old = {"optim": {"lr": 1e-5, "wd": 0.0}, "neg": {}}
    new = {"optim": {"lr": 3e-5, "wd": 0.0, "ema": 0.0}, "aug": {"geo": True}}
    assert diff(old, new) == {"aug": ("<缺>", "<有>"), "neg": ("<有>", "<缺>"),
                              "optim.lr": (1e-5, 3e-5), "optim.ema": ("<缺>", 0.0)}
