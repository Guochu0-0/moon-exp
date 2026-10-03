"""finetune 里不依赖 GPU / 底座权重的部分：配置的展开与校验、伪标签几何、扰动、标签读取、lr 调度、驱动列方法。"""
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _texture(n=128, shift=0, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 2.0)
    return np.roll(img, shift, axis=1)          # 内容右移 shift px：光学 x → SAR x + shift


def _grid(w=10, h=10, step=8):
    ii = np.arange(w * h)
    return np.stack([ii % w, ii // w], 1).astype(np.float64) * step


def _write(d, name, text):
    (d / f"{name}.toml").write_text(text, encoding="utf-8")
    return d / f"{name}.toml"


# ---------- 配置 ----------

def test_config_defaults_and_model_overrides(tmp_path):
    from finetune import config as C

    lo = C.load(_write(tmp_path, "a", '[model]\nname = "loftr"\n'))
    assert lo["optim"]["wd"] == 0.0 and lo["run"]["save_zero"] is True
    assert lo["model"]["config"] == "configs/baselines/anymatch_loftr.json" and C.parts_of(lo) == []
    ro = C.load(_write(tmp_path, "b", '[model]\nname = "roma"\n[cexp]\nr_out = -0.25\n'))
    assert ro["optim"]["wd"] == 0.01 and ro["run"]["save_zero"] is False and ro["model"]["train_vgg"] is False
    assert ro["cexp"]["cert_out"] == -0.25 and ro["cexp"]["num"] == 5000      # cert_out 默认同 r_out


def test_config_base_inheritance(tmp_path):
    from finetune import config as C

    _write(tmp_path, "_base", '[model]\nname = "loftr"\n[optim]\nsteps = 4000\nlr = 3e-5\n[cexp]\nr_out = -0.25\n[neg]\n')
    c = C.load(_write(tmp_path, "Q7", 'base = "_base"\n[optim]\nlr = 1e-5\n[aug]\nphoto = false\n'))
    assert c["optim"]["steps"] == 4000 and c["optim"]["lr"] == 1e-5
    assert C.parts_of(c) == ["cexp", "neg", "aug"] and c["aug"] == {**c["aug"], "geo": True, "photo": False}
    with pytest.raises(C.ConfigError):
        C.load(_write(tmp_path, "x", 'base = "../other/Q4"\n[model]\nname = "loftr"\n'))
    _write(tmp_path, "c1", 'base = "c2"\n')
    with pytest.raises(C.ConfigError):
        C.load(_write(tmp_path, "c2", 'base = "c1"\n'))


@pytest.mark.parametrize("text", [
    '[model]\nname = "sift"\n',                                       # 未知模型
    '[model]\nname = "loftr"\n[optim]\nlearning_rate = 1e-5\n',        # 未知键
    '[model]\nname = "loftr"\n[rl]\n',                                # 未知成分（细级 RL 不再搬）
    '[model]\nname = "loftr"\n[optim]\nsteps = "4000"\n',              # 类型不对
    '[model]\nname = "loftr"\n[cexp]\nw_cert = 0.0\n',                # RoMa 专有参数用在 LoFTR 上
    '[model]\nname = "loftr"\n[neg]\n',                               # 负样本对要配合打分成分
    '[model]\nname = "loftr"\n[pseudo]\n',                            # 伪标签必须给标签文件
    '[model]\nname = "loftr"\n[pseudo]\nlabels = "l.jsonl"\n[cexp]\n[neg]\n',   # LoFTR 上伪标签不配负样本对
    '[model]\nname = "loftr"\n[aug]\ngeo = false\nphoto = false\n',
])
def test_config_rejects(tmp_path, text):
    from finetune import config as C

    with pytest.raises(C.ConfigError):
        C.load(_write(tmp_path, "m", text))


def _from_old(a):
    """旧 finetune.train 的 args.json → 新格式里应有的值（只含旧参数能表达、且被搬过来的部分）。"""
    assert a["rl_pair"] == 0 and a["rl_match"] == 0 and a["w_l2sp"] == 0 and a["inject_shift"] == 0
    assert a["init"] is None and a.get("train_modules", "all") == "all" and a["coarse_set"] == "all"
    want = {"optim": {k: a[k] for k in ("steps", "batch", "lr", "wd", "warmup", "warmup_start", "sched", "lr_min",
                                        "clip", "accum")},
            "run": {"seed": a["seed"], "workers": a["workers"], "save_every": a["save_every"], "limit": a["limit"],
                    "save_at": [int(x) for x in a["save_at"].split(",") if x]}}
    if a["w_cexp"] > 0:
        want["cexp"] = {"weight": a["w_cexp"], "r_out": a["cexp_out"], "ransac": a["ransac"],
                        "placebo": a["cexp_placebo"]}
    if a["neg"]:
        want["neg"] = {}
    if a["labels"]:
        assert a["w_pseudo"] > 0
        want["pseudo"] = {"labels": a["labels"], "top": a["label_top"], "weight": a["w_pseudo"],
                          "w_coarse": a["w_coarse"], "w_fine": a["w_fine"]}
    aug = a["aug"].split(",") if a["aug"] else []
    if aug:
        want["aug"] = {"geo": "geo" in aug, "photo": "photo" in aug, "shift": a["aug_shift"], "rot": a["aug_rot"],
                       "scale": a["aug_scale"]}
    return want


@pytest.mark.parametrize("name,old", [("Q4", "runs/Q/extra/args_Q4.json"), ("P8", "runs/P/extra/args_P8.json")])
def test_reference_configs_match_old_args(name, old):
    """configs/finetune/ 里按新格式改写的旧方法，展开后与当年 args.json 的设置一致。"""
    from finetune import config as C

    a = json.loads((REPO / old).read_text(encoding="utf-8"))["args"]
    cfg = C.load(REPO / "configs" / "finetune" / f"{name}.toml")
    want = _from_old(a)
    assert C.parts_of(cfg) == [n for n in ("pseudo", "cexp", "neg", "aug") if n in want]
    for table, vals in want.items():
        assert {k: cfg[table][k] for k in vals} == vals, table


def test_reference_configs_all_expand():
    from finetune import config as C

    for p in sorted((REPO / "configs" / "finetune").glob("*.toml")):
        C.load(p)


def test_driver_lists_methods(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ft_run", REPO / "scripts" / "finetune" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    (tmp_path / "configs").mkdir()
    for n in ("_base", "Q7", "Q10", "P1"):
        _write(tmp_path / "configs", n, "")
    assert run.methods(tmp_path) == ["P1", "Q10", "Q7"]


# ---------- 训练成分与工具 ----------

def test_diagnose_reward_surface_peak():
    from baselines import diagnose_reward as dr

    Fo, Fs = dr.cfog(_texture(160)), dr.cfog(_texture(160, shift=3))
    A = lambda t: np.array([[1.0, 0, t], [0, 1.0, 0]])
    dx, dy, _ = dr.peak(dr.surface(dr.warp_feat(Fo, A(3), Fs.shape[1:]), Fs, "cfog"))
    assert abs(dx) < 0.3 and abs(dy) < 0.3                  # 基准已对准：峰在 0
    dx, dy, _ = dr.peak(dr.surface(dr.warp_feat(Fo, A(1), Fs.shape[1:]), Fs, "cfog"))
    assert abs(dx + 2) < 0.3 and abs(dy) < 0.3              # 基准差 2 px：峰在 d = −2，A' = A − d 对准


def test_pseudo_targets_translation():
    from finetune.parts.pseudo import coarse_targets, fine_targets

    s = 512 / 640
    A = np.array([[1.0, 0, 8 * s], [0, 1.0, 0]])          # 原网格右移 6.4 px = 输入网格 8 px = 一个粗格
    i, j = coarse_targets(A, (10, 10), 8, s)
    assert len(i) == 90 and np.all(j == i + 1)            # 最右一列落到图外，不监督
    k0 = _grid()[:5]
    A2 = np.array([[1.0, 0, 10 * s], [0, 1.0, 0]])        # 输入网格右移 10 px
    g = fine_targets(A2, k0, k0 + [8, 0], 4.0, s)          # 粗级落在 +8，窗口半宽 4 px
    assert np.allclose(g, [[0.5, 0]] * 5)


def test_pseudo_loss_loftr():
    import torch
    from finetune.parts.pseudo import loftr_loss

    s, w = 512 / 640, 10
    torch.manual_seed(0)
    logits = torch.randn(1, w * w, w * w, requires_grad=True)
    conf = torch.softmax(logits, 1) * torch.softmax(logits, 2)
    k0 = _grid()
    expec = torch.zeros(len(k0), 3)
    expec[:, 2] = 0.5
    expec.requires_grad_(True)
    ids = torch.arange(len(k0))
    data = {"conf_matrix": conf, "expec_f": expec, "hw0_c": (w, w), "hw0_i": (80, 80), "hw0_f": (40, 40), "W": 5,
            "b_ids": torch.zeros(len(k0), dtype=torch.long), "i_ids": ids, "j_ids": ids,
            "mkpts0_c": torch.from_numpy(k0), "mkpts1_c": torch.from_numpy(k0 + [8, 0])}
    A = np.array([[1.0, 0, 9 * s], [0, 1.0, 0]])          # 真实对应在 +9：细级偏移 0.25 窗口
    loss, st = loftr_loss(data, s, [A])
    loss.backward()
    assert st["pairs_used"] == 1 and st["n_fine"] == 100 and st["n_coarse_pos"] > 0
    assert logits.grad.abs().sum() > 0 and expec.grad.abs().sum() > 0
    assert expec.grad[:, 0].mean() < 0                      # 梯度下降把 x 偏移往 +0.25 推
    loss, st = loftr_loss(data, s, [None])
    assert st["pairs_used"] == 0 and float(loss) == 0


def test_roma_affine_norm_roundtrip():
    """原网格角点约定的仿射换到 RoMa 归一化坐标后，作用在归一化网格上与原网格上的结果一致。"""
    from finetune.geom import apply_affine
    from finetune.parts.pseudo import affine_norm

    A = np.array([[1.01, 0.02, 5.0], [-0.01, 0.99, -7.0]])
    p = np.array([[100.0, 300.0], [10.0, 500.0]])            # 原网格中心约定
    xn = (p + 0.5) / 256 - 1
    An = affine_norm(A)
    q = (xn @ An[:, :2].T + An[:, 2] + 1) * 256 - 0.5
    assert np.allclose(q, apply_affine(A, p))


def test_lr_schedule():
    from finetune.optim import lr_at

    assert lr_at(123, 8000, 1e-5) == 1e-5                              # const、无 warmup（S1 的配方）
    assert abs(lr_at(0, 8000, 1e-5, warmup=500) - 1e-6) < 1e-12        # warmup 起点 0.1·lr
    assert abs(lr_at(250, 8000, 1e-5, warmup=500) - 5.5e-6) < 1e-12
    assert lr_at(500, 8000, 1e-5, 500, sched="cosine") == 1e-5         # warmup 结束即峰值
    assert abs(lr_at(4250, 8000, 1e-5, 500, sched="cosine") - 5e-6) < 1e-12
    assert abs(lr_at(8000, 8000, 1e-5, 500, sched="cosine", lr_min=0.1) - 1e-6) < 1e-12


def test_augment_warp_matches_label_transform():
    """SAR 按已知 T warp 后，原来落在 A(p) 的内容应出现在 (T∘A)(p)（#53）。"""
    from finetune.augment import sample_T, warp
    from finetune.geom import apply_affine, compose

    A = np.array([[1.0, 0.02, 5.0], [-0.01, 1.0, -7.0]])
    p = np.array([[200.0, 300.0]])                                    # 光学点，中心约定
    q = np.rint(apply_affine(A, p)[0]).astype(int)                   # SAR 上的对应像素
    img = np.zeros((512, 512), np.float32)
    img[q[1] - 1:q[1] + 2, q[0] - 1:q[0] + 2] = 1.0
    T = sample_T(np.random.default_rng(0), shift=12, rot=3, scale=0.03)
    out = warp(img, T)
    ys, xs = np.nonzero(out > 0.1)
    got = np.array([(xs * out[ys, xs]).sum(), (ys * out[ys, xs]).sum()]) / out[ys, xs].sum()
    want = apply_affine(T, q[None].astype(float))[0]
    assert np.abs(got - want).max() < 0.3
    assert np.allclose(apply_affine(compose(T, A), p), apply_affine(T, apply_affine(A, p)))


def test_load_labels_top(tmp_path):
    from finetune.label import load_labels

    rows = [{"pair": f"R/{i}", "A": [[1, 0, 0], [0, 1, 0]], "n_match": 200, "n_inliers": i, "keep": i > 0}
            for i in range(5)]
    f = tmp_path / "l.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert sum(v is not None for v in load_labels(f).values()) == 4
    top = load_labels(f, 0.5)
    assert [k for k, v in top.items() if v is not None] == ["R/3", "R/4"]
