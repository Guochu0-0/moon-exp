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


# ---------- #120 第一批新增的训练选项 ----------

def test_soft_anchor_target():
    import torch
    from finetune.parts.pseudo import grid, soft_anchor_target

    r = 8
    G = grid(r, r, "cpu")                                   # 锚点中心，G[y, x]
    t = soft_anchor_target(G[3, 5][None, None, None], r)    # 正好落在第 (y=3, x=5) 个锚点上
    assert t.shape == (1, r * r, 1, 1) and torch.isclose(t[0, 3 * r + 5, 0, 0], torch.tensor(1.0))
    mid = ((G[3, 5] + G[4, 6]) / 2)[None, None, None]       # 四个锚点正中：各 1/4
    t = soft_anchor_target(mid, r)[0, :, 0, 0]
    assert torch.allclose(t[[3 * r + 5, 3 * r + 6, 4 * r + 5, 4 * r + 6]], torch.full((4,), 0.25))
    edge = torch.tensor([[[[0.999, -0.999]]]])              # 网格外缘：并到最近的锚点，总和仍为 1
    t = soft_anchor_target(edge, r)[0, :, 0, 0]
    assert torch.isclose(t.sum(), torch.tensor(1.0)) and torch.isclose(t[r - 1], torch.tensor(1.0))


def test_drop_largest():
    import torch
    from finetune.parts.pseudo import drop_largest

    epe = torch.arange(20.0).reshape(1, 4, 5)
    m = torch.ones(1, 4, 5, dtype=torch.bool)
    m[0, 0, 0] = False
    out = drop_largest(epe, m, 0.1)                         # 19 个像素去掉最大的 1 个
    assert int(out.sum()) == 18 and not out[0, 3, 4] and not out[0, 0, 0]


def test_occluded_follows_label_and_swap():
    import torch
    from finetune.parts.pseudo import affine_norm, occluded

    mask = torch.zeros(1, 1, 512, 512)
    mask[..., :, 256:] = 1                                  # SAR 右半被遮
    An = torch.from_numpy(affine_norm(np.c_[np.eye(2), [-256.0, 0]])[None]).float()   # 光学 x → SAR x − 256
    assert not occluded((mask, torch.tensor([False])), An, 4, 4)[0].any()   # 光学点都落到 SAR 左半或图外
    o = occluded((mask, torch.tensor([True])), An, 4, 4)[0]                 # 交换后查询图就是 SAR：右半被遮
    assert o[:, 2:].all() and not o[:, :2].any()


def test_roma_loss_new_options_change_loss():
    import torch
    from finetune.parts.pseudo import roma_loss

    An = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]])
    flow = torch.randn(1, 2, 8, 8, requires_grad=True)
    cert = torch.zeros(1, 1, 8, 8, requires_grad=True)
    cls = torch.randn(1, 16, 8, 8, requires_grad=True)
    c = lambda: {16: {"flow": flow, "certainty": cert, "gm_cls": cls, "gm_certainty": cert}}
    l0, _ = roma_loss(c(), An)
    for kw in ({"cls_soft": True}, {"drop_top": 0.1}, {"occ": (torch.ones(1, 1, 512, 512), torch.tensor([False]))}):
        l1, _ = roma_loss(c(), An, **kw)
        assert torch.isfinite(l1) and float(l1) != float(l0), kw


def test_erase_area():
    from finetune import augment

    rng = np.random.default_rng(0)
    img = rng.random((512, 512)).astype(np.float32)
    for _ in range(20):
        out, m = augment.erase(img, rng)
        assert 0.0 < m.mean() <= 0.25 + 1e-3
        assert np.allclose(out[m > 0], img.mean()) and np.array_equal(out[m == 0], img[m == 0])


def test_optim_lr_scale_groups():
    import torch
    from finetune import config as C
    from finetune.optim import Optim

    a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    o = {**C.OPTIM, "warmup": 0, "sched": "const", "lr": 1e-3}
    opt = Optim([a, b], o, amp=False, groups=[{"params": [a], "lr_scale": 1.0}, {"params": [b], "lr_scale": 0.1}])
    (a + b).sum().backward()
    opt.step(1)
    assert [g["lr"] for g in opt.opt.param_groups] == pytest.approx([1e-3, 1e-4])


def test_config_new_options(tmp_path):
    from finetune import config as C

    c = C.load(_write(tmp_path, "t", '[model]\nname = "roma"\ntrain_vgg = true\nvgg_lr = 3.0\n[optim]\nema = 0.999\n'
                                     '[pseudo]\nlabels = "l.jsonl"\ncls_soft = true\ndrop_top = 0.1\n[aug]\ngeo = false\n'
                                     'photo = false\nerase = true\n'))
    assert c["model"]["vgg_lr"] == 3.0 and c["optim"]["ema"] == 0.999 and c["pseudo"]["cls_soft"] is True
    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n'))
    assert d["optim"]["ema"] == 0.0 and d["model"]["vgg_dropout"] == 0.0 and d["model"]["train_decoder"] is True
    with pytest.raises(C.ConfigError):                      # 擦除只配合 RoMa 的伪标签
        C.load(_write(tmp_path, "e", '[model]\nname = "roma"\n[aug]\nerase = true\n'))


def test_merge_average_and_wise():
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location("merge", REPO / "scripts/finetune/merge.py")
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    a = {"w": torch.zeros(2), "n": torch.tensor(3)}
    b = {"w": torch.ones(2), "n": torch.tensor(5)}
    avg = mg.average([a, b])
    assert torch.allclose(avg["w"], torch.full((2,), 0.5)) and int(avg["n"]) == 3
    zero = {"w": torch.zeros(2), "dino": torch.ones(1)}
    out, extra = mg.wise(zero, {"w": torch.full((2,), 4.0), "x": torch.ones(1)}, 0.25)
    assert torch.allclose(out["w"], torch.ones(2)) and extra == ["x"] and "dino" not in out


# ---------- #127 第四批新增的训练选项 ----------

def test_pair_weights_inlier_ratio(tmp_path):
    from finetune.label import pair_weights

    rows = [{"pair": f"R/{i}", "A": [[1, 0, 0], [0, 1, 0]], "n_match": 100, "n_inliers": 20 * (i + 1), "keep": True}
            for i in range(3)]
    f = tmp_path / "l.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    w = pair_weights(f, ["R/0", "R/2"])                     # 内点率 0.2、0.6，只在给定的对上归一化
    assert w == pytest.approx({"R/0": 0.5, "R/2": 1.5})


def test_roma_loss_tol_and_fine_w():
    import torch
    from finetune.parts.pseudo import affine_gt, roma_loss

    An = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]])
    x2, _ = affine_gt(An, 8, 8)
    small = (x2 + 0.5 * 2 / 512).permute(0, 3, 1, 2)       # 处处偏 (0.5, 0.5) 原网格 px，EPE 约 0.71 px
    cert = torch.zeros(1, 1, 8, 8)
    c = lambda f: {16: {"flow": f, "certainty": cert}, 1: {"flow": f, "certainty": cert}}
    l0, _ = roma_loss(c(small), An)
    l_tol, _ = roma_loss(c(small), An, reg_tol=1.0 * 2 / 512)   # 1 层误差 < 1 px 不罚
    l_ex, _ = roma_loss({16: c(small)[16], 1: c(x2.permute(0, 3, 1, 2))[1]}, An)
    assert float(l_tol) < float(l0) and float(l_tol) == pytest.approx(float(l_ex), rel=1e-5)
    l_off, _ = roma_loss(c(small), An, fine_w=0.0)          # 只剩 16 层
    l_16, _ = roma_loss({16: c(small)[16]}, An)
    assert float(l_off) == pytest.approx(float(l_16))


def test_blockmask_ratio_and_config(tmp_path):
    from finetune import augment
    from finetune import config as C

    rng = np.random.default_rng(0)
    img = rng.random((512, 512)).astype(np.float32)
    fr = np.mean([np.mean(augment.blockmask(img, rng, 32, 0.3) != img) for _ in range(50)])
    assert 0.25 < fr < 0.35
    c = C.load(_write(tmp_path, "m", '[model]\nname = "roma"\n[pseudo]\nlabels = "l.jsonl"\npair_weight = "inlier_ratio"\n'
                                     'reg_tol = 1.0\nfine_ramp = [2000, 3000]\n[aug]\ngeo = false\nphoto = false\n'
                                     'blockmask = true\n'))
    assert c["aug"]["mask_ratio"] == 0.3 and c["pseudo"]["fine_ramp"] == [2000, 3000]
    with pytest.raises(C.ConfigError):
        C.load(_write(tmp_path, "b", '[model]\nname = "roma"\n[pseudo]\nlabels = "l.jsonl"\nfine_ramp = [3000, 2000]\n'))
    with pytest.raises(C.ConfigError):                      # mixlap 要不确定度头
        C.load(_write(tmp_path, "x", '[model]\nname = "roma"\n[pseudo]\nlabels = "l.jsonl"\nreg_loss = "mixlap"\n'))
    C.load(_write(tmp_path, "y", '[model]\nname = "roma"\nunc_head = true\n[pseudo]\nlabels = "l.jsonl"\n'
                                 'reg_loss = "mixlap"\nmatch_dir = "t"\n'))


def test_matchable_mask():
    import torch
    from finetune.parts.pseudo import affine_gt, grid, matchable

    An = torch.tensor([[[1.0, 0, 0.1], [0, 1.0, 0]]])      # 光学 → SAR 右移 0.1（归一化）
    S = 16
    x2, _ = affine_gt(An, S, S)
    g = grid(S, S, "cpu")[None]
    fwd = torch.cat([x2.permute(0, 3, 1, 2), torch.ones(1, 1, S, S)], 1)
    bwd = torch.cat([(g - torch.tensor([0.1, 0.0])).permute(0, 3, 1, 2), torch.ones(1, 1, S, S)], 1)
    mk = matchable((fwd, bwd, 3 * 2 / 512), x2, S, S)[0]
    assert mk[:, : S - 2].all()                             # 往返一致、与伪仿射一致；右缘落到图外的走不回来
    bad = fwd.clone()
    bad[:, 0, :, : S // 2] += 0.2                           # 左半老师与伪仿射不一致
    mk = matchable((bad, bwd, 3 * 2 / 512), x2, S, S)[0]
    assert not mk[:, : S // 2].any()


def test_roma_loss_ema_target_mix():
    import torch
    from finetune.parts.pseudo import affine_gt, roma_loss

    An = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]])
    x2, prob = affine_gt(An, 8, 8)
    flow = torch.randn(1, 2, 8, 8) * 0.1
    t = (x2 + 0.05).permute(0, 3, 1, 2)                     # 老师落点
    cert = torch.zeros(1, 1, 8, 8)
    c = {1: {"flow": flow, "certainty": cert}}
    mixed = (0.5 * x2 + 0.5 * (x2 + 0.05), prob)
    l_ema, _ = roma_loss(c, An, ema=({1: t}, 0.5))
    l_gt, _ = roma_loss(c, An, gt=lambda h, w: mixed)
    assert float(l_ema) == pytest.approx(float(l_gt))


def test_mixlap_nll_prefers_wide_component_for_outliers():
    import torch
    from finetune.parts.pseudo import affine_gt, mixlap_nll

    An = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]])
    x2, _ = affine_gt(An, 4, 4)
    flow = (x2 + 20 * 2 / 512).permute(0, 3, 1, 2)          # 处处偏 20 px
    m = torch.ones(1, 4, 4, dtype=torch.bool)
    narrow = torch.zeros(1, 2, 4, 4)
    wide = torch.stack([torch.full((1, 4, 4), -5.0), torch.full((1, 4, 4), 3.0)], 1)   # 几乎全用 e^3 px 的分量
    assert float(mixlap_nll(flow, x2, wide, m)) < float(mixlap_nll(flow, x2, narrow, m))


# ---------- #157 难版范围的训练扰动 ----------

def test_sample_T_disk():
    from finetune.augment import sample_T

    T = [sample_T(np.random.default_rng(k), shift=128, rot=0, scale=0, disk=True) for k in range(4000)]
    r = np.array([np.linalg.norm(x[:, 2]) for x in T])      # rot = scale = 0 时 L = I，第三列就是平移
    assert r.max() <= 128 and abs(np.mean(r < 64) - 0.25) < 0.03   # 圆盘内按面积均匀：半径一半以内占 1/4
    a = sample_T(np.random.default_rng(3), shift=12, rot=3, scale=0.03)    # 默认仍是方形，随机数用法不变
    g = np.random.default_rng(3)
    g.uniform(-3, 3), g.uniform(-0.03, 0.03)
    c = np.array([256.0, 256.0])
    assert np.allclose(a[:, 2] - (c - a[:, :2] @ c), g.uniform(-12, 12, 2))


def test_mask_fill_loftr():
    """A 把光学右移 60 px、T 再左移回来：T∘A 是恒等，所有格都落在扰动后的画面内；
    但最右一列经 A 落到原 SAR 画面外（补零区），mask_fill 时粗级、细级都不监督（#157）。"""
    import torch
    from finetune.parts.pseudo import coarse_targets, loftr_loss

    s, w = 512 / 640, 10                                    # 输入网格 640，粗格 64 px，格点 0…576
    A0 = np.array([[1.0, 0, 60.0], [0, 1.0, 0]])
    TA = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert len(coarse_targets(TA, (w, w), 64, s)[0]) == 100
    i, _ = coarse_targets(TA, (w, w), 64, s, A0)
    assert len(i) == 90 and not np.any(i % w == w - 1)
    k0 = _grid(step=64)
    data = {"conf_matrix": torch.full((1, 100, 100), 0.5), "expec_f": torch.full((100, 3), 0.5),
            "hw0_c": (w, w), "hw0_i": (640, 640), "hw0_f": (80, 80), "W": 5,
            "b_ids": torch.zeros(100, dtype=torch.long), "mkpts0_c": torch.from_numpy(k0),
            "mkpts1_c": torch.from_numpy(k0)}
    _, st = loftr_loss(data, s, [TA])
    _, st_m = loftr_loss(data, s, [TA], src=[A0])
    assert (st["n_coarse_pos"], st["n_fine"]) == (100, 100) and (st_m["n_coarse_pos"], st_m["n_fine"]) == (90, 90)


def test_mask_fill_roma():
    """RoMa：src 把查询像素映到原 SAR 画面外的部分 prob = 0，损失随之变化；src 为恒等时与不加相同（#157）。"""
    import torch
    from finetune.parts.pseudo import affine_norm, roma_loss

    An = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]])
    c = {16: {"flow": torch.randn(1, 2, 8, 8), "certainty": torch.zeros(1, 1, 8, 8)}}
    l0, _ = roma_loss(c, An)
    l_id, _ = roma_loss(c, An, src=An.clone())
    half = torch.from_numpy(affine_norm(np.c_[np.eye(2), [256.0, 0]])[None]).float()   # 右半经它落到原画面外
    l_half, _ = roma_loss(c, An, src=half)
    assert torch.isclose(l0, l_id) and not torch.isclose(l0, l_half)


def test_config_hard_aug_options(tmp_path):
    from finetune import config as C

    c = C.load(_write(tmp_path, "t", '[model]\nname = "loftr"\n[pseudo]\nlabels = "l.jsonl"\nmask_fill = true\n'
                                     '[aug]\nshift = 128.0\nrot = 20.0\nscale = 0.2\nshift_disk = true\n'))
    assert c["pseudo"]["mask_fill"] is True and c["aug"]["shift_disk"] is True
    d = C.load(_write(tmp_path, "d", '[model]\nname = "loftr"\n[pseudo]\nlabels = "l.jsonl"\n[aug]\n'))
    assert d["pseudo"]["mask_fill"] is False and d["aug"]["shift_disk"] is False


def test_rloo_ref_cells_graded_reward():
    """ref：粗 warp 与参照仿射相距 d = 0 / 8 / 16 / 24 px 时，分级奖励为 1、0.5、−1（阈值内侧照 RIPE++ 不修）、r_out（#183）。"""
    import torch
    from finetune.parts.pseudo import grid
    from finetune.parts.rloo import ref_cells

    A = np.array([[1.0, 0, 16], [0, 1.0, 0]])                     # 参照：右移 16 px
    g = grid(4, 4, "cpu").permute(2, 0, 1)                        # (2,h,w)
    off = torch.tensor([16.0, 24.0, 32.0, 40.0])                  # 粗 warp 右移的 px → d = 0、8、16、24
    fc = g[None].repeat(4, 1, 1, 1)
    fc[:, 0] += (off / 256)[:, None, None]
    r, d = ref_cells(fc, A, delta=8.0, tau=16.0, r_out=-0.1)
    assert torch.allclose(d.mean((1, 2)), off - 16, atol=1e-3)
    assert torch.allclose(r[:, 0, 0], torch.tensor([1.0, 0.5, -1.0, -0.1]), atol=1e-4)
    r0, d0 = ref_cells(fc, None, 8.0, 16.0, -0.1)
    assert d0 is None and (r0 == 0).all()


def test_rloo_distance_reward_shapes():
    """奖励形式（#191）：ripe 照 RIPE++ 在 tau 处跳回 r_out；clip 单调连续、到 r_out 截平；gauss 为 exp(−½(d/δ)²)；
    长度参数同比缩放时奖励只看 d/δ（供细化按级复用）。"""
    import torch
    from finetune.parts.rloo import distance_reward

    d = torch.tensor([0.0, 8.0, 12.0, 16.0, 16.5, 24.0])
    ripe = distance_reward(d, "ripe", 8.0, 16.0, -0.1)
    assert torch.allclose(ripe, torch.tensor([1.0, 0.5, -0.125, -1.0, -0.1, -0.1]))
    assert torch.equal(distance_reward(d), ripe)                                   # 默认即 E15 原式
    clip = distance_reward(d, "clip", 8.0, 16.0, -0.1)
    assert torch.allclose(clip, torch.tensor([1.0, 0.5, -0.1, -0.1, -0.1, -0.1]))
    gauss = distance_reward(d, "gauss", 8.0, 16.0, -0.1)
    assert torch.allclose(gauss, torch.exp(-0.5 * (d / 8) ** 2))
    fine = torch.linspace(0, 64, 641)
    for shape in ("clip", "gauss"):                                                # 单调不增、无跳变
        r = distance_reward(fine, shape, 8.0, 16.0, -0.1)
        assert (r[1:] <= r[:-1] + 1e-7).all() and (r[:-1] - r[1:]).max() < 0.02
    assert (ripe[4] - ripe[3]) > 0.8                                               # 原式在 tau 处向上跳
    for shape in ("ripe", "clip", "gauss"):                                        # 尺度 s：delta、tau 同乘 s/8
        assert torch.allclose(distance_reward(d / 4, shape, 2.0, 4.0, -0.1), distance_reward(d, shape, 8.0, 16.0, -0.1))
    with pytest.raises(ValueError):
        distance_reward(d, "box")


def test_rloo_ref_cells_shape():
    import torch
    from finetune.parts.pseudo import grid
    from finetune.parts.rloo import distance_reward, ref_cells

    A = np.array([[1.0, 0, 16], [0, 1.0, 0]])
    fc = grid(4, 4, "cpu").permute(2, 0, 1)[None].repeat(2, 1, 1, 1)
    fc[1, 0] += 20 / 256
    r, d = ref_cells(fc, A, 4.0, 8.0, -0.1, shape="gauss")
    assert torch.allclose(r, distance_reward(d, "gauss", 4.0)) and r.min() < r.max() <= 1


def test_rloo_in_frame():
    """画面外的格子：自估仿射把格子中心映射到 SAR 画面外、或补零区时为 False。"""
    import torch
    from finetune.parts.rloo import in_frame

    sar = torch.ones(1, 64, 64)
    I = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert in_frame(I, sar, 4, 4).all()
    m = in_frame(np.array([[1.0, 0, 256], [0, 1.0, 0]]), sar, 4, 4)     # 右移半幅：右半落到画面外
    assert m[:, :2].all() and not m[:, 2:].any()
    sar[:, :, :16] = 0                                                 # 左边 1/4 补零
    m = in_frame(I, sar, 4, 4)
    assert not m[:, 0].any() and m[:, 1:].all()


def test_config_rloo_frame_mask(tmp_path):
    from finetune import config as C
    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\n'))
    assert d["rloo"]["frame_mask"] is False and d["rloo"]["cert_bce"] == 0.0      # 默认同旧行为
    c = C.load(_write(tmp_path, "c", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nframe_mask = true\ncert_bce = 0.01\n'))
    assert c["rloo"]["frame_mask"] is True and c["rloo"]["cert_bce"] == 0.01
    for bad in ('reward = "r1"\nframe_mask = true', 'reward = "ref"\ncert_bce = -1.0',
                'reward = "ref"\nframe_mask = true\nfrozen_ref = true'):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", f'[model]\nname = "roma"\n[rloo]\n{bad}\n'))


def _blobs(pts, H, peaks, sig=4.0):
    """原网格（中心约定）上若干点处的高斯斑（宽 sig 原网格 px），画在 H×H 的网络输入网格上 → (1,H,H) 张量。"""
    import torch
    from finetune.geom import HW, orig_to_in
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float64)
    img = np.zeros((H, H))
    for (x, y), a in zip(orig_to_in(np.asarray(pts, np.float64), HW / H), peaks):
        img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (sig * H / HW) ** 2))
    return torch.from_numpy(img.astype(np.float32))[None]


def _blob_estimator(opt_pts, peaks):
    """只看图像的估计器：找出各高斯斑（按峰值认出是哪一个）、取亮度加权质心，换回原网格，与光学上的已知位置拟合仿射。"""
    import cv2
    from finetune.geom import HW, in_to_orig

    def est(x):
        a = x[0].numpy().astype(np.float64)
        H = a.shape[0]
        n, lab = cv2.connectedComponents((a > 0.05).astype(np.uint8))
        found = {}
        for i in range(1, n):
            m = lab == i
            k = int(np.argmin([abs(a[m].max() - pk) for pk in peaks]))
            ys, xs = np.nonzero(m)
            wgt = a[m]
            found[k] = in_to_orig(np.array([(xs * wgt).sum(), (ys * wgt).sum()]) / wgt.sum(), HW / H)
        if len(found) != len(peaks):
            return None
        P = np.array([found[k] for k in range(len(peaks))]) + 0.5        # 角点约定
        Q = np.asarray(opt_pts, np.float64) + 0.5
        return np.linalg.lstsq(np.c_[Q, np.ones(len(Q))], P, rcond=None)[0].T
    return est


def test_rloo_perturb_sar_direction():
    """额外扰动（#217）：在输入网格（560）上施加原网格的 T 后，原 SAR 上 p 处的内容出现在 T(p) 处（与训练扰动同一约定）；图外补 0。"""
    import torch
    import torch.nn.functional as F
    from finetune.geom import HW, apply_affine, orig_to_in
    from finetune.parts.rloo import perturb_sar

    H = 560
    rng = np.random.default_rng(0)
    img = torch.from_numpy(rng.random((1, H, H)).astype(np.float32))
    img = F.avg_pool2d(F.avg_pool2d(img[None], 9, 1, 4), 9, 1, 4)[0]       # 平滑，便于插值比较
    th = np.deg2rad(8.0)
    L = 1.07 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.array([HW / 2, HW / 2])
    T = np.c_[L, c - L @ c + [30.0, -45.0]]
    out = perturb_sar(img, T)
    p = rng.uniform(150, 360, (200, 2))                                     # 原网格中心约定，映射后仍在图内
    q = apply_affine(T, p)

    def at(im, pts):
        xy = (orig_to_in(pts, HW / H) + 0.5) / H * 2 - 1
        g = torch.from_numpy(xy).float()[None, None]
        return F.grid_sample(im[None], g, align_corners=False)[0, 0, 0].numpy()
    assert np.abs(at(out, q) - at(img, p)).max() < 0.01
    assert np.abs(at(out, q) - at(img, q)).mean() > 0.02                    # 不是原样返回
    assert float((out == 0).float().mean()) > 0.02 and float((img == 0).float().mean()) == 0.0   # 图外补 0


def test_rloo_consensus():
    """参照取逐格中位数再拟合；距离取最大的一对；有拟合失败时 (None, inf)。"""
    from finetune.parts.rloo import affine_denorm, consensus
    from finetune.parts.pseudo import affine_norm

    A = np.array([[1.02, 0.05, 10.0], [-0.03, 0.98, -7.0]])
    assert np.allclose(affine_denorm(affine_norm(A)), A)
    B = A.copy()
    B[0, 2] += 20.0                                                         # 右移 20 px
    ref, gap = consensus([A, A, B], 40, 40)
    assert np.allclose(ref, A, atol=1e-6) and abs(gap - 20.0) < 1e-3
    ref, gap = consensus([A, A, A], 40, 40)
    assert np.allclose(ref, A, atol=1e-6) and gap < 1e-6
    assert consensus([A, None, A], 40, 40) == (None, float("inf"))


def test_rloo_multi_ref_recovers_and_rejects():
    """对同一张图施加已知 T、估计后换算回来，与原估计一致（#217）；与图像无关、固定在一处的估计被拦。"""
    from finetune.augment import sample_T
    from finetune.geom import HW, apply_affine
    from finetune.parts.rloo import multi_ref

    H = 560
    A = np.array([[0.95, 0.08, 20.0], [-0.06, 1.03, -12.0]])                # 光学 → SAR（角点约定）
    opt = np.array([[200.0, 210.0], [300.0, 190.0], [250.0, 320.0]])
    peaks = (0.3, 0.6, 0.9)
    sar = _blobs(apply_affine(A, opt), H, peaks)
    est = _blob_estimator(opt, peaks)
    A0 = est(sar)
    assert np.abs(A0[:, :2] - A[:, :2]).max() < 0.01 and np.abs(A0[:, 2] - A[:, 2]).max() < 1.0   # 估计器本身可用
    rng = np.random.default_rng(1)
    Ts = [sample_T(rng, 64.0, 10.0, 0.1, HW, disk=True) for _ in range(2)]
    ref, gap, As = multi_ref(est, sar, A0, Ts, 40, 40)
    assert gap < 0.5 and all(a is not None for a in As)
    assert np.abs(ref - A0).max() < 0.5
    c = np.array([HW / 2, HW / 2])
    th = np.deg2rad(8.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Ts = [np.c_[np.eye(2), [40.0, 0.0]], np.c_[R, c - R @ c]]               # 平移 40 px；绕中心转 8°
    ref, gap, _ = multi_ref(lambda x: A, sar, A, Ts, 40, 40)                # 不看图像、总给同一个仿射
    assert gap > 16.0
    _, gap, _ = multi_ref(lambda x: None, sar, A0, Ts, 40, 40)
    assert gap == float("inf")


def _fake_term(monkeypatch, **over):
    """rloo.Term 的整条路径，模型换成假的（#217 review）：粗级 logits = w × 固定随机数（4×4 格、16 个锚点），细化输出全 0，
    拟合总给恒等仿射（与图像无关），额外扰动固定为平移 40 px。返回 (Term, 一步的输入, w)。"""
    import types
    import torch
    from finetune.parts import rloo as R

    base = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(0))

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(()))

    I = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    monkeypatch.setattr(R, "features", lambda model, a, b: ({16: a}, {16: b}))
    monkeypatch.setattr(R, "coarse", lambda model, f1, f2: (model.model.w * base.expand(len(f1[16]), -1, -1, -1),
                                                            torch.zeros(len(f1[16]), 1, 4, 4)))
    monkeypatch.setattr(R, "refine", lambda model, f1, f2, flow, cert, traj=None:
                        (torch.zeros(len(flow), 2, 32, 32), torch.zeros(len(flow), 1, 32, 32)))
    monkeypatch.setattr(R, "fit_many", lambda flows, certs, num, thr, gen=None: [(I, 10)] * len(flows))
    monkeypatch.setattr(R.augment, "sample_T", lambda rng, *a, **k: np.c_[np.eye(2), [40.0, 0.0]])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)     # 全量测试不给 GPU（close_ticket 的 TEST_CMD）
    p = {**R.PARAMS, "reward": "ref", "group": 4, "temp": 0.3, "shape": "gauss", **over}
    model = types.SimpleNamespace(model=M())
    term = R.Term(p, model, None)
    img = torch.rand(1, 1, 32, 32) + 0.1
    st = types.SimpleNamespace(images=lambda: (img, img), batch={}, trace={}, optim=None)
    return term, st, model.model.w


def test_rloo_term_mp_blocks_image_independent(monkeypatch):
    """mp_n > 0：与图像无关的估计在额外扰动下换算回来差 40 px，超过 mp_thr 时这对图优势置零、粗级没有梯度；放宽 mp_thr 则照常更新。"""
    term, st, w = _fake_term(monkeypatch, mp_n=2)
    loss, stats, _ = term(st)
    loss.backward()
    assert stats["mp_block"] == 1.0 and abs(stats["mp_gap"] - 40.0) < 0.1 and stats["mp_fail"] == 0.0
    assert float(w.grad) == 0.0
    term, st, w = _fake_term(monkeypatch, mp_n=2, mp_thr=100.0)
    loss, stats, _ = term(st)
    loss.backward()
    assert stats["mp_block"] == 0.0 and float(w.grad) != 0.0


def test_rloo_term_frozen_ref_without_mp(monkeypatch):
    """回归（#217 review）：frozen_ref 打开、mp_n = 0 时冻结模型的前向照旧执行。"""
    term, st, w = _fake_term(monkeypatch, frozen_ref=True)
    loss, stats, _ = term(st)
    assert "ref_gap" in stats and "mp_block" not in stats


def test_config_rloo_mp(tmp_path):
    from finetune import config as C
    head = '[model]\nname = "roma"\n[rloo]\n'
    d = C.load(_write(tmp_path, "d", head + 'reward = "ref"\n'))
    assert d["rloo"]["mp_n"] == 0                                           # 默认不做，同旧行为
    c = C.load(_write(tmp_path, "c", head + 'reward = "ref"\nmp_n = 2\n'))
    assert (c["rloo"]["mp_n"], c["rloo"]["mp_shift"], c["rloo"]["mp_thr"]) == (2, 64.0, 16.0)
    for bad in ('reward = "r1"\nmp_n = 2', 'reward = "ref"\nmp_n = -1', 'reward = "ref"\nmp_n = 2\nfrozen_ref = true',
                'reward = "ref"\nmp_n = 2\nref_ema = 0.99',
                'reward = "ref"\nmp_n = 2\nmp_thr = 0.0'):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", head + bad + "\n"))


def test_rloo_advantage_baselines():
    import torch
    from finetune.parts.rloo import advantage

    r = torch.rand(16, 3, 3)
    assert torch.equal(advantage(r, "none"), r)
    a = advantage(r, "loo")
    assert torch.allclose(a[0], r[0] - r[1:].mean(0)) and not torch.allclose(a, r)


def test_config_rloo_ref(tmp_path):
    from finetune import config as C

    c = C.load(_write(tmp_path, "a", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nbaseline = "none"\ntemp = 0.3\n'))
    assert (c["rloo"]["delta"], c["rloo"]["tau"], c["rloo"]["r_out"]) == (8.0, 16.0, -0.1)
    assert c["rloo"]["shape"] == "ripe"                                         # 默认同 E15
    g = C.load(_write(tmp_path, "g", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nshape = "gauss"\ndelta = 4.0\n'))
    assert (g["rloo"]["shape"], g["rloo"]["delta"]) == ("gauss", 4.0)
    with pytest.raises(C.ConfigError):
        C.load(_write(tmp_path, "s", '[model]\nname = "roma"\n[rloo]\nshape = "box"\n'))
    d = C.load(_write(tmp_path, "b", '[model]\nname = "roma"\n[rloo]\n'))
    assert d["rloo"]["baseline"] == "loo" and d["rloo"]["reward"] == "r1"     # 默认同首批
    with pytest.raises(C.ConfigError):
        C.load(_write(tmp_path, "c", '[model]\nname = "roma"\n[rloo]\nbaseline = "mean"\n'))


# ---------- 细化加噪（#192） ----------

def _fake_roma(c=3, seed=0):
    """只有细化层的假 RoMa：每级一个小卷积，输入同 romatch ConvRefiner（两侧特征、warp、certainty）。
    返回 (model, dec, θ, feats, P, cert)：feats() 给出两侧各尺度特征，为 θ · 固定随机图（θ 可训，当作编码器）。"""
    import types

    import torch
    torch.manual_seed(seed)

    class Ref(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2 * c + 3, 3, 3, padding=1)

        def forward(self, a, b, flow, scale_factor=1, logits=None):
            o = self.conv(torch.cat([a, b, flow, logits], 1))
            return o[:, :2], o[:, 2:]

    dec = torch.nn.Module()
    dec.conv_refiner = torch.nn.ModuleDict({s: Ref() for s in ("16", "8", "4", "2", "1")})
    dec.refine_init, dec.flow_upsample_mode = 4.0, "bilinear"
    theta = torch.nn.Parameter(torch.ones(c))
    base1 = {s: torch.randn(1, c, 64 // s, 64 // s) for s in (16, 8, 4, 2, 1)}
    base2 = {s: torch.randn(1, c, 64 // s, 64 // s) for s in (16, 8, 4, 2, 1)}
    feats = lambda: ({s: theta.view(1, c, 1, 1) * x for s, x in base1.items()},
                     {s: theta.view(1, c, 1, 1) * x for s, x in base2.items()})
    model = types.SimpleNamespace(model=types.SimpleNamespace(decoder=dec))
    P = torch.randn(1, 16, 4, 4).softmax(1)                  # 4×4 个锚点
    cert = torch.randn(1, 1, 4, 4)
    return model, dec, theta, feats, P, cert


def test_rloo_rollout_noisy_matches_refine_without_noise():
    """σ = 0 或不加噪时，抽签链的尺度 1 输出与正常细化（rollout）相同；加噪级记下的输入、输出形状对。"""
    import torch
    from finetune.parts.rloo import rollout, rollout_noisy

    model, dec, theta, feats, P, cert = _fake_roma()
    f1, f2 = feats()
    k = torch.randint(0, 16, (5, 4, 4))
    ref, _ = rollout(model, f1, f2, P, cert, k, chunk=2)
    rec0, (fl0, _) = rollout_noisy(model, f1, f2, P, cert, k, {}, chunk=2)
    rec, (fl, _) = rollout_noisy(model, f1, f2, P, cert, k, {8: 0.0, 2: 0.0}, chunk=3)
    assert rec0 == {} and torch.allclose(fl0, ref, atol=1e-6) and torch.allclose(fl, ref, atol=1e-6)
    assert rec[8][0].shape == (5, 2, 8, 8) and rec[2][2].shape == (5, 2, 32, 32) and rec[2][1].shape == (5, 1, 32, 32)
    recn, (fln, _) = rollout_noisy(model, f1, f2, P, cert, k, {8: 0.05}, chunk=3)
    assert not torch.allclose(fln, ref)                          # 加噪的级把噪声带进后面各级


def test_rloo_fine_surrogate_gradient_equals_direct():
    """分段反向（每 chunk 条链立刻反向 + 特征上的替身项）与一次性求整体损失的梯度相同，且不受 GradScaler 倍数影响；
    未加噪的细化层收不到梯度；替身项的值等于细化各级损失之和。"""
    import torch
    from finetune.parts.rloo import _disp, fine_surrogate, rollout_noisy

    model, dec, theta, feats, P, cert = _fake_roma()
    params = [theta] + list(dec.parameters())
    sig = {8: 0.05, 2: 0.02}
    with torch.no_grad():
        f1, f2 = feats()
        rec, _ = rollout_noisy(model, f1, f2, P, cert, torch.randint(0, 16, (4, 4, 4)), sig, chunk=4)
    adv = {s: torch.randn(4, *rec[s][2].shape[-2:]) for s in sig}
    w = 0.7

    f1, f2 = feats()                                              # 直接算：整体损失一次反向
    L = 0
    for s, (fin, cin, x) in rec.items():
        d_flow, _ = dec.conv_refiner[str(s)](f1[s].expand(4, -1, -1, -1), f2[s].expand(4, -1, -1, -1), fin,
                                             scale_factor=1, logits=cin)
        mu = fin + _disp(dec, s, d_flow, 64, 64)
        L = L - w * (adv[s] * (-(x - mu).square().sum(1) / (2 * sig[s] ** 2))).mean()
    L.backward()
    want = [None if p.grad is None else p.grad.clone() for p in params]
    for p in params:
        p.grad = None

    fac = 8.0                                                     # 模拟 GradScaler：backward 反向的是 loss × fac
    f1, f2 = feats()
    sur = fine_surrogate(dec, f1, f2, rec, adv, sig, 3, lambda l: (l * fac).backward(), fac, w)
    (sur * fac).backward()
    assert float(sur) == pytest.approx(float(L), rel=1e-5)
    for p, g in zip(params, want):
        assert (p.grad is None) == (g is None)
        assert g is None or torch.allclose(p.grad / fac, g, rtol=1e-4, atol=1e-6)
    for s in ("16", "4", "1"):
        assert all(p.grad is None for p in dec.conv_refiner[s].parameters())
    assert theta.grad.abs().sum() > 0


def test_rloo_level_reward_scales_with_level():
    """细化级的奖励长度参数按级缩放：尺度 s 上 delta·s/8、tau·s/8（高斯核 δ 8 → 尺度 4 上 δ 4）。"""
    import torch
    from finetune.parts.pseudo import grid
    from finetune.parts.rloo import distance_reward, level_reward

    A = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    x = grid(8, 8, "cpu").permute(2, 0, 1)[None].repeat(3, 1, 1, 1)
    x[:, 0] += (torch.tensor([0.0, 4.0, 8.0]) / 256)[:, None, None]
    p = {"delta": 8.0, "tau": 16.0, "r_out": -0.1, "shape": "gauss", "fine_shape": ""}
    r, d = level_reward(x, A, 4, p)
    assert torch.allclose(d[:, 0, 0], torch.tensor([0.0, 4.0, 8.0]), atol=1e-3)
    assert torch.allclose(r, distance_reward(d, "gauss", 4.0))
    r2, _ = level_reward(x, A, 4, {**p, "shape": "ripe"})
    assert torch.allclose(r2, distance_reward(d, "ripe", 4.0, 8.0, -0.1))


def test_config_rloo_levels(tmp_path):
    from finetune import config as C

    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\n'))
    assert d["rloo"]["levels"] == [] and d["rloo"]["k"] == 0.5                # 默认细化不加噪
    c = C.load(_write(tmp_path, "c", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nlevels = [8, 4]\nk = 1\n'))
    assert c["rloo"]["levels"] == [8, 4] and c["rloo"]["k"] == 1.0
    for bad in ('reward = "ref"\nlevels = [3]', 'reward = "ref"\nlevels = [8, 8]', 'levels = [8]',
                'reward = "ref"\nlevels = [8]\nk = 0.0'):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", f'[model]\nname = "roma"\n[rloo]\n{bad}\n'))


def test_optim_factor():
    import torch
    from finetune import config as C
    from finetune.optim import Optim

    a = torch.nn.Parameter(torch.zeros(1))
    assert Optim([a], {**C.OPTIM, "accum": 4}, amp=False).factor() == 0.25


def test_rloo_fit_many_degenerate_output_counts_as_failed_fit():
    """模型发散后 certainty 全为 −inf / NaN 或 warp 含非有限值时，这份样本记为拟合失败 (None, 0)，不在抽点处报错
    （E20 l84_k025 在 460 步因此中止，#192）；正常样本不受影响。"""
    import torch
    from finetune.parts.pseudo import grid
    from finetune.parts.rloo import fit_many

    flow = grid(64, 64, "cpu").permute(2, 0, 1)[None].repeat(3, 1, 1, 1).clone()
    flow[:, 0] += 8 / 256                                         # 右移 8 px
    cert = torch.full((3, 1, 64, 64), 5.0)
    cert[1] = -float("inf")
    flow[2, 0, 3, 3] = float("nan")
    out = fit_many(flow, cert, 500, 3.0)
    assert out[1] == (None, 0) and out[2] == (None, 0)
    assert out[0][0] is not None and abs(out[0][0][0, 2] - 8) < 0.5


# ---------- 细化第二批（#197） ----------

def test_rloo_cauchy_reward_does_not_underflow():
    """cauchy 1 / (1 + ½(d/δ)²)：d 小时与 gauss 一致到二阶；单精度下 gauss 在 d 超过约 14δ 时为 0，cauchy 在远处
    仍不为 0、单调下降，相邻距离（差 1%）的奖励分得开。"""
    import torch
    from finetune.parts.rloo import distance_reward

    d = torch.tensor([0.0, 0.1, 0.2])
    assert torch.allclose(distance_reward(d, "cauchy", 4.0), distance_reward(d, "gauss", 4.0), atol=1e-6)
    far = torch.tensor([15.0, 100.0, 1e3, 1e4]) * 4                       # 15δ … 10⁴δ，δ = 4
    assert (distance_reward(far, "gauss", 4.0) == 0).all()
    r = distance_reward(far, "cauchy", 4.0)
    assert (r > 0).all() and (r[1:] < r[:-1]).all()
    assert (distance_reward(far * 1.01, "cauchy", 4.0) < r).all()
    assert torch.allclose(r, 1 / (1 + 0.5 * (far / 4) ** 2))


def test_rloo_level_reward_fine_shape_and_level16():
    """fine_shape 非空时细化各级用它，空时同粗级 shape；尺度 16 的长度参数为 delta·2。"""
    import torch
    from finetune.parts.pseudo import grid
    from finetune.parts.rloo import distance_reward, level_reward

    A = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    x = grid(4, 4, "cpu").permute(2, 0, 1)[None].repeat(2, 1, 1, 1)
    x[1, 0] += 40 / 256
    p = {"delta": 8.0, "tau": 16.0, "r_out": -0.1, "shape": "gauss", "fine_shape": "cauchy"}
    r, d = level_reward(x, A, 16, p)
    assert torch.allclose(r, distance_reward(d, "cauchy", 16.0))
    r0, _ = level_reward(x, A, 16, {**p, "fine_shape": ""})
    assert torch.allclose(r0, distance_reward(d, "gauss", 16.0))


def test_config_rloo_level16_and_fine_shape(tmp_path):
    from finetune import config as C

    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\n'))
    assert d["rloo"]["fine_shape"] == "" and d["model"]["refine_lr"] == 1.0 and d["run"]["log_update"] is False
    c = C.load(_write(tmp_path, "c", '[model]\nname = "roma"\nrefine_lr = 0.5\n[run]\nlog_update = true\n'
                                     '[rloo]\nreward = "ref"\nlevels = [16, 8, 4]\nfine_shape = "cauchy"\n'))
    assert c["rloo"]["levels"] == [16, 8, 4] and c["rloo"]["fine_shape"] == "cauchy"
    assert c["model"]["refine_lr"] == 0.5 and c["run"]["log_update"] is True
    with pytest.raises(C.ConfigError):
        C.load(_write(tmp_path, "b", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nfine_shape = "box"\n'))


def test_rloo_rollout_noisy_level16():
    """尺度 16 也能加噪：记下的是尺度 16 细化层的输入与抽到的输出（4×4 格）。"""
    import torch
    from finetune.parts.rloo import rollout, rollout_noisy

    model, dec, theta, feats, P, cert = _fake_roma()
    f1, f2 = feats()
    k = torch.randint(0, 16, (3, 4, 4))
    ref, _ = rollout(model, f1, f2, P, cert, k)
    rec, (fl, _) = rollout_noisy(model, f1, f2, P, cert, k, {16: 0.0})
    assert rec[16][2].shape == (3, 2, 4, 4) and torch.allclose(fl, ref, atol=1e-6)


def test_roma_refine_part_names():
    """细化部分 = 各级 conv_refiner 与尺度 8–1 的 proj；尺度 16 的 proj、GP、embedding decoder 归粗级。"""
    from finetune.models.roma import _part, _refine_level

    assert _refine_level("decoder.conv_refiner.16.block1.0.weight") == "16"
    assert _refine_level("decoder.conv_refiner.4.out_conv.bias") == "4"
    assert _refine_level("decoder.proj.8.0.weight") == "8"
    for name in ("decoder.proj.16.0.weight", "decoder.gps.16.foo", "decoder.embedding_decoder.x", "encoder.cnn.layers.0"):
        assert _refine_level(name) is None
    assert [_part(n) for n in ("decoder.proj.16.0.weight", "decoder.proj.4.0.weight", "encoder.cnn.layers.0.weight",
                               "encoder.dinov2.x")] == ["coarse", "refine", "vgg", None]


def test_roma_split_param_groups():
    """三组参数：粗级 decoder、细化、VGG；train_vgg 关时 VGG 不训、不进任何组。"""
    import torch
    from finetune.models.roma import _split

    names = ["decoder.gps.16.w", "decoder.proj.16.w", "decoder.proj.8.w", "decoder.conv_refiner.16.w",
             "encoder.cnn.layers.0.w", "encoder.other.w"]
    ps = {n: torch.nn.Parameter(torch.zeros(1)) for n in names}
    dec, refine, vgg = _split(ps.items(), {"train_decoder": True, "train_vgg": True})
    assert dec == [ps["decoder.gps.16.w"], ps["decoder.proj.16.w"]]
    assert refine == [ps["decoder.proj.8.w"], ps["decoder.conv_refiner.16.w"]]
    assert vgg == [ps["encoder.cnn.layers.0.w"]] and not ps["encoder.other.w"].requires_grad
    dec, refine, vgg = _split(ps.items(), {"train_decoder": True, "train_vgg": False})
    assert vgg == [] and not ps["encoder.cnn.layers.0.w"].requires_grad


def test_update_monitor_relative_change():
    """普通参数：‖Δθ‖/‖θ‖；LoRA：‖scale·Δ(BA)‖/‖W‖，与直接算等效权重的结果相同；fine 合并 ref* 与 vgg；
    没收到梯度的参数（未加噪的细化级）不计入，整部分没有梯度时不出现。"""
    import torch
    from finetune.optim import UpdateMonitor

    torch.manual_seed(0)
    a, v = torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.full((2,), 2.0))
    c, idle = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.full((100,), 5.0))
    A, B, W = torch.randn(2, 5), torch.randn(6, 2), torch.randn(6, 5)
    for p in (a, v, c):
        p.grad = torch.zeros_like(p)
    mon = UpdateMonitor({"ref8": [a], "ref1": [idle], "vgg": [v], "coarse": [c]}, [(A, B, 2.0, W)])
    mon.before()
    BA0 = (B @ A).clone()
    with torch.no_grad():
        a += 0.1
        v -= 0.2
        A += 0.05 * torch.randn_like(A)
        B += 0.05 * torch.randn_like(B)
    out = mon.after()
    assert out["upd_coarse"] == 0.0 and "upd_ref1" not in out
    assert out["upd_ref8"] == pytest.approx(0.1, rel=1e-3)
    assert out["upd_vgg"] == pytest.approx(0.1, rel=1e-3)
    assert out["upd_fine"] == pytest.approx(float(np.sqrt((4 * 0.01 + 2 * 0.04) / (4 + 8))), rel=1e-3)
    want = float((2.0 * (B @ A - BA0)).norm() / W.norm())
    assert out["upd_lora"] == pytest.approx(want, rel=1e-3)


def test_update_monitor_lora_tiny_change_stays_accurate():
    """LoRA 的 BA 已很大、这一步只改一点时（E22 训练中在 ~1000 步出现），‖Δ(BA)‖ 不因大数相减变负或失准。"""
    import torch
    from finetune.optim import UpdateMonitor

    torch.manual_seed(0)
    A, B, W = torch.randn(8, 1024) * 10, torch.randn(3072, 8) * 10, torch.randn(3072, 1024)
    mon = UpdateMonitor({}, [(A, B, 2.0, W)])
    for _ in range(5):
        mon.before()
        BA0 = (B.double() @ A.double()).clone()
        with torch.no_grad():
            A += 1e-6 * torch.randn_like(A)
        want = float((2.0 * (B.double() @ A.double() - BA0)).norm() / W.double().norm())
        assert mon.after()["upd_lora"] == pytest.approx(want, rel=1e-2)


# ---------- 冻结参照（#202） ----------

def test_config_rloo_frozen_ref(tmp_path):
    from finetune import config as C

    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\n'))
    assert d["rloo"]["frozen_ref"] is False                                   # 默认参照随当前模型（旧行为）
    c = C.load(_write(tmp_path, "c", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nfrozen_ref = true\n'))
    assert c["rloo"]["frozen_ref"] is True
    for bad in ('[rloo]\nreward = "r1"\nfrozen_ref = true', 'dino_unfreeze = 2\n[rloo]\nreward = "ref"\nfrozen_ref = true',
                'lora_r = 8\ndino_ckpt = true\n[rloo]\nreward = "ref"\nfrozen_ref = true',
                'lora_r = 8\ndrop_path = 0.1\n[rloo]\nreward = "ref"\nfrozen_ref = true',
                'vgg_dropout = 0.1\n[rloo]\nreward = "ref"\nfrozen_ref = true'):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", f'[model]\nname = "roma"\n{bad}\n'))


def test_rloo_frozen_copy_is_independent_and_frozen():
    """冻结副本：参数与原模型数值相同但不共享、不求梯度、eval 模式；原模型训练后副本不变。encoder 上引用原模型的
    实例级闭包 forward 被删掉，副本走类自己的 forward（用副本的参数）；其余模块有实例级 forward 时报错。"""
    import types
    import torch
    from finetune.parts.rloo import frozen_copy

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.lin(x)

    net = torch.nn.Module()
    net.encoder, net.head = Enc(), torch.nn.Linear(2, 2)
    orig = net.encoder
    net.encoder.forward = lambda x: orig.lin(x) * 0 + 7          # 同 finetune.models.roma._open_dino 的闭包写法
    net.train()
    model = types.SimpleNamespace(model=net)
    fz = frozen_copy(model).model
    assert not fz.training and not any(p.requires_grad for p in fz.parameters())
    assert all(p.requires_grad for p in net.parameters()) and net.training
    x = torch.randn(3, 2)
    assert torch.allclose(fz.encoder(x), net.encoder.lin(x))          # 类 forward，不是原模型的闭包
    with torch.no_grad():
        net.encoder.lin.weight.add_(1.0)
        net.head.weight.add_(1.0)
    assert not torch.allclose(fz.encoder.lin.weight, net.encoder.lin.weight)
    assert not torch.allclose(fz.head.weight, net.head.weight)
    net.head.forward = lambda x: x
    with pytest.raises(ValueError):
        frozen_copy(model)
    del net.head.forward
    net.encoder.dinov2_vitl14 = [torch.nn.Linear(2, 2)]           # 同 romatch：DINOv2 放在列表里，不是子模块
    fz = frozen_copy(model).model
    dv = fz.encoder.dinov2_vitl14[0]
    assert dv is not net.encoder.dinov2_vitl14[0] and not dv.training and not dv.weight.requires_grad
    net.head.register_forward_hook(lambda mod, inp, out: out)
    with pytest.raises(ValueError):
        frozen_copy(model)


def test_rloo_affine_gap():
    """两个仿射把格子中心映射到的位置之间的平均距离：相同为 0，相差平移 (3, 4) px 为 5；任一为 None 时 None。"""
    from finetune.parts.rloo import affine_gap

    A = np.array([[1.0, 0.1, 5.0], [-0.1, 1.0, 2.0]])
    B = A.copy()
    B[:, 2] += [3.0, 4.0]
    assert affine_gap(A, A, 8, 8) == pytest.approx(0.0, abs=1e-6)
    assert affine_gap(A, B, 8, 8) == pytest.approx(5.0, rel=1e-4)
    assert affine_gap(None, A, 8, 8) is None
# ---------- 细化训练方式：精确梯度与逐级局部抽签（#201） ----------

def _traj(model, f1, f2, P, cert):
    import torch
    from finetune.parts.rloo import anchor_flow, refine

    traj = {}
    with torch.no_grad():
        out, _ = refine(model, f1, f2, anchor_flow(P, P.argmax(1)), cert, traj)
    return traj, out


def test_rloo_refine_records_trajectory():
    """refine 的 traj 记下每级的输入与输出 μ：尺度 1 的 μ 就是返回的 warp；不给 traj 时返回值不变。"""
    import torch
    from finetune.parts.rloo import anchor_flow, refine

    model, dec, theta, feats, P, cert = _fake_roma()
    f1, f2 = feats()
    traj, out = _traj(model, f1, f2, P, cert)
    assert sorted(traj) == [1, 2, 4, 8, 16] and traj[8][0].shape == (1, 2, 8, 8) and traj[8][1].shape == (1, 1, 8, 8)
    assert torch.allclose(traj[1][2], out)
    with torch.no_grad():
        plain, _ = refine(model, f1, f2, anchor_flow(P, P.argmax(1)), cert)
    assert torch.equal(plain, out)


def test_rloo_fine_direct_grad_equals_direct():
    """grad 方式：分段反向 + 替身项的梯度与直接对 −Σ_s mean r(μ_s) 求导相同（不受 GradScaler 倍数影响），
    替身项的值等于这个损失；不训练的级没有梯度。"""
    import torch
    from finetune.parts.rloo import _disp, fine_direct, level_reward

    model, dec, theta, feats, P, cert = _fake_roma()
    params = [theta] + list(dec.parameters())
    with torch.no_grad():
        traj, _ = _traj(model, *feats(), P, cert)
    A = np.array([[1.0, 0.02, 3.0], [-0.01, 1.0, -2.0]])
    p = {"delta": 8.0, "tau": 16.0, "r_out": -0.1, "shape": "gauss", "fine_shape": "cauchy", "fine_mode": "grad",
         "fine_group": 8, "baseline": "loo"}
    sig = {8: 0.5 * 8 / 256, 2: 0.5 * 2 / 256}
    w = 0.7

    f1, f2 = feats()
    L = 0
    for s in sig:
        fin, cin, _ = traj[s]
        d_flow, _ = dec.conv_refiner[str(s)](f1[s], f2[s], fin, scale_factor=1, logits=cin)
        L = L - w * level_reward(fin + _disp(dec, s, d_flow, 64, 64), A, s, p)[0].mean()
    L.backward()
    want = [None if q.grad is None else q.grad.clone() for q in params]
    for q in params:
        q.grad = None

    fac = 8.0
    f1, f2 = feats()
    sur, rd, _ = fine_direct(dec, f1, f2, traj, A, sig, p, lambda l: (l * fac).backward(), fac, w)
    (sur * fac).backward()
    assert float(sur) == pytest.approx(float(L), rel=1e-5)
    assert rd[8][0].shape == (1, 8, 8) and rd[2][1].shape == (1, 32, 32)
    for q, g in zip(params, want):
        assert (q.grad is None) == (g is None)
        assert g is None or torch.allclose(q.grad / fac, g, rtol=1e-4, atol=1e-7)
    for s in ("16", "4", "1"):
        assert all(q.grad is None for q in dec.conv_refiner[s].parameters())
    assert theta.grad.abs().sum() > 0


def test_rloo_local_gradient_agrees_with_exact_for_many_draws():
    """local 方式：抽签次数很多时，它的梯度方向趋近精确梯度（grad_cos 接近 1）；抽 2 次时余弦明显更小但有定义；
    local 的损失只经 log π 带梯度，抽到的位置与奖励不带梯度。"""
    import torch
    from finetune.parts.rloo import grad_cos, level_loss

    torch.manual_seed(0)
    model, dec, theta, feats, P, cert = _fake_roma()
    traj, _ = _traj(model, *feats(), P, cert)
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = {"delta": 8.0, "tau": 16.0, "r_out": -0.1, "shape": "gauss", "fine_shape": "cauchy", "fine_mode": "local",
         "baseline": "loo"}
    sig = {8: 0.5 * 8 / 256, 4: 0.5 * 4 / 256}
    f1, f2 = feats()
    state = torch.get_rng_state()
    many = grad_cos(dec, f1, f2, traj, A, sig, {**p, "fine_group": 20000})
    few = grad_cos(dec, f1, f2, traj, A, sig, {**p, "fine_group": 2})
    assert torch.equal(torch.get_rng_state(), state)                             # 不动全局随机数流
    assert set(many) == {"cos_8", "cos_4", "cosf_8", "cosf_4", "cos"}
    assert many["cos"] > 0.95 and many["cos_8"] > 0.95 and many["cosf_4"] > 0.9
    assert -1 <= few["cos"] <= 1 and few["cos"] < many["cos"]
    assert theta.grad is None and all(q.grad is None for q in dec.parameters())   # 诊断不碰 .grad
    mu = traj[8][2].clone().requires_grad_()
    loss, (r, d) = level_loss(mu, A, 8, sig[8], {**p, "fine_group": 4}, "local")
    assert r.shape == (4, 8, 8) and not r.requires_grad and loss.requires_grad


def test_rloo_fine_direct_local_equals_direct():
    """local 方式：分段反向 + 替身项的梯度与同一批抽签下直接求 −Σ_s mean(RLOO 优势 · log π_s) 的梯度相同。"""
    import torch
    from finetune.parts.rloo import _disp, advantage, fine_direct, level_reward

    model, dec, theta, feats, P, cert = _fake_roma()
    params = [theta] + list(dec.parameters())
    with torch.no_grad():
        traj, _ = _traj(model, *feats(), P, cert)
    A = np.array([[1.0, 0.02, 3.0], [-0.01, 1.0, -2.0]])
    p = {"delta": 8.0, "tau": 16.0, "r_out": -0.1, "shape": "gauss", "fine_shape": "cauchy", "fine_mode": "local",
         "fine_group": 6, "baseline": "loo"}
    sig = {16: 0.5 * 16 / 256, 4: 0.5 * 4 / 256}

    torch.manual_seed(5)
    f1, f2 = feats()
    L = 0
    for s in sig:
        fin, cin, _ = traj[s]
        d_flow, _ = dec.conv_refiner[str(s)](f1[s], f2[s], fin, scale_factor=1, logits=cin)
        mu = fin + _disp(dec, s, d_flow, 64, 64)
        x = mu.detach() + sig[s] * torch.randn((6,) + tuple(mu.shape[1:]))
        adv = advantage(level_reward(x, A, s, p)[0], "loo")
        L = L - (adv * (-(x - mu).square().sum(1) / (2 * sig[s] ** 2))).mean()
    L.backward()
    want = [None if q.grad is None else q.grad.clone() for q in params]
    for q in params:
        q.grad = None

    torch.manual_seed(5)
    f1, f2 = feats()
    sur, rd, _ = fine_direct(dec, f1, f2, traj, A, sig, p, lambda l: (l * 4.0).backward(), 4.0)
    (sur * 4.0).backward()
    assert float(sur) == pytest.approx(float(L), rel=1e-5) and rd[16][0].shape == (6, 4, 4)
    for q, g in zip(params, want):
        assert (q.grad is None) == (g is None)
        assert g is None or torch.allclose(q.grad / 4.0, g, rtol=1e-4, atol=1e-7)


def test_config_rloo_fine_mode(tmp_path):
    from finetune import config as C

    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nlevels = [8]\n'))
    assert (d["rloo"]["fine_mode"], d["rloo"]["fine_group"], d["rloo"]["cos_every"]) == ("chain", 8, 0)  # 默认整链
    c = C.load(_write(tmp_path, "c", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nlevels = [16, 8, 4]\n'
                                     'fine_mode = "local"\ncos_every = 50\n'))
    assert c["rloo"]["fine_mode"] == "local" and c["rloo"]["cos_every"] == 50
    for bad in ('fine_mode = "exact"\nlevels = [8]', 'fine_mode = "grad"', 'cos_every = 50',
                'fine_mode = "local"\nlevels = [8]\nfine_group = 1', 'levels = [8]\ncos_every = -1'):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", f'[model]\nname = "roma"\n[rloo]\nreward = "ref"\n{bad}\n'))


# ---------- 逐步记录与续训（#209） ----------

def test_config_run_resume_and_trace_options(tmp_path):
    from finetune import config as C

    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\n'))
    assert (d["run"]["stop_at"], d["run"]["save_optim"], d["run"]["resume"], d["run"]["sweep"]) == (0, False, "", "all")
    assert (d["rloo"]["trace"], d["rloo"]["obs_every"], d["rloo"]["obs_n"]) == (False, 0, 16)   # 默认不记录（旧行为）
    c = C.load(_write(tmp_path, "c", '[model]\nname = "roma"\n[optim]\nsteps = 4000\n[run]\nstop_at = 1200\n'
                                     'save_optim = true\nsweep = "last"\n[rloo]\nreward = "ref"\ntrace = true\nobs_every = 10\n'))
    assert c["run"]["stop_at"] == 1200 and c["run"]["sweep"] == "last" and c["rloo"]["obs_every"] == 10
    for bad in ('[run]\nsweep = "none"', '[optim]\nsteps = 100\n[run]\nstop_at = 200', '[run]\nstop_at = -1',
                '[optim]\nema = 0.99\n[run]\nsave_optim = true', '[optim]\nema = 0.99\n[run]\nresume = "x.pt"',
                '[rloo]\nreward = "r1"\ntrace = true', '[rloo]\nreward = "ncc"\nobs_every = 10',
                '[rloo]\nreward = "ref"\nobs_every = 10\nobs_n = 0'):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", f'[model]\nname = "roma"\n{bad}\n'))
    with pytest.raises(C.ConfigError):         # loftr 没有 load_weights，续训装不回全部权重
        C.load(_write(tmp_path, "l", '[model]\nname = "loftr"\n[run]\nresume = "x.pt"\n'))


class _RandSet:
    """每个样本 = (序号, worker 里抽的随机数)：数据顺序与 worker 的随机数流都要在续训后对得上。"""

    def __len__(self):
        return 12

    def __getitem__(self, i):
        import torch
        return torch.tensor([float(i), float(torch.rand(()))])


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_resume_continues_same_batches(workers):
    """从第一个 epoch 中途的状态续上，之后的 batch（顺序与 worker 内的随机数）与不中断时相同；第二个 epoch 的状态拒绝续训。"""
    import itertools
    import torch
    from finetune.trace import Loader

    def make():
        torch.manual_seed(5)                     # workers = 0 时随机数取自主进程的全局流
        g = torch.Generator().manual_seed(3)
        dl = torch.utils.data.DataLoader(_RandSet(), batch_size=2, shuffle=True, generator=g, num_workers=workers,
                                         persistent_workers=workers > 0)
        return Loader(dl, g)

    full = [b.clone() for b in itertools.islice(iter(make()), 9)]          # 跨过第一个 epoch（6 个 batch）
    a = make()
    it = iter(a)
    head = [next(it).clone() for _ in range(2)]
    s = a.state()
    assert s["epoch"] == 0 and s["batch"] == 2
    b = make()
    b.load(s)
    tail = [x.clone() for x in itertools.islice(iter(b), 7)]
    for x, y in zip(head + tail, full):
        assert torch.equal(x, y)
    for _ in range(5):
        next(it)
    with pytest.raises(ValueError):
        make().load(a.state())


def test_rng_state_roundtrip():
    import random
    import torch
    from finetune.trace import rng_state, set_rng_state

    rng = np.random.default_rng(0)
    s = rng_state(rng)
    first = (float(torch.rand(())), float(np.random.rand()), random.random(), float(rng.random()))
    set_rng_state(s, rng)
    assert (float(torch.rand(())), float(np.random.rand()), random.random(), float(rng.random())) == first


def test_optim_state_roundtrip():
    """AdamW 的状态存下再装回，之后的更新与不中断时相同。"""
    import copy
    import torch
    from finetune import config as C
    from finetune.optim import Optim

    o = {**C.OPTIM, "warmup": 0, "sched": "const", "lr": 1e-2, "wd": 0.01}

    def train(w, opt, steps):
        for st in steps:
            (w * torch.arange(1.0, 4.0)).square().sum().backward()
            opt.step(st)

    w = torch.nn.Parameter(torch.ones(3))
    opt = Optim([w], o, amp=False)
    train(w, opt, [1, 2])
    sd, w_mid = copy.deepcopy(opt.state_dict()), w.detach().clone()
    train(w, opt, [3, 4])
    v = torch.nn.Parameter(w_mid.clone())
    opt2 = Optim([v], o, amp=False)
    opt2.load_state_dict(sd)
    train(v, opt2, [3, 4])
    assert torch.equal(v, w)


def test_roma_load_weights_roundtrip():
    """Model.state_dict 的输出经 load_weights 装回（网络与 LoRA）；LoRA 层对不上时报错。"""
    import types
    import torch
    from finetune.models.roma import LoRA, Model, load_weights

    def make():
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        return types.SimpleNamespace(model=net, lora={"blocks.0.attn.qkv": LoRA(torch.nn.Linear(4, 6), 2, 4.0)},
                                     unc_heads=None, dino_blocks={}, adapter=types.SimpleNamespace(lora_merged=[]))

    a = make()
    with torch.no_grad():
        a.model[0].weight.add_(1.0)
        a.lora["blocks.0.attn.qkv"].B.add_(0.5)
    sd = Model.state_dict(a)
    b = make()
    load_weights(b, sd)
    assert torch.equal(b.model[0].weight, a.model[0].weight)
    assert torch.equal(b.lora["blocks.0.attn.qkv"].B, a.lora["blocks.0.attn.qkv"].B)
    b.lora = {}
    with pytest.raises(ValueError):
        load_weights(b, sd)


def test_trace_stack_rows_and_writer(tmp_path):
    """缺的量浮点补 NaN、整数补 −1；每 every 步写一个 npz，末尾不足的由 flush 写出；没有记录时不建目录。"""
    from finetune.trace import TraceWriter, stack_rows

    out = stack_rows([{"a": np.ones(2, np.float16), "k": np.int16(3)}, {"b": np.zeros((1, 2))}])
    assert np.isnan(out["a"][1]).all() and out["k"].tolist() == [3, -1] and out["b"].shape == (2, 1, 2)
    w = TraceWriter(tmp_path, every=2)
    w.add(1, {})
    assert not (tmp_path / "trace").exists()
    for st in (2, 3, 4):
        w.add(st, {"x": np.full((1, 2), st)})
    w.flush()
    names = sorted(p.name for p in (tmp_path / "trace").iterdir())
    assert names == ["train_00002-00003.npz", "train_00004-00004.npz"]
    z = np.load(tmp_path / "trace" / names[0])
    assert z["step"].tolist() == [2, 3] and z["x"].shape == (2, 1, 2)


def test_rloo_observer_failed_fit_rows_fill_nan(monkeypatch):
    """观察集里自估仿射拟合失败的样本没有 dmu_* 等量：叠起时补 NaN，不因各样本的键不同而出错（#209 的修复）。"""
    import torch
    from finetune.parts import rloo as R
    from finetune.parts.rloo_trace import Observer

    C, h = 16, 4
    eye = np.c_[np.eye(2), np.zeros(2)]
    monkeypatch.setattr(R, "features", lambda model, i0, i1: ({}, {}))
    monkeypatch.setattr(R, "coarse", lambda model, f1, f2: (torch.randn(1, C, h, h), torch.zeros(1, 1, h, h)))
    monkeypatch.setattr(R, "anchor_flow", lambda P, k: torch.zeros(P.shape[0], 2, h, h))

    def refine(model, f1, f2, flow, cert, traj):
        traj[16] = (None, None, torch.zeros(1, 2, 8, 8))
        return torch.zeros(1, 2, 16, 16), torch.zeros(1, 1, 16, 16)

    fits = iter([[(eye, 50)], [(None, 0)]])
    monkeypatch.setattr(R, "refine", refine)
    monkeypatch.setattr(R, "fit_many", lambda *a, **k: next(fits))
    ob = object.__new__(Observer)
    ob.p = {"temp": 0.3, "num": 100, "ransac": 3.0, "obs_seed": 0}
    ob.model = type("M", (), {"device": "cpu"})()
    pts = np.array([[10.0, 20.0], [100.0, 200.0]])
    ob.items = [{"pair": f"p{v}", "version": v, "T": eye, "image0": torch.rand(1, 1, 64, 64),
                 "image1": torch.rand(1, 1, 64, 64), "cp_opt": pts, "cp_sar": pts, "A_gt": eye} for v in (0, 1)]
    out = ob(10)
    assert out["dmu_16"].shape[0] == 2 and np.isfinite(out["dmu_16"][0]).all() and np.isnan(out["dmu_16"][1]).all()
    assert out["err"][0] == 0 and np.isinf(out["err"][1]) and out["pair"].tolist() == ["p0", "p1"]


def test_rloo_cell_signal_matches_autograd():
    """每格的 v = Σ_g a_g (e_k − q) / (G·temp) 等于粗级损失（逐格求和）对 logits 的负梯度；第二个量是它在最高分位置上的分量。"""
    import torch
    from finetune.parts.rloo import _logp
    from finetune.parts.rloo_trace import cell_signal

    torch.manual_seed(0)
    G, C, h, w, temp = 5, 16, 3, 2, 0.3
    s = torch.randn(C, h, w, requires_grad=True)
    k = torch.randint(0, C, (G, h, w))
    adv = torch.randn(G, h, w)
    logp = torch.log_softmax(s / temp, 0)
    (-(adv * _logp(logp, k)).mean(0).sum()).backward()
    q = logp.detach().exp()
    norm, up = cell_signal(q, k, adv, temp)
    v = -s.grad
    assert torch.allclose(norm, v.norm(dim=0), atol=1e-5)
    assert torch.allclose(up, v.gather(0, q.argmax(0)[None])[0], atol=1e-5)


def test_rloo_trace_mapped_and_pool():
    """恒等仿射把网格各点中心映射到 512/n·(j + 0.5)；pool 取块均值。"""
    import torch
    from finetune.parts.rloo_trace import mapped, pool

    m = mapped(np.c_[np.eye(2), np.zeros(2)], 40, 40, "cpu")
    assert torch.allclose(m[0, 0], 12.8 * (torch.arange(40) + 0.5), atol=1e-3)
    assert torch.allclose(m[1, :, 0], 12.8 * (torch.arange(40) + 0.5), atol=1e-3)
    x = torch.arange(16.0).reshape(4, 4)
    assert pool(x, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]


def test_rloo_fit_many_with_generator_leaves_global_rng():
    """给了发生器的抽点不动全局随机数流（观察集的记录不改变训练，#209）。"""
    import torch
    from finetune.parts.pseudo import grid
    from finetune.parts.rloo import fit_many

    flow = grid(32, 32, "cpu").permute(2, 0, 1)[None].clone()
    cert = torch.zeros(1, 1, 32, 32)
    s = torch.get_rng_state()
    out = fit_many(flow, cert, 200, 3.0, gen=torch.Generator().manual_seed(1))
    assert torch.equal(torch.get_rng_state(), s) and out[0][0] is not None


# ---------- 不让参照跑偏（#216）：EMA 参照、参照合理性检查、格子分散约束 ----------

def test_config_rloo_ref_guards(tmp_path):
    from finetune import config as C

    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\n'))
    r = d["rloo"]
    assert (r["ref_ema"], r["gate"], r["spread"]) == (0.0, False, 0.0)                # 默认同旧行为
    c = C.load(_write(tmp_path, "c", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nframe_mask = true\n'
                                     'cert_bce = 0.01\nref_ema = 0.99\ngate = true\ngate_in = 0.43\nspread = 0.1\n'))
    assert (c["rloo"]["ref_ema"], c["rloo"]["gate_in"], c["rloo"]["spread_h0"]) == (0.99, 0.43, 4.0)
    for bad in ('[rloo]\nreward = "r1"\nref_ema = 0.99', '[rloo]\nreward = "r1"\ngate = true',
                '[rloo]\nreward = "r1"\nspread = 0.1', '[rloo]\nreward = "ref"\nref_ema = 1.0',
                '[rloo]\nreward = "ref"\nref_ema = 0.99\nfrozen_ref = true',
                'vgg_dropout = 0.1\n[rloo]\nreward = "ref"\nref_ema = 0.99',
                '[optim]\naccum = 2\n[rloo]\nreward = "ref"\nref_ema = 0.99',
                '[rloo]\nreward = "ref"\ngate = true\ngate_det_lo = 3.0', '[rloo]\nreward = "ref"\ngate_sv = 1.5',
                '[rloo]\nreward = "ref"\nspread = -0.1',
                '[rloo]\nreward = "ref"\nframe_mask = true\nfrozen_ref = true'):     # 冻结参照仍不配掩码
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", f'[model]\nname = "roma"\n{bad}\n'))


def _toy_roma():
    """frozen_copy 能处理的小模型：本体一个可训层、一个冻结层，encoder 的列表里放一个「DINOv2」（冻结底座 + 可训支路）。"""
    import types
    import torch

    net = torch.nn.Module()
    net.encoder, net.head, net.fixed = torch.nn.Module(), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    net.fixed.requires_grad_(False)
    dv = torch.nn.Module()
    dv.base, dv.lora = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    dv.base.requires_grad_(False)
    net.encoder.dinov2_vitl14 = [dv]
    return types.SimpleNamespace(model=net)


def test_rloo_slow_ref_is_ema_of_trainable_params():
    """EMA 参照：每次 update 后 θ_slow = τ·θ_slow + (1−τ)·θ，只动可训练参数（含列表里 DINOv2 的可训支路）；
    冻结参数保持拷贝时的值；存取后照样接着平均。"""
    import torch
    from finetune.parts.rloo import SlowRef, frozen_copy

    model = _toy_roma()
    slow = frozen_copy(model)
    sr = SlowRef(model, slow, 0.9)
    assert len(sr.pairs) == 4                                   # head.weight/bias、lora.weight/bias
    net, sm = model.model, slow.model
    sdv, dv = sm.encoder.dinov2_vitl14[0], net.encoder.dinov2_vitl14[0]
    w0, l0, f0, b0 = (x.detach().clone() for x in (net.head.weight, dv.lora.weight, net.fixed.weight, dv.base.weight))
    with torch.no_grad():
        for x in (net.head.weight, dv.lora.weight, net.fixed.weight, dv.base.weight):
            x.add_(1.0)
    sr.update()
    assert torch.allclose(sm.head.weight, w0 + 0.1) and torch.allclose(sdv.lora.weight, l0 + 0.1)
    assert torch.equal(sm.fixed.weight, f0) and torch.equal(sdv.base.weight, b0)
    sd = sr.state_dict()
    sr2 = SlowRef(model, frozen_copy(model), 0.9)               # 新副本从当前参数起，装回存档后与 sr 相同
    sr2.load_state_dict(sd)
    sr.update()
    sr2.update()
    assert torch.allclose(sr2.pairs[0][1], sr.pairs[0][1])
    assert torch.allclose(sm.head.weight, w0 + 0.1 + 0.09)


def test_rloo_gate_check():
    """参照合理性检查：恒等、扰动范围内的相似变换通过；面积倍数、压扁程度、画面内比例各自超限时只那一项不过。"""
    from finetune.parts.rloo import affine_shape, gate_check

    p = {"gate_det_lo": 0.5, "gate_det_hi": 2.0, "gate_sv": 0.7, "gate_in": 0.43}
    c, s_ = np.cos(0.3), np.sin(0.3)
    sim = np.array([[1.2 * c, -1.2 * s_, 50.0], [1.2 * s_, 1.2 * c, -20.0]])     # 缩放 1.2、旋转、平移
    assert affine_shape(sim) == pytest.approx((1.44, 1.0))
    assert gate_check(sim, 0.8, p) == {"det": False, "sv": False, "in": False}
    assert gate_check(np.array([[0.6, 0, 0], [0, 0.6, 0.0]]), 0.8, p) == {"det": True, "sv": False, "in": False}
    assert gate_check(np.array([[1.0, 0, 0], [0, 0.6, 0.0]]), 0.8, p) == {"det": False, "sv": True, "in": False}
    assert gate_check(np.array([[1.0, 0, 0], [0, -1.0, 0.0]]), 0.8, p)["det"]                # 镜像：行列式为负
    assert gate_check(np.array([[1.0, 2.0, 0], [0.5, 1.0, 0.0]]), 0.8, p)["det"]             # 退成一条线
    assert gate_check(sim, 0.3, p) == {"det": False, "sv": False, "in": True}


def test_rloo_spread_loss():
    """格子分散约束：各格挤到同一锚点时 H(p̄) ≈ 0、损失为 h0；各格分到不同锚点时 H(p̄) = log(格数) ≥ h0、损失为 0；
    掩码只取画面内格子；H(p̄) < h0 时梯度不为 0。"""
    import torch
    import torch.nn.functional as F
    from finetune.parts.rloo import spread_loss

    C_, h, w = 256, 8, 8
    same = torch.full((C_, h, w), -30.0)
    same[5] = 30.0
    l, H = spread_loss(F.log_softmax(same, 0), None, 4.0)
    assert float(H) == pytest.approx(0.0, abs=1e-4) and float(l) == pytest.approx(4.0, abs=1e-4)
    diff = torch.full((C_, h, w), -30.0)
    for i in range(h * w):
        diff[i, i // w, i % w] = 30.0
    l, H = spread_loss(F.log_softmax(diff, 0), None, 4.0)
    assert float(H) == pytest.approx(np.log(h * w), abs=1e-3) and float(l) == 0.0
    m = torch.zeros(h, w, dtype=torch.bool)
    m[0, :2] = True                                                 # 只取两格：p̄ 平摊在两个锚点上
    _, H = spread_loss(F.log_softmax(diff, 0), m, 4.0)
    assert float(H) == pytest.approx(np.log(2), abs=1e-3)
    _, H0 = spread_loss(F.log_softmax(diff, 0), torch.zeros(h, w, dtype=torch.bool), 4.0)   # 全 False：取全部格子
    assert float(H0) == pytest.approx(np.log(h * w), abs=1e-3)
    x = torch.randn(C_, h, w, requires_grad=True)
    l, H = spread_loss(F.log_softmax(x * 10, 0), None, 8.0)
    assert float(H) < 8.0
    l.backward()
    assert float(x.grad.abs().sum()) > 0


# ---------- 细化与防坍缩办法同用（#221） ----------

def test_config_rloo_fine_w_mask_and_mp_with_levels(tmp_path):
    from finetune import config as C
    head = '[model]\nname = "roma"\n[rloo]\nreward = "ref"\n'
    d = C.load(_write(tmp_path, "d", head + 'levels = [16]\nfine_mode = "local"\n'))
    assert (d["rloo"]["fine_w"], d["rloo"]["fine_mask"], d["rloo"]["fine_cert_bce"]) == (1.0, False, 0.0)  # 默认同旧行为
    c = C.load(_write(tmp_path, "c", head + 'levels = [16, 8, 4]\nfine_mode = "local"\nfine_w = 0.3\nfine_mask = true\n'
                                            'fine_cert_bce = 0.01\nmp_n = 2\n'))
    assert (c["rloo"]["fine_w"], c["rloo"]["fine_mask"], c["rloo"]["fine_cert_bce"], c["rloo"]["mp_n"]) == \
        (0.3, True, 0.01, 2)                                                  # mp_n 与 levels 可以同开
    for bad in ('fine_mask = true', 'levels = [8]\nfine_mask = true', 'levels = [8]\nfine_cert_bce = 0.01',
                'levels = [8]\nfine_mode = "local"\nfine_w = -1.0',
                'levels = [8]\nfine_mode = "local"\nfine_cert_bce = -0.1',
                'levels = [8]\nfine_mode = "local"\nfine_mask = true\nfrozen_ref = true'):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", head + bad + "\n"))


def test_rloo_fine_direct_keep_and_cert_bce_equal_direct():
    """keep（逐像素权重）与细化 certainty 的 BCE：分段反向 + 替身项的梯度与直接求
    Σ_s [w · −mean(keep · 优势 · log π_s) + bce · BCE(cin + Δcert, 掩码)] 的梯度相同；keep 全 0 时只剩 BCE 的梯度。"""
    import torch
    import torch.nn.functional as F
    from finetune.parts.rloo import _disp, advantage, fine_direct, level_reward

    model, dec, theta, feats, P, cert = _fake_roma()
    params = [theta] + list(dec.parameters())
    with torch.no_grad():
        traj, _ = _traj(model, *feats(), P, cert)
    A = np.array([[1.0, 0.02, 3.0], [-0.01, 1.0, -2.0]])
    p = {"delta": 8.0, "tau": 16.0, "r_out": -0.1, "shape": "gauss", "fine_shape": "cauchy", "fine_mode": "local",
         "fine_group": 6, "baseline": "loo"}
    sig = {16: 0.5 * 16 / 256, 4: 0.5 * 4 / 256}
    g = torch.Generator().manual_seed(1)
    masks = {s: torch.rand(traj[s][0].shape[-2:], generator=g) > 0.3 for s in sig}
    w, bce = 0.7, 0.05

    def direct(keep):
        torch.manual_seed(5)
        f1, f2 = feats()
        L = 0
        for s in sig:
            fin, cin, _ = traj[s]
            d_flow, d_cert = dec.conv_refiner[str(s)](f1[s], f2[s], fin, scale_factor=1, logits=cin)
            mu = fin + _disp(dec, s, d_flow, 64, 64)
            x = mu.detach() + sig[s] * torch.randn((6,) + tuple(mu.shape[1:]))
            adv = advantage(level_reward(x, A, s, p)[0], "loo") * keep[s]
            L = L - w * (adv * (-(x - mu).square().sum(1) / (2 * sig[s] ** 2))).mean()
            L = L + bce * F.binary_cross_entropy_with_logits((cin + d_cert)[0, 0], masks[s].float())
        L.backward()
        out = [None if q.grad is None else q.grad.clone() for q in params]
        for q in params:
            q.grad = None
        return float(L), out

    for keep in ({s: m.float() for s, m in masks.items()},
                 {s: torch.zeros(m.shape) for s, m in masks.items()}):
        L, want = direct(keep)
        torch.manual_seed(5)
        f1, f2 = feats()
        sur, rd, lc = fine_direct(dec, f1, f2, traj, A, sig, p, lambda l: (l * 4.0).backward(), 4.0, w, keep, masks, bce)
        (sur * 4.0).backward()
        assert float(sur) == pytest.approx(L, rel=1e-5) and set(lc) == {16, 4}
        for q, gw in zip(params, want):
            assert (q.grad is None) == (gw is None)
            assert gw is None or torch.allclose(q.grad / 4.0, gw, rtol=1e-4, atol=1e-7)
            q.grad = None
    sur, rd, lc = fine_direct(dec, *feats(), traj, A, sig, p, lambda l: l.backward(), 1.0)   # 默认：不加 BCE
    assert lc == {}


def _fake_term_fine(monkeypatch, **over):
    """rloo.Term 的整条路径、带细化（#221）：粗级同 _fake_term（4×4 格、16 个锚点，拟合总给同一个仿射、与图像无关），
    细化用 _fake_roma 的小卷积细化层、照 refine 真跑。返回 (Term, 一步的输入, 粗级参数 w, 细化层)。"""
    import types
    import torch
    from finetune.parts import rloo as R

    _, dec, theta, feats, _, _ = _fake_roma()
    base = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(0))

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(()))
            self.decoder = dec

    I = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    monkeypatch.setattr(R, "features", lambda model, a, b: feats())
    monkeypatch.setattr(R, "coarse", lambda model, f1, f2: (model.model.w * base, torch.zeros(1, 1, 4, 4)))
    monkeypatch.setattr(R, "fit_many", lambda flows, certs, num, thr, gen=None: [(I, 10)] * len(flows))
    monkeypatch.setattr(R.augment, "sample_T", lambda rng, *a, **k: np.c_[np.eye(2), [40.0, 0.0]])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    p = {**R.PARAMS, "reward": "ref", "group": 4, "temp": 0.3, "shape": "gauss", "fine_shape": "cauchy",
         "levels": [16, 8], "fine_mode": "local", "fine_group": 4, **over}
    model = types.SimpleNamespace(model=M())
    term = R.Term(p, model, None)
    img = torch.rand(1, 1, 64, 64) + 0.1
    optim = types.SimpleNamespace(backward=lambda l: l.backward(), factor=lambda: 1.0)
    st = types.SimpleNamespace(images=lambda: (img, img), batch={}, trace={}, optim=optim)
    return term, st, model.model.w, dec


def _fine_grad(dec, s):
    return sum(float(q.grad.abs().sum()) for q in dec.conv_refiner[str(s)].parameters() if q.grad is not None)


def test_rloo_term_fine_stops_when_mp_or_gate_fails(monkeypatch):
    """检查没过时细化也不学（#221）：多扰动核对没过（与图像无关的估计差 40 px > mp_thr）或参照合理性检查没过
    （面积倍数 1 不在 [1.5, 2] 内）时，粗级与训练的细化层都没有梯度；放宽时都有。未训练的级始终没有梯度。"""
    for over, stopped in (({"mp_n": 2}, True), ({"mp_n": 2, "mp_thr": 100.0}, False),
                          ({"gate": True, "gate_det_lo": 1.5}, True), ({"gate": True}, False)):
        term, st, w, dec = _fake_term_fine(monkeypatch, **over)
        loss, stats, _ = term(st)
        loss.backward()
        assert (float(w.grad) == 0.0) == stopped, over
        assert (_fine_grad(dec, 16) == 0.0) == stopped and (_fine_grad(dec, 8) == 0.0) == stopped, over
        assert _fine_grad(dec, 4) == 0.0
        assert "rew_16" in stats and "d_8" in stats


def test_rloo_term_fine_mask_bce_and_fine_w(monkeypatch):
    """fine_mask、fine_cert_bce（#221）：参照仿射把整张图移出画面时细化掩码全 0 —— fine_mask 下细化策略项没有梯度，
    加上 fine_cert_bce 后细化层又有梯度（来自 BCE），日志记各级画面内比例与 BCE；fine_w 按比例缩放细化损失。"""
    import torch
    from finetune.parts import rloo as R

    term, st, w, dec = _fake_term_fine(monkeypatch, fine_mask=True)
    _, stats, _ = term(st)
    assert stats["in_16"] == 1.0 and stats["in_8"] == 1.0 and _fine_grad(dec, 16) > 0     # 恒等仿射：全在画面内
    out = np.array([[1.0, 0, 600.0], [0, 1.0, 0]])                                          # 平移 600 px：全在画面外
    for over, has in (({"fine_mask": True}, False), ({"fine_mask": True, "fine_cert_bce": 0.01}, True)):
        term, st, w, dec = _fake_term_fine(monkeypatch, **over)
        monkeypatch.setattr(R, "fit_many", lambda flows, certs, num, thr, gen=None: [(out, 10)] * len(flows))
        _, stats, _ = term(st)
        assert stats["in_16"] == 0.0 and (_fine_grad(dec, 16) > 0) == has, over
        assert ("l_cert_16" in stats) == has
    vals = []
    for fw in (1.0, 0.25):
        term, st, w, dec = _fake_term_fine(monkeypatch, fine_w=fw)
        torch.manual_seed(3)
        _, stats, _ = term(st)
        vals.append((stats["l_fine"], _fine_grad(dec, 16)))
    assert vals[1][0] == pytest.approx(0.25 * vals[0][0], abs=1e-5)                     # 日志只留 5 位小数
    assert vals[1][1] == pytest.approx(0.25 * vals[0][1], rel=1e-3)


# ---------- 提前停（#221） ----------

def test_halt_window_median():
    """窗口满之前不停；中位数越线才停（单个尖峰不停）；缺这个量的步不进窗口；NaN 不触发。"""
    from finetune.train import Halt

    h = Halt({"d_ref": 100.0}, {"in_frame": 0.5}, 3)
    assert h({"d_ref": 500.0, "in_frame": 0.9}) is None and h({"d_ref": 50.0}) is None   # 窗口未满
    assert h({"d_ref": 50.0, "in_frame": 0.9}) is None                                   # 中位数 50，尖峰不停
    assert h({"d_ref": 150.0}) is None and h({}) is None
    hit = h({"d_ref": 150.0, "in_frame": 0.9})
    assert hit["key"] == "d_ref" and hit["op"] == ">" and hit["median"] == 150.0
    h = Halt({}, {"in_frame": 0.5}, 2)
    assert h({"in_frame": 0.4}) is None and h({"in_frame": 0.3})["key"] == "in_frame"
    h = Halt({"d_ref": 1.0}, {}, 1)
    assert h({"d_ref": float("nan")}) is None
    assert not Halt({}, {}, 50)


def test_config_run_halt(tmp_path):
    from finetune import config as C
    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n'))
    assert (d["run"]["halt_above"], d["run"]["halt_below"], d["run"]["halt_window"]) == ({}, {}, 50)  # 默认不停
    c = C.load(_write(tmp_path, "c", '[model]\nname = "roma"\n[run]\nhalt_above = {d_ref = 100}\n'
                                     'halt_below = {in_frame = 0.5}\nhalt_window = 200\n'))
    assert c["run"]["halt_above"] == {"d_ref": 100} and c["run"]["halt_window"] == 200
    for bad in ('halt_above = {d_ref = "x"}', 'halt_below = {in_frame = true}', 'halt_window = 0', 'halt_above = 3'):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", f'[model]\nname = "roma"\n[run]\n{bad}\n'))


def test_rloo_term_fine_bce_skipped_on_gate_but_kept_on_mp(monkeypatch):
    """细化 certainty 的 BCE 同粗级 cert_bce（#221 review）：参照合理性检查没过时不做（细化层没有梯度），
    多扰动核对没过时照常做（策略项置零，细化层的梯度只来自 BCE）。"""
    for over, has in (({"gate": True, "gate_det_lo": 1.5}, False), ({"mp_n": 2}, True)):
        term, st, w, dec = _fake_term_fine(monkeypatch, fine_mask=True, fine_cert_bce=0.01, **over)
        loss, stats, _ = term(st)
        loss.backward()
        assert float(w.grad) == 0.0 and (_fine_grad(dec, 16) > 0) == has, over
        assert ("l_cert_16" in stats) == has


# ---------- 多卡数据并行（#226） ----------

def test_halt_takes_all_records_of_an_update():
    """多卡时一次更新进 N 个记录（各卡各一对），全部进窗口后再判断一次。"""
    from finetune.train import Halt

    h = Halt({"d_ref": 100.0}, {}, 4)
    assert h({"d_ref": 150.0}, {"d_ref": 150.0}) is None          # 窗口 4 个记录，还差 2 个
    assert h({"d_ref": 50.0}, {"d_ref": 150.0})["median"] == 150.0
    h = Halt({"d_ref": 100.0}, {}, 4)
    assert h({"d_ref": 150.0}, {"d_ref": 150.0}) is None and h({"d_ref": 50.0}, {"d_ref": 50.0}) is None   # 中位数 100


def test_dist_mean_stats_and_single_process_noops():
    import torch
    from finetune import dist as D

    assert D.mean_stats([{"a": 1.0, "b": 2.0}, {"a": 3.0}]) == {"a": 2.0, "b": 2.0}   # 只对有这个键的卡平均
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.full((3,), 2.0)
    D.average_grads([p])
    assert D.world() == 1 and D.gather({"x": 1}) == [{"x": 1}] and D.max_diff([p]) == 0.0
    assert torch.equal(p.grad, torch.full((3,), 2.0))
    assert D.init(0) == 0


def test_dist_init_rejects_mismatched_process_count(monkeypatch):
    import pytest
    from finetune import dist as D

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="ddp = 0"):        # torchrun 起了 2 个进程，配置却是单卡
        D.init(0)
    with pytest.raises(RuntimeError, match="WORLD_SIZE = 2"):  # 配置 4 卡，torchrun 只起了 2 个
        D.init(4)


DDP_SCRIPT = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
import torch, torch.distributed as dist
from finetune import dist as D
from finetune.data import PairSet
dist.init_process_group("gloo")
r = dist.get_rank()
a = torch.nn.Parameter(torch.zeros(4))            # 两卡都有梯度
b = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))   # 只有 rank 0 有梯度（另一种 dtype，分桶）
c = torch.nn.Parameter(torch.zeros(3))            # 都没有
a.grad = torch.full((4,), float(r + 1))
if r == 0:
    b.grad = torch.full((2,), 4.0, dtype=torch.float64)
w = torch.full((2,), float(r))
D.broadcast([w])
d0 = D.max_diff([a, w])
a.data += r                                       # rank 1 的参数故意偏 1
out = {"w": w.tolist(), "d0": d0,
       "d1": D.max_diff([a]), "g": D.gather(r)}
D.average_grads([a, b, c])
out.update(a=a.grad.tolist(), b=b.grad.tolist(), c=c.grad is None)
ds = list(range(7))
smp = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=0)
from finetune.trace import Loader
dl = torch.utils.data.DataLoader(ds, batch_size=1, sampler=smp, generator=torch.Generator().manual_seed(r))
it = iter(Loader(dl, torch.Generator()))
out["ep"] = [[int(next(it)) for _ in range(4)] for _ in range(2)]
json.dump(out, open(sys.argv[2] + f"/r{r}.json", "w"))
dist.destroy_process_group()
'''


def test_dist_two_processes_average_and_sampler(tmp_path):
    """两个进程（gloo，CPU）：梯度取平均，某卡没有梯度的参数按 0 参与，全都没有的保持 None；广播、最大差、gather；
    DistributedSampler 各卡分到不同的图，Loader 每个 epoch 调 set_epoch 换顺序。"""
    import subprocess
    import sys
    (tmp_path / "w.py").write_text(DDP_SCRIPT, encoding="utf-8")
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    str(tmp_path / "w.py"), str(REPO), str(tmp_path)], check=True, timeout=300)
    o = [json.loads((tmp_path / f"r{i}.json").read_text()) for i in range(2)]
    for x in o:
        assert x["a"] == [1.5] * 4 and x["b"] == [2.0, 2.0] and x["c"] is True
        assert x["w"] == [0.0, 0.0] and x["d0"] == 0.0 and x["d1"] == 1.0 and x["g"] == [0, 1]
    e0, e1 = o[0]["ep"], o[1]["ep"]
    assert sorted(e0[0] + e1[0]) == sorted(list(range(7)) + [e0[0][0]])   # 7 张补齐到 8：各卡 4 张，第 1 张重复一次
    assert e0[0] != e0[1]                                            # 第二个 epoch 顺序不同


def test_config_ddp(tmp_path):
    from finetune import config as C
    base = '[model]\nname = "roma"\n'
    assert C.load(_write(tmp_path, "d", base))["run"]["ddp"] == 0
    assert C.load(_write(tmp_path, "c", base + "[run]\nddp = 4\n"))["run"]["ddp"] == 4
    for bad in ("ddp = -1", "ddp = 2\nsave_optim = true", "ddp = 2\nlog_update = true"):
        with pytest.raises(C.ConfigError):
            C.load(_write(tmp_path, "b", base + f"[run]\n{bad}\n"))
    with pytest.raises(C.ConfigError):
        C.load(_write(tmp_path, "b", base + '[run]\nddp = 2\n[rloo]\nreward = "ref"\ntrace = true\n'))


def test_optim_reduce_runs_before_update():
    """Optim 的 reduce 在更新前、对全部参数调用一次（多卡平均梯度的接口）；不是更新步时不调用。"""
    import torch
    from finetune.config import OPTIM
    from finetune.optim import Optim

    p = torch.nn.Parameter(torch.zeros(2))
    seen = []

    def reduce(ps):
        seen.append([float(q.grad[0]) for q in ps])
        for q in ps:
            q.grad.mul_(0.5)
    o = Optim([p], {**OPTIM, "accum": 2, "warmup": 0, "lr": 0.1}, amp=False, reduce=reduce)
    p.grad = torch.ones(2)
    o.step(1)
    assert seen == []
    o.step(2)
    assert seen == [[1.0]] and p.grad is None


# ---------- 单卡提速（#226）：与原写法算的是同一个东西 ----------

def test_roma_sdpa_forward_matches_attention():
    """sdpa_forward 与 DINOv2 Attention 原写法（q·d^-1/2 kᵀ → softmax → v）输出相同、梯度相同。"""
    import types
    import torch
    from finetune.models.roma import sdpa_forward, use_sdpa

    class Attention(torch.nn.Module):          # 照 romatch/models/transformer/layers/attention.py
        def __init__(self, dim=32, num_heads=4):
            super().__init__()
            self.num_heads, self.scale = num_heads, (dim // num_heads) ** -0.5
            self.qkv, self.proj = torch.nn.Linear(dim, dim * 3), torch.nn.Linear(dim, dim)
            self.attn_drop, self.proj_drop = torch.nn.Dropout(0.0), torch.nn.Dropout(0.0)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            attn = self.attn_drop((q @ k.transpose(-2, -1)).softmax(dim=-1))
            return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))

    torch.manual_seed(0)
    a = Attention().double()
    x = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
    y0 = a(x)
    g0 = torch.autograd.grad(y0.square().sum(), [x, a.qkv.weight])
    y1 = sdpa_forward(a, x)
    g1 = torch.autograd.grad(y1.square().sum(), [x, a.qkv.weight])
    assert torch.allclose(y0, y1) and all(torch.allclose(u, v) for u, v in zip(g0, g1))
    dv = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn=a)])
    use_sdpa(dv)
    assert Attention.forward is sdpa_forward
    a.attn_drop.p = 0.1
    with pytest.raises(ValueError):
        use_sdpa(dv)


def test_config_sdpa_default_off(tmp_path):
    from finetune import config as C
    d = C.load(_write(tmp_path, "d", '[model]\nname = "roma"\n[rloo]\nreward = "ref"\nref_ema = 0.999\n'))
    assert d["model"]["sdpa"] is False


def test_rloo_autocast_does_not_stringify_features(monkeypatch):
    """_autocast 不再把特征图转成字符串（#226）；返回的设置与 romatch get_autocast_params 只按设备给出的相同。"""
    import sys
    import types
    import torch
    from finetune.parts import rloo

    def gap(device=None, enabled=False, dtype=None):       # romatch.utils.utils.get_autocast_params 的行为：只看设备
        on = "cuda" in str(device)
        return str(device).split(":")[0] if on else "cpu", on, dtype if on else torch.bfloat16
    mod = types.ModuleType("romatch.utils.utils")
    mod.get_autocast_params = gap
    for name in ("romatch", "romatch.utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "romatch.utils.utils", mod)

    class NoStr(torch.Tensor):
        def __repr__(self):
            raise AssertionError("不应把特征图转成字符串")
        __str__ = __repr__
    x = torch.zeros(2).as_subclass(NoStr)
    ac, on = rloo._autocast(types.SimpleNamespace(amp_dtype=torch.float16), x)
    assert on is False


def test_roma_state_dict_keeps_init_lora():
    """从带 LoRA 的存档起步时（#238），新存档记下起点已合并进 DINOv2 的 LoRA；评测适配器照存档装回的 DINOv2
    与训练时的等效权重（起点合并后 + 本次 LoRA）相同。DINOv2 不在 state_dict 里，漏存起点的 LoRA 会让评测丢掉起点的改动。"""
    import types
    import torch
    from baselines.adapters.romatch import merge_lora, merge_saved
    from finetune.models.roma import LoRA, Model

    def dino():
        torch.manual_seed(0)
        d = torch.nn.Module()
        d.blocks = torch.nn.ModuleList([torch.nn.Module()])
        d.blocks[0].attn = torch.nn.Module()
        d.blocks[0].attn.qkv = torch.nn.Linear(4, 6)
        return d

    def lora(seed):
        g = torch.Generator().manual_seed(seed)
        return {"blocks.0.attn.qkv": {"A": torch.randn(2, 4, generator=g), "B": torch.randn(6, 2, generator=g), "scale": 2.0}}

    l0, l1 = lora(1), lora(2)
    base = dino()                                    # 训练时：适配器装入起点存档，合并起点的 LoRA，记下合并链
    adapter = types.SimpleNamespace(lora_merged=merge_saved(base, {"model": {}, "dinov2_lora": l0}))
    cur = LoRA(base.blocks[0].attn.qkv, 2, 4.0)      # 再挂一套新 LoRA，训练后取 l1 的值
    with torch.no_grad():
        cur.A.copy_(l1["blocks.0.attn.qkv"]["A"])
        cur.B.copy_(l1["blocks.0.attn.qkv"]["B"])
    cur.scale = 2.0
    m = types.SimpleNamespace(model=torch.nn.Linear(1, 1), lora={"blocks.0.attn.qkv": cur}, unc_heads=None,
                              dino_blocks={}, adapter=adapter)
    sd = Model.state_dict(m)
    assert len(sd["dinov2_lora_base"]) == 1
    ev = dino()                                      # 评测时：原始 DINOv2 + 存档
    assert len(merge_saved(ev, sd)) == 2
    x = torch.randn(3, 4)
    assert torch.allclose(ev.blocks[0].attn.qkv(x), cur(x), atol=1e-5)
    ref = dino()                                     # 等效权重 = W + 起点 LoRA + 本次 LoRA
    merge_lora(ref, l0)
    merge_lora(ref, l1)
    assert torch.allclose(ev.blocks[0].attn.qkv.weight, ref.blocks[0].attn.qkv.weight)
    no_init = types.SimpleNamespace(**{**vars(m), "adapter": types.SimpleNamespace(lora_merged=[])})
    assert "dinov2_lora_base" not in Model.state_dict(no_init)   # 不带 init 的存档与原来相同
