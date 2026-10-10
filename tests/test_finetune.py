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
