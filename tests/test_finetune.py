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
    p = {"delta": 8.0, "tau": 16.0, "r_out": -0.1, "shape": "gauss"}
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
    from finetune.models.roma import _refine_part

    assert _refine_part("decoder.conv_refiner.16.block1.0.weight") == "16"
    assert _refine_part("decoder.conv_refiner.4.out_conv.bias") == "4"
    assert _refine_part("decoder.proj.8.0.weight") == "8"
    for name in ("decoder.proj.16.0.weight", "decoder.gps.16.foo", "decoder.embedding_decoder.x", "encoder.cnn.layers.0"):
        assert _refine_part(name) is None


def test_update_monitor_relative_change():
    """普通参数：‖Δθ‖/‖θ‖；LoRA：‖scale·Δ(BA)‖/‖W‖，与直接算等效权重的结果相同；fine 合并 ref* 与 vgg。"""
    import torch
    from finetune.optim import UpdateMonitor

    torch.manual_seed(0)
    a, v = torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.full((2,), 2.0))
    c = torch.nn.Parameter(torch.ones(3))
    A, B, W = torch.randn(2, 5), torch.randn(6, 2), torch.randn(6, 5)
    mon = UpdateMonitor({"ref8": [a], "vgg": [v], "coarse": [c]}, [(A, B, 2.0, W)])
    mon.before()
    BA0 = (B @ A).clone()
    with torch.no_grad():
        a += 0.1
        v -= 0.2
        A += 0.05 * torch.randn_like(A)
        B += 0.05 * torch.randn_like(B)
    out = mon.after()
    assert out["upd_coarse"] == 0.0
    assert out["upd_ref8"] == pytest.approx(0.1, rel=1e-3)
    assert out["upd_vgg"] == pytest.approx(0.1, rel=1e-3)
    assert out["upd_fine"] == pytest.approx(float(np.sqrt((4 * 0.01 + 2 * 0.04) / (4 + 8))), rel=1e-3)
    want = float((2.0 * (B @ A - BA0)).norm() / W.norm())
    assert out["upd_lora"] == pytest.approx(want, rel=1e-3)
