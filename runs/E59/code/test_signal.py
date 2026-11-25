"""Regression checks against RIPE's dense reference objective and Moon coordinates."""
from pathlib import Path
import sys
from subprocess import CompletedProcess

import numpy as np
import pytest
import torch

from runs.E59.code.predictions import prediction_location, run_record
from runs.E59.code.train import affine_mask, other_compute_process_present, pair_batches, reward_sum

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/RIPE"))
from ripe.utils.utils import get_rewards


def test_positive_negative_pairing_and_deterministic_resume():
    full = list(pair_batches(9, 6, 19, 0, 6))
    for epoch in (full[:3], full[3:]):
        pairs = [pair for batch in epoch for pair in batch]
        assert sorted(optical for optical, sar in pairs if optical == sar) == list(range(9))
        assert sorted(optical for optical, sar in pairs if optical != sar) == list(range(9))
        assert any(optical != sar and optical // 3 == sar // 3 for optical, sar in pairs)
        assert all(0 <= sar < 9 for _, sar in pairs)
    assert list(pair_batches(9, 6, 19, 2, 6)) == full[2:]


@pytest.mark.parametrize("pids,shared", [("7311\n", False), ("7311\n9876\n", True)])
def test_gpu_sharing_detection(monkeypatch, pids, shared):
    from runs.E59.code import train

    def query(args, **kwargs):
        assert args[:3] == ["nvidia-smi", "-i", "1"]
        return CompletedProcess(args, 0, pids, "")

    monkeypatch.setattr(train.subprocess, "run", query)
    assert other_compute_process_present("1") == shared


@pytest.mark.parametrize("update,method", [(0, "init"), (1000, "step1000"), (20000, "main")])
def test_predictions_can_be_reopened_with_the_workbench_reader(tmp_path, update, method):
    run_dir = tmp_path / "E59"
    directory, name = prediction_location(run_dir, update, 20000)
    assert name == method
    out = directory / "preds" / name
    out.mkdir(parents=True)
    (directory / "exp.toml").write_text(f'id = "{directory.name}"\ntitle = "RIPE"\n', encoding="utf-8")
    (out / "val.jsonl").write_text('{"pair":"ROI_001/patch_1","A":null,"fail":"few_matches"}\n', encoding="utf-8")
    assert run_record(directory).preds(name, "val")["ROI_001/patch_1"].fail == "few_matches"


def test_affine_mask_uses_original_pixel_threshold(monkeypatch):
    from runs.E59.code import train

    original = np.array([[10, 5], [98, 3], [32, 90], [256, 233], [2, 120], [71, 113], [43, 90], [73, 154]])
    target = original @ np.array([[1.1, 0.1], [-0.2, 0.9]]).T + [7, -8]
    target[-1] += [40, -30]
    fit = train.fit_affine

    def assert_grid(matches, threshold):
        np.testing.assert_allclose(matches[:, :2], original, atol=5e-5)
        np.testing.assert_allclose(matches[:, 2:4], target, atol=5e-5)
        assert threshold == 3.0
        return fit(matches, threshold)

    monkeypatch.setattr(train, "fit_affine", assert_grid)
    resized = lambda points: torch.from_numpy((points + 0.5) * 560 / 512 - 0.5)
    transform, inliers = affine_mask(resized(original), resized(target), 3.0, 512, 560)
    np.testing.assert_allclose(transform[:, :2], [[1.1, 0.1], [-0.2, 0.9]], atol=1e-5)
    assert inliers.tolist() == [True] * 7 + [False]


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("inlier_bits", [[True, False, True], [False, False, False]])
def test_sparse_objective_equals_upstream_dense_value_and_gradient(positive, inlier_bits):
    log0 = torch.tensor([-0.125, -0.8, -2.0, -3.0], requires_grad=True)
    log1 = torch.tensor([-0.5, -1.2, -2.3], requires_grad=True)
    kps0, kps1 = torch.zeros(4, 2), torch.zeros(3, 2)
    all0, all1 = torch.ones(4, dtype=torch.bool), torch.ones(3, dtype=torch.bool)
    matches = torch.tensor([[0, 2], [1, 1], [3, 0]])
    inliers = torch.tensor(inlier_bits)
    penalty = -5e-8
    reward = get_rewards(1.0, kps0, kps1, all0, all1, all0, all1, matches, matches,
                         inliers, positive, penalty, filter_mode="ignore")
    dense = (reward * (log0[:, None] + log1[None, :])).sum()
    sparse = reward_sum(log0, log1, matches, inliers, positive, penalty)
    torch.testing.assert_close(sparse, dense)
    for compact, reference in zip(torch.autograd.grad(sparse, (log0, log1)), torch.autograd.grad(dense, (log0, log1))):
        torch.testing.assert_close(compact, reference)
