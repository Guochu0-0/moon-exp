import pickle

import torch

from finetune.data import collate_for_pipe


def test_pipe_transport_preserves_dataset_batch_values_and_types():
    samples = [
        {
            "pair": f"ROI_000/patch_{index}",
            "image0": torch.arange(15, dtype=torch.float32).reshape(1, 3, 5) / (index + 16),
            "image1": torch.ones(1, 3, 5, dtype=torch.float32) * index,
            "T": torch.arange(6, dtype=torch.float64).reshape(2, 3) + index,
            "occ": torch.arange(15, dtype=torch.uint8).reshape(1, 3, 5) % 2,
        }
        for index in range(2)
    ]
    reference = torch.utils.data.default_collate(samples)
    worker_payload = pickle.loads(pickle.dumps(collate_for_pipe(samples)))
    actual = torch.utils.data.default_collate(worker_payload)
    assert actual.pop("pair") == reference.pop("pair")
    for key in reference:
        assert actual[key].dtype == reference[key].dtype
        assert torch.equal(actual[key], reference[key])
