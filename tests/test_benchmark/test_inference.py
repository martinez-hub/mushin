import torch
from torch.utils.data import DataLoader, TensorDataset

from mushin.benchmark._inference import run_inference


def _loader(n=10, d=4):
    x = torch.randn(n, d)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_run_inference_collects_full_dataset():
    model = torch.nn.Linear(4, 3)
    data = _loader(n=10)

    preds, probs, targets = run_inference(model, data)

    assert preds.shape == (10,)
    assert probs.shape == (10, 3)
    assert targets.shape == (10,)
    expected = torch.cat([y for _, y in data])
    assert torch.equal(targets, expected)
