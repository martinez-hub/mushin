import torch
from torch.utils.data import DataLoader, TensorDataset

from mushin.benchmark import BenchmarkResult, compare


def _loader(seed, n=64, d=4, num_classes=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


class _Perfect(torch.nn.Module):
    """Cheats: reads the target off a fixed mapping -> always correct."""

    def __init__(self, loader, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self._map = {
            tuple(x.tolist()): int(y) for xb, yb in loader for x, y in zip(xb, yb)
        }

    def forward(self, x):
        idx = torch.tensor([self._map[tuple(row.tolist())] for row in x])
        return torch.nn.functional.one_hot(idx, self.num_classes).float() * 10.0


def test_compare_end_to_end():
    data = _loader(seed=0)
    good = [_Perfect(data) for _ in range(3)]
    bad = [torch.nn.Linear(4, 3) for _ in range(3)]

    result = compare(
        methods={"good": good, "bad": bad},
        data=data,
        task="classification",
        num_classes=3,
        test="welch",
    )

    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert result.data.sizes == {"method": 2, "seed": 3}
    assert "accuracy" in result.data.data_vars
    assert float(result.data["accuracy"].sel({"method": "good"}).mean()) == 1.0
    assert len(result.summary()) == 2 * len(result.data.data_vars)


def test_compare_rejects_unknown_task():
    import pytest

    with pytest.raises(NotImplementedError):
        compare(methods={"a": []}, data=[], task="regression", num_classes=2)
