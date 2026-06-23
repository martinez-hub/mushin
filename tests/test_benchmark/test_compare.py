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


def test_compare_default_test_with_tied_methods():
    # two perfect methods => identical metrics => default wilcoxon must not crash
    data = _loader(seed=0)
    a = [_Perfect(data) for _ in range(3)]
    b = [_Perfect(data) for _ in range(3)]
    result = compare(methods={"a": a, "b": b}, data=data, num_classes=3)
    assert isinstance(result, BenchmarkResult)
    # all comparisons are between identical methods -> none significant
    assert not result.comparisons["significant"].any()


def test_compare_segmentation_end_to_end():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 1, 8, 8, generator=g)
    masks = torch.randint(0, 3, (12, 8, 8), generator=g)
    loader = DataLoader(TensorDataset(x, masks), batch_size=4)

    class Perfect(torch.nn.Module):
        def __init__(self, masks):
            super().__init__()
            self._m = {tuple(xi.flatten().tolist()): mi for xi, mi in zip(x, masks)}

        def forward(self, xb):
            out = []
            for xi in xb:
                m = self._m[tuple(xi.flatten().tolist())]
                out.append(
                    torch.nn.functional.one_hot(m, 3).permute(2, 0, 1).float() * 10
                )
            return torch.stack(out)

    class Bad(torch.nn.Module):
        def forward(self, xb):
            return torch.zeros(xb.shape[0], 3, 8, 8)

    result = compare(
        methods={
            "good": [Perfect(masks) for _ in range(3)],
            "bad": [Bad() for _ in range(3)],
        },
        data=loader,
        task="segmentation",
        num_classes=3,
        test="welch",
    )
    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert "miou" in result.data.data_vars
    assert float(result.data["miou"].sel({"method": "good"}).mean()) == 1.0


def test_compare_rejects_unknown_task():
    import pytest

    from mushin.benchmark import compare

    with pytest.raises(NotImplementedError, match="not supported"):
        compare(methods={"a": []}, data=[], task="detection", num_classes=2)
