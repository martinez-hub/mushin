import warnings

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mushin.benchmark import BenchmarkResult
from mushin.study._load import evaluate_checkpoints, warn_if_underpowered


def _loader(n=40, d=4, num_classes=3):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


def _save_models(tmp_path, names, n_seeds=2, d=4, num_classes=3):
    checkpoints = {}
    for name in names:
        paths = []
        for s in range(n_seeds):
            torch.manual_seed(hash((name, s)) % 1000)
            model = torch.nn.Linear(d, num_classes)
            p = tmp_path / f"{name}_{s}.pt"
            torch.save(model, p)
            paths.append(str(p))
        checkpoints[name] = paths
    return checkpoints


def test_evaluate_checkpoints_returns_benchmark_result(tmp_path):
    checkpoints = _save_models(tmp_path, ["m1", "m2"])
    result = evaluate_checkpoints(
        checkpoints,
        load_fn=lambda p: torch.load(p, weights_only=False),
        data=_loader(),
        task="classification",
        num_classes=3,
        test="welch",
    )
    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert result.data.sizes == {"method": 2, "seed": 2}
    assert "accuracy" in result.data.data_vars


def test_warn_if_underpowered_fires_for_small_n_wilcoxon():
    with pytest.warns(UserWarning, match="cannot reach"):
        warn_if_underpowered("wilcoxon", n_seeds=3, alpha=0.05)


def test_warn_if_underpowered_silent_for_welch_small_n():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> failure
        warn_if_underpowered("welch", n_seeds=3, alpha=0.05)


def test_evaluate_checkpoints_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_checkpoints(
            {}, load_fn=lambda p: p, data=None, task="classification", num_classes=2
        )
