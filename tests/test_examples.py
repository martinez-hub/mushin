from pathlib import Path

import pytest


@pytest.mark.usefixtures("cleandir")
def test_build_dataset_returns_labeled_grid():
    import sweep_to_dataset as ex

    ds = ex.build_dataset()

    # dims are the swept parameters; data var is the returned metric
    assert set(ds.dims) == {"lr", "seed"}
    assert ds.sizes == {"lr": len(ex.LEARNING_RATES), "seed": len(ex.SEEDS)}
    assert "accuracy" in ds.data_vars
    # accuracy is a probability in [0, 1]
    assert float(ds["accuracy"].min()) >= 0.0
    assert float(ds["accuracy"].max()) <= 1.0


@pytest.mark.usefixtures("cleandir")
def test_main_writes_plot():
    import sweep_to_dataset as ex

    ex.main()
    assert Path("sweep_accuracy.png").exists()


def test_compare_classifiers_example_runs_on_synthetic():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from compare_classifiers import run
    from mushin.benchmark import BenchmarkResult

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (32,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    result = run(loader, loader, seeds=(0, 1))
    assert isinstance(result, BenchmarkResult)
    assert result.data.sizes["seed"] == 2
    assert "accuracy" in result.data.data_vars
