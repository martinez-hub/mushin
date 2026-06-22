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
    import matplotlib

    matplotlib.use("Agg")  # headless backend for CI
    import sweep_to_dataset as ex

    ex.main()
    assert Path("sweep_accuracy.png").exists()
