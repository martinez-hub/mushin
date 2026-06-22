from mushin.benchmark._aggregate import to_dataset


def test_to_dataset_shape_and_values():
    results = {
        "ours": [{"accuracy": 0.9, "f1": 0.8}, {"accuracy": 0.92, "f1": 0.81}],
        "base": [{"accuracy": 0.7, "f1": 0.6}, {"accuracy": 0.72, "f1": 0.61}],
    }
    ds = to_dataset(results)

    assert set(ds.dims) == {"method", "seed"}
    assert ds.sizes == {"method": 2, "seed": 2}
    assert set(ds.data_vars) == {"accuracy", "f1"}
    assert list(ds["method"].values) == ["ours", "base"]
    assert float(ds["accuracy"].sel(method="ours").isel(seed=0)) == 0.9
