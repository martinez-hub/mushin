from mushin._sweep_io import combo_key, read_metrics_sidecar, write_metrics_sidecar


def test_combo_key_is_canonical_and_order_stable():
    assert combo_key({"lr": 0.1, "seed": 2}) == combo_key({"seed": 2, "lr": 0.1})
    assert combo_key({"lr": 0.1, "seed": 2}) == "lr=0.1,seed=2"


def test_metrics_sidecar_roundtrip(tmp_path):
    write_metrics_sidecar(tmp_path, {"accuracy": 0.9, "loss": 0.1})
    assert read_metrics_sidecar(tmp_path) == {"accuracy": 0.9, "loss": 0.1}
    assert (tmp_path / "mushin_metrics.json").exists()
    assert read_metrics_sidecar(tmp_path / "nope") is None  # absent -> None


def test_metrics_sidecar_coerces_numpy_and_tensors(tmp_path):
    import numpy as np

    write_metrics_sidecar(tmp_path, {"a": np.float32(0.5), "b": np.array([1, 2])})
    got = read_metrics_sidecar(tmp_path)
    assert got == {"a": 0.5, "b": [1, 2]}
