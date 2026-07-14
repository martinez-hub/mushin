from mushin._sweep_io import (
    Manifest,
    combo_key,
    read_metrics_sidecar,
    write_metrics_sidecar,
)


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


def test_manifest_tracks_and_replaces_cells(tmp_path):
    m = Manifest.load_or_new(tmp_path, params=["lr", "seed"])
    m.mark({"lr": 0.1, "seed": 0}, dir="0", status="completed")
    m.mark({"lr": 0.1, "seed": 2}, dir="5", status="failed", error="OOM")
    m.save()

    m2 = Manifest.load_or_new(tmp_path, params=["lr", "seed"])
    assert m2.status({"lr": 0.1, "seed": 0}) == "completed"
    assert m2.status({"lr": 0.1, "seed": 2}) == "failed"
    assert m2.status({"lr": 1.0, "seed": 0}) == "pending"  # unseen -> pending
    # a re-run REPLACES in place (no duplicate entry)
    m2.mark({"lr": 0.1, "seed": 2}, dir="9", status="completed")
    assert m2.status({"lr": 0.1, "seed": 2}) == "completed"
    assert m2.dir({"lr": 0.1, "seed": 2}) == "9"
    assert m2.failed_cells() == []
    m2.mark({"lr": 1.0, "seed": 1}, dir="3", status="failed", error="boom")
    assert not m2.is_complete()
    assert {"lr=1.0,seed=1"} == {c["key"] for c in m2.failed_cells()}
