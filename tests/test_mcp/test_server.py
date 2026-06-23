# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mushin.mcp.server import RootError, _resolve, _to_jsonable


def test_to_jsonable_scalar_tensor():
    assert _to_jsonable(torch.tensor(0.5)) == 0.5


def test_to_jsonable_tensor_array():
    assert _to_jsonable(torch.tensor([1.0, 2.0])) == [1.0, 2.0]


def test_to_jsonable_numpy_and_nested():
    out = _to_jsonable({"a": np.float32(1.5), "b": [np.int64(2)]})
    assert out == {"a": 1.5, "b": [2]}


def test_to_jsonable_omegaconf():
    cfg = OmegaConf.create({"lr": 0.1, "nested": {"seed": 0}})
    assert _to_jsonable(cfg) == {"lr": 0.1, "nested": {"seed": 0}}


def test_to_jsonable_non_finite_float_becomes_string():
    assert _to_jsonable(float("inf")) == "inf"


def test_resolve_no_root_returns_absolute(tmp_path):
    target = tmp_path / "exp"
    target.mkdir()
    assert _resolve(target, None) == target.resolve()


def test_resolve_inside_root_ok(tmp_path):
    root = tmp_path
    target = tmp_path / "exp"
    target.mkdir()
    assert _resolve(target, root) == target.resolve()


def test_resolve_outside_root_raises(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(RootError):
        _resolve(outside, root)


def _make_experiment(base: Path, lrs=(0.1, 0.2)) -> Path:
    """Build a minimal 2-run Hydra multirun layout under ``base``."""
    for i, lr in enumerate(lrs):
        run = base / str(i)
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create({"lr": lr, "seed": 0}),
            run / ".hydra" / "config.yaml",
        )
        torch.save({"accuracy": torch.tensor(0.8 + 0.1 * i)}, run / "metrics.pt")
    return base


def test_list_experiments_finds_runs(tmp_path):
    from mushin.mcp.server import _list_experiments

    base = _make_experiment(tmp_path / "exp")
    out = _list_experiments(base)
    assert out["count"] == 2
    assert sorted(Path(r).name for r in out["runs"]) == ["0", "1"]


def test_describe_experiment_reports_sweep(tmp_path):
    from mushin.mcp.server import _describe_experiment

    base = _make_experiment(tmp_path / "exp")
    out = _describe_experiment(base)
    assert out["num_runs"] == 2
    assert "metrics" in out["metric_keys"]
    assert out["swept_params"]["lr"] == [0.1, 0.2]
    assert "seed" not in out["swept_params"]  # constant across runs


def test_get_metrics_per_run_and_reduce(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    out = _get_metrics(base, reduce="mean")
    assert out["num_runs"] == 2
    # metrics saved as metrics.pt -> {"metrics": {"accuracy": ...}}
    assert out["per_run"][0]["metrics"]["accuracy"] == pytest.approx(0.8, abs=1e-5)
    assert out["reduced"]["metrics.accuracy"] == pytest.approx(0.85, abs=1e-5)


def test_get_metrics_filter(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    out = _get_metrics(base, metrics=["does-not-exist"])
    assert out["per_run"][0] == {}


def test_get_config_returns_configs(tmp_path):
    from mushin.mcp.server import _get_config

    base = _make_experiment(tmp_path / "exp")
    out = _get_config(base)
    assert [c["lr"] for c in out["configs"]] == [0.1, 0.2]


def test_get_config_single_job(tmp_path):
    from mushin.mcp.server import _get_config

    base = _make_experiment(tmp_path / "exp")
    out = _get_config(base, job=1)
    assert out["config"]["lr"] == 0.2


def test_read_dataset_summarizes(tmp_path):
    import xarray as xr

    from mushin.mcp.server import _read_dataset

    ds = xr.Dataset(
        {"accuracy": ("lr", [0.8, 0.9])},
        coords={"lr": [0.1, 0.2]},
    )
    nc = tmp_path / "result.nc"
    ds.to_netcdf(nc, engine="scipy")

    out = _read_dataset(nc)
    assert out["dims"] == {"lr": 2}
    assert out["coords"]["lr"] == [0.1, 0.2]
    assert out["data_vars"]["accuracy"]["max"] == pytest.approx(0.9)


def test_describe_outside_root_raises(tmp_path):
    from mushin.mcp.server import RootError, _describe_experiment

    _make_experiment(tmp_path / "allowed" / "exp")
    with pytest.raises(RootError):
        _describe_experiment(tmp_path / "elsewhere", root=tmp_path / "allowed")


def test_create_server_registers_tools():
    pytest.importorskip("mcp")  # mcp requires Python >= 3.10
    from mushin.mcp.server import create_server

    server = create_server(root=None)
    assert server.name == "mushin"


def test_main_builds_server_without_running(monkeypatch):
    pytest.importorskip("mcp")
    import mushin.mcp.__main__ as cli

    captured = {}

    class _FakeServer:
        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(cli, "create_server", lambda root: _FakeServer())
    cli.main(["--root", "."])
    assert captured["ran"] is True


# Fix A tests
def test_list_experiments_outside_root_raises(tmp_path):
    from mushin.mcp.server import RootError, _list_experiments

    _make_experiment(tmp_path / "allowed" / "exp")
    with pytest.raises(RootError):
        _list_experiments(tmp_path / "elsewhere", root=tmp_path / "allowed")


def test_list_experiments_defaults_base_to_root(tmp_path):
    from mushin.mcp.server import _list_experiments

    _make_experiment(tmp_path / "exp")
    out = _list_experiments(None, root=tmp_path)
    assert out["count"] == 2


# Fix B test
def test_describe_missing_experiment_raises(tmp_path):
    from mushin.mcp.server import _describe_experiment

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _describe_experiment(empty)


# Fix C test
def test_get_config_job_out_of_range_raises(tmp_path):
    from mushin.mcp.server import _get_config

    base = _make_experiment(tmp_path / "exp")
    with pytest.raises(ValueError):
        _get_config(base, job=5)


# Fix D tests
def test_get_metrics_reduce_std(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    out = _get_metrics(base, reduce="std")
    # population std of [0.8, 0.9]
    assert out["reduced"]["metrics.accuracy"] == pytest.approx(0.05, abs=1e-4)


def test_get_config_single_run(tmp_path):
    from mushin.mcp.server import _get_config

    base = _make_experiment(tmp_path / "exp", lrs=(0.3,))
    out = _get_config(base)
    assert out["config"]["lr"] == 0.3
    assert "configs" not in out


def test_metrics_loaded_weights_only(tmp_path, monkeypatch):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    seen = {}
    real_load = torch.load

    def spy(*args, **kwargs):
        seen["weights_only"] = kwargs.get("weights_only")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    _get_metrics(base)
    assert seen["weights_only"] is True


def test_unreadable_metrics_skipped(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    (base / "0" / "bad.pt").write_bytes(b"not a real torch file")
    out = _get_metrics(base)  # must not raise
    assert "metrics" in out["per_run"][0]  # good file still loaded
    assert "bad" not in out["per_run"][0]  # unreadable file skipped, not executed


def test_get_metrics_filter_by_leaf(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    out = _get_metrics(base, metrics=["accuracy"])
    assert out["per_run"][0] == {"metrics.accuracy": pytest.approx(0.8, abs=1e-5)}
