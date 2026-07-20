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


def test_resolve_relative_path_under_root(tmp_path, monkeypatch):
    """A relative path with a configured root resolves under root, not the CWD."""
    root = tmp_path / "outputs"
    (root / "exp").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # server launched from a different directory
    assert _resolve("exp", root) == (root / "exp").resolve()


def _make_jobs(base: Path, n: int) -> Path:
    """Build ``n`` Hydra job dirs named 0..n-1, each with config lr=float(i)."""
    for i in range(n):
        run = base / str(i)
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create({"lr": float(i)}), run / ".hydra" / "config.yaml"
        )
    return base


def _make_experiment(base: Path, lrs=(0.1, 0.2)) -> Path:
    """Build a minimal 2-run Hydra multirun layout under ``base``."""
    for i, lr in enumerate(lrs):
        run = base / str(i)
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create({"lr": lr, "seed": 0}),
            run / ".hydra" / "config.yaml",
        )
        torch.save({"accuracy": 0.8 + 0.1 * i}, run / "metrics.pt")
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
    assert "metrics.accuracy" in out["metric_keys"]
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


def test_weights_only_used_only_when_safe(tmp_path, monkeypatch):
    """torch.load(weights_only) is used only on torch >= 2.6 (CVE-2025-32434)."""
    from mushin.mcp import server

    base = _make_experiment(tmp_path / "exp")
    seen = []
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    server._get_metrics(base)
    if server._TORCH_WEIGHTS_ONLY_SAFE:
        assert seen and all(w is True for w in seen)  # only ever weights_only=True
    else:
        assert seen == []  # never invoke torch.load on CVE-affected torch


def test_unreadable_metrics_skipped(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    (base / "0" / "bad_metrics.pt").write_bytes(b"not a real torch file")
    out = _get_metrics(base)  # must not raise
    assert "metrics" in out["per_run"][0]  # good file still loaded
    assert (
        "bad_metrics" not in out["per_run"][0]
    )  # unreadable file skipped, not executed


def test_get_metrics_filter_by_leaf(tmp_path):
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    out = _get_metrics(base, metrics=["accuracy"])
    assert out["per_run"][0] == {"metrics.accuracy": pytest.approx(0.8, abs=1e-5)}


def test_metrics_defaultdict_loaded(tmp_path):
    """MetricsCallback saves defaultdict(list); the safe loader must read it."""
    import collections

    from mushin.mcp.server import _get_metrics

    base = tmp_path / "exp" / "0"
    (base / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"lr": 0.1}), base / ".hydra" / "config.yaml")
    metrics = collections.defaultdict(list)
    metrics["accuracy"] = [0.8, 0.9]
    metrics["per_class"] = [np.array([0.1, 0.2])]
    torch.save(metrics, base / "fit_metrics.pt")

    out = _get_metrics(tmp_path / "exp")
    assert out["per_run"][0]["fit_metrics"]["accuracy"] == [0.8, 0.9]


def test_malicious_metrics_not_executed(tmp_path):
    """A metrics file whose unpickling would run code must be skipped, not run."""
    import os

    from mushin.mcp.server import _get_metrics

    marker = tmp_path / "pwned"

    class _Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    base = tmp_path / "exp" / "0"
    (base / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"lr": 0.1}), base / ".hydra" / "config.yaml")
    torch.save({"accuracy": 0.8}, base / "metrics.pt")  # good file
    torch.save({"x": _Evil()}, base / "evil_metrics.pt")  # malicious file

    out = _get_metrics(tmp_path / "exp")  # must not raise, must not execute
    assert not marker.exists()  # code never ran
    assert "metrics" in out["per_run"][0]  # good file still loaded
    assert "evil_metrics" not in out["per_run"][0]  # malicious file skipped


def test_ddp_config_loaded(tmp_path):
    """A run with both .hydra and .pl_hydra_rank_* configs must load .hydra one."""
    from mushin.mcp.server import _get_config

    base = tmp_path / "exp" / "0"
    (base / ".hydra").mkdir(parents=True)
    (base / ".pl_hydra_rank_1").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"lr": 0.1}), base / ".hydra" / "config.yaml")
    OmegaConf.save(
        OmegaConf.create({"pl_testing": True}),
        base / ".pl_hydra_rank_1" / "config.yaml",
    )

    out = _get_config(tmp_path / "exp")
    assert out["config"]["lr"] == 0.1  # not None, and the .hydra config


def test_job_dirs_sorted_numerically(tmp_path):
    from mushin.mcp.server import _get_config

    base = _make_jobs(tmp_path / "exp", 11)  # jobs 0..10
    assert _get_config(base, job=10)["config"]["lr"] == 10.0
    assert _get_config(base, job=2)["config"]["lr"] == 2.0


def test_metric_symlink_outside_root_skipped(tmp_path):
    from mushin.mcp.server import _get_metrics

    root = tmp_path / "allowed"
    exp = root / "exp" / "0"
    (exp / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"lr": 0.1}), exp / ".hydra" / "config.yaml")
    torch.save({"accuracy": 0.8}, exp / "metrics.pt")  # in-root
    secret = tmp_path / "outside.pt"
    torch.save({"secret": torch.tensor(42.0)}, secret)
    (exp / "leak_metrics.pt").symlink_to(secret)  # symlink escaping root

    out = _get_metrics(root / "exp", root=root)
    assert "metrics" in out["per_run"][0]  # in-root metric still read
    assert "leak_metrics" not in out["per_run"][0]  # escaping symlink refused


def test_tensor_metrics_skipped_on_unsafe_torch(tmp_path):
    """A raw-tensor metrics file is skipped (never unsafely loaded) on old torch."""
    from mushin.mcp import server

    base = tmp_path / "exp" / "0"
    (base / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"lr": 0.1}), base / ".hydra" / "config.yaml")
    torch.save({"w": torch.tensor([1.0, 2.0])}, base / "tensor_metrics.pt")

    out = server._get_metrics(tmp_path / "exp")  # must not raise
    if server._TORCH_WEIGHTS_ONLY_SAFE:
        assert out["per_run"][0]["tensor_metrics"]["w"] == [1.0, 2.0]
    else:
        assert "tensor_metrics" not in out["per_run"][0]  # safely skipped


def test_non_metric_pt_files_ignored(tmp_path):
    """model.pt / state_dict.pt must NOT be loaded as metrics."""
    from mushin.mcp.server import _get_metrics

    base = tmp_path / "exp" / "0"
    (base / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"lr": 0.1}), base / ".hydra" / "config.yaml")
    torch.save({"accuracy": 0.8}, base / "fit_metrics.pt")  # a metric file
    torch.save({"layers": [1, 2, 3]}, base / "model.pt")  # model weights

    out = _get_metrics(tmp_path / "exp")
    assert "fit_metrics" in out["per_run"][0]
    assert "model" not in out["per_run"][0]


def test_to_jsonable_non_finite_in_array():
    out = _to_jsonable(np.array([1.0, np.inf, np.nan]))
    assert out == [1.0, "inf", "nan"]


def test_to_jsonable_non_finite_in_tensor():
    out = _to_jsonable(torch.tensor([1.0, float("inf")]))
    assert out == [1.0, "inf"]


def test_read_dataset_non_finite_stats(tmp_path):
    import xarray as xr

    from mushin.mcp.server import _read_dataset

    # All-NaN so xarray's skipna mean is also NaN (not skipped-to-finite).
    ds = xr.Dataset({"m": ("x", [float("nan"), float("nan")])}, coords={"x": [0, 1]})
    nc = tmp_path / "d.nc"
    ds.to_netcdf(nc, engine="scipy")

    out = _read_dataset(nc)
    assert out["data_vars"]["m"]["mean"] == "nan"  # non-finite normalized to string


def test_env_interpolation_not_resolved(tmp_path, monkeypatch):
    """A ${oc.env:...} interpolation in a config must NOT be resolved/leaked."""
    from mushin.mcp.server import _get_config

    monkeypatch.setenv("MUSHIN_TEST_SECRET", "topsecret")
    base = tmp_path / "exp" / "0"
    (base / ".hydra").mkdir(parents=True)
    (base / ".hydra" / "config.yaml").write_text(
        "token: ${oc.env:MUSHIN_TEST_SECRET}\n"
    )

    out = _get_config(base)
    assert "topsecret" not in str(out)  # secret never resolved
    assert out["config"]["token"] == "${oc.env:MUSHIN_TEST_SECRET}"  # kept raw


def test_job_order_uses_hydra_job_num(tmp_path):
    """Non-numeric run dir names must still order by recorded hydra.job.num."""
    from mushin.mcp.server import _get_config

    base = tmp_path / "exp"
    # name "zzz" is job 0, "aaa" is job 1 — lexicographic name order would swap them
    for name, num, lr in [("zzz", 0, 0.1), ("aaa", 1, 0.2)]:
        run = base / name
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(OmegaConf.create({"lr": lr}), run / ".hydra" / "config.yaml")
        OmegaConf.save(
            OmegaConf.create({"hydra": {"job": {"num": num}}}),
            run / ".hydra" / "hydra.yaml",
        )

    assert _get_config(base, job=0)["config"]["lr"] == 0.1  # job.num 0 -> "zzz"
    assert _get_config(base, job=1)["config"]["lr"] == 0.2  # job.num 1 -> "aaa"


def test_server_root_defaults_to_cwd(tmp_path, monkeypatch):
    """No --root confines the server to the current directory, not the whole FS."""
    from mushin.mcp.server import _server_root

    monkeypatch.chdir(tmp_path)
    assert _server_root(None) == tmp_path.resolve()
    assert _server_root(tmp_path / "x") == (tmp_path / "x").resolve()


def test_job_sort_key_ignores_out_of_root_hydra_yaml(tmp_path):
    """A symlinked .hydra/hydra.yaml pointing outside --root must not be read."""
    from mushin.mcp.server import _job_sort_key

    root = tmp_path / "allowed"
    run = root / "exp" / "7"  # numeric dir name -> fallback key (0, 7, "")
    (run / ".hydra").mkdir(parents=True)
    outside = tmp_path / "evil_hydra.yaml"
    OmegaConf.save(OmegaConf.create({"hydra": {"job": {"num": 0}}}), outside)
    (run / ".hydra" / "hydra.yaml").symlink_to(outside)

    # If the outside file were read, the key would be (0, 0, "") (job.num 0).
    assert _job_sort_key(run / ".hydra", root.resolve()) == (0, 7, "")


def test_legacy_nonzip_metrics_not_misread(tmp_path):
    """A legacy non-zip torch.save file must be skipped, never read as garbage."""
    import zipfile

    from mushin.mcp import server

    base = tmp_path / "exp" / "0"
    (base / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"lr": 0.1}), base / ".hydra" / "config.yaml")
    torch.save({"accuracy": 0.8}, base / "metrics.pt")  # modern zip format
    legacy = base / "legacy_metrics.pt"
    torch.save({"accuracy": 0.9}, legacy, _use_new_zipfile_serialization=False)
    assert not zipfile.is_zipfile(legacy)  # sanity: it really is non-zip

    out = server._get_metrics(tmp_path / "exp")  # must not raise
    assert out["per_run"][0]["metrics"]["accuracy"] == 0.8  # zip metric loads
    if server._TORCH_WEIGHTS_ONLY_SAFE:
        # torch >= 2.6 reads the legacy file safely via weights_only
        assert out["per_run"][0]["legacy_metrics"]["accuracy"] == 0.9
    else:
        # older torch: data-only loader fails closed — skipped, not header garbage
        assert "legacy_metrics" not in out["per_run"][0]


def test_tool_relative_path_resolved_under_root(tmp_path, monkeypatch):
    """A rooted server accepts client paths relative to root, not its launch dir."""
    from mushin.mcp.server import _get_metrics

    root = tmp_path / "outputs"
    _make_experiment(root / "exp")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    out = _get_metrics("exp", root=root)  # relative to root, despite the CWD
    assert out["num_runs"] == 2


def test_describe_reports_flattened_metric_leaves(tmp_path):
    """metric_keys must be the flattened leaves get_metrics accepts, not stems."""
    from mushin.mcp.server import _describe_experiment

    base = tmp_path / "exp" / "0"
    (base / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"lr": 0.1}), base / ".hydra" / "config.yaml")
    torch.save({"accuracy": [0.8, 0.9], "loss": [0.5, 0.4]}, base / "fit_metrics.pt")

    out = _describe_experiment(tmp_path / "exp")
    assert "fit_metrics.accuracy" in out["metric_keys"]
    assert "fit_metrics.loss" in out["metric_keys"]
    assert "fit_metrics" not in out["metric_keys"]  # bare stem no longer reported


def test_reduce_metrics_summarizes_list_metrics(tmp_path):
    """List-valued (per-epoch) metrics reduce by final value across runs."""
    from mushin.mcp.server import _get_metrics

    base = tmp_path / "exp"
    for i, accs in enumerate([[0.7, 0.8], [0.9, 1.0]]):  # final values 0.8, 1.0
        run = base / str(i)
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create({"lr": 0.1 * (i + 1)}), run / ".hydra" / "config.yaml"
        )
        torch.save({"accuracy": accs}, run / "fit_metrics.pt")

    out = _get_metrics(base, reduce="mean")
    # mean of final-epoch accuracy across runs: mean(0.8, 1.0) == 0.9
    assert out["reduced"]["fit_metrics.accuracy"] == pytest.approx(0.9, abs=1e-6)


def _make_workflow_experiment(base: Path, accs=(0.8, 0.9)) -> Path:
    """Hydra multirun layout whose metrics live in the workflow JSON sidecar
    (mushin_metrics.json) rather than MetricsCallback .pt files."""
    import json

    for i, acc in enumerate(accs):
        run = base / str(i)
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create({"lr": float(i)}), run / ".hydra" / "config.yaml"
        )
        (run / "mushin_metrics.json").write_text(json.dumps({"acc": acc}))
    return base


def test_get_metrics_reads_workflow_json_sidecar(tmp_path):
    # The resilient-sweep path stores metrics in mushin_metrics.json; the MCP
    # server must surface them, not report an empty per_run.
    from mushin.mcp.server import _get_metrics

    base = _make_workflow_experiment(tmp_path / "exp")
    out = _get_metrics(base, reduce="mean")
    assert out["num_runs"] == 2
    assert out["per_run"][0]["mushin_metrics"]["acc"] == pytest.approx(0.8)
    assert out["reduced"]["mushin_metrics.acc"] == pytest.approx(0.85)


def test_one_corrupt_config_does_not_kill_experiment(tmp_path):
    # A single half-written config.yaml (e.g. from a killed job) must not make
    # every query for the whole experiment raise.
    from mushin.mcp.server import _get_metrics

    base = _make_experiment(tmp_path / "exp")
    (base / "1" / ".hydra" / "config.yaml").write_text(":\n- [broken")
    out = _get_metrics(base, reduce=None)
    assert out["num_runs"] == 2
    assert out["per_run"][0]["metrics"]["accuracy"] == pytest.approx(0.8, abs=1e-5)
