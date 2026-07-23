"""Defensive-handling / correctness cleanup fixes from the bug hunt.

Each test pins a small robustness fix: corrupt-but-valid-JSON sidecars must
degrade rather than crash resume; public load paths must raise real errors (not
-O-stripped asserts); the resume contextvar must not leak on a failed status
write; a plot exception must not leak a figure; a malformed tuning pin must
reach its friendly guard; lazy submodule access must work; a validation message
must be well-formed; and `correction=` must be validated before any paid work.
"""

import json

import pytest


# --- resume / sweep-IO durability -------------------------------------------
def test_build_resume_context_degrades_on_bad_attempt(tmp_path):
    from mushin._resume import STATUS_FILE, build_resume_context

    combo = {"a": 1}
    for bad in ("null", '"3"'):
        (tmp_path / STATUS_FILE).write_text(
            f'{{"status": "completed", "combo": {{"a": 1}}, "attempt": {bad}}}'
        )
        rc = build_resume_context(tmp_path, combo)  # must not raise
        assert isinstance(rc.attempt, int) and rc.attempt >= 1


def test_from_cell_status_skips_wrong_shape_sidecars(tmp_path):
    from mushin._resume import STATUS_FILE
    from mushin._sweep_io import Manifest

    good = tmp_path / "0"
    good.mkdir()
    (good / STATUS_FILE).write_text(
        json.dumps({"status": "completed", "combo": {"a": 1}})
    )
    for i, bad in enumerate(["[1, 2, 3]", '"just a string"', "42"], start=1):
        d = tmp_path / str(i)
        d.mkdir()
        (d / STATUS_FILE).write_text(bad)
    m = Manifest.from_cell_status(tmp_path, ["a"])  # must not raise
    assert len(m.cells) == 1  # only the well-formed cell


def test_read_cell_status_returns_none_for_non_dict(tmp_path):
    from mushin._resume import STATUS_FILE, read_cell_status

    (tmp_path / STATUS_FILE).write_text("[1, 2, 3]")  # valid JSON, wrong type
    assert read_cell_status(tmp_path) is None


def test_load_from_dir_missing_config_raises_filenotfound(tmp_path):
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2]), working_dir=str(wd))
    # delete one cell's config to simulate an incomplete/cleaned sweep dir
    next(wd.glob("*/.hydra/config.yaml")).unlink()
    with pytest.raises(FileNotFoundError):
        MultiRunMetricsWorkflow().load_from_dir(str(wd), "mushin_metrics.json")


def test_resume_contextvar_not_leaked_when_status_write_fails(tmp_path, monkeypatch):
    import mushin._resume as R
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, mushin_resume=None):
            return dict(v=float(a))

    def boom(cell_dir, *, status, **kw):
        if status == "running":
            raise RuntimeError("disk full")
        return None

    monkeypatch.setattr(R, "write_cell_status", boom)
    with pytest.raises(Exception):
        W().run(a=multirun([1]), working_dir=str(tmp_path / "s"))
    assert R.current_resume() is None  # contextvar restored, no leak


# --- resource / robustness --------------------------------------------------
def test_plot_does_not_leak_figure_on_exception():
    import matplotlib.pyplot as plt

    from mushin.workflows import RobustnessCurve

    class RC(RobustnessCurve):
        @staticmethod
        def task(epsilon):
            return dict(result=100 - epsilon**2)

    wf = RC()
    wf.run(epsilon=[0, 1, 2])
    before = len(plt.get_fignums())
    for _ in range(3):
        with pytest.raises(Exception):
            wf.plot("does_not_exist")  # KeyError inside plotting
    assert len(plt.get_fignums()) == before  # no accumulated figures


def test_malformed_pin_reaches_friendly_guard(tmp_path):
    from mushin._tuning import _read_pin

    (tmp_path / "pin.yaml").write_text("- 1\n- 2\n")  # a YAML list, not a mapping
    pin = _read_pin(tmp_path / "pin.yaml")  # must not raise an opaque error
    assert not isinstance(pin, dict)  # callers' isinstance(pin, dict) guard fires


def test_lazy_submodule_access(monkeypatch):
    import subprocess
    import sys

    code = (
        "import mushin;"
        "print(mushin.lightning.__name__, mushin.benchmark.__name__, "
        "mushin.testing.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "mushin.lightning mushin.benchmark mushin.testing"


def test_value_check_optional_message_is_well_formed():
    from mushin._validate import value_check

    with pytest.raises(TypeError, match="None or of type"):
        value_check("x", 1, type_=str, optional=True)


# --- validate correction before any paid work -------------------------------
def test_compare_rejects_unknown_correction_before_evaluating():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import compare

    called = {"n": 0}

    class _M(torch.nn.Module):
        def forward(self, x):
            called["n"] += 1
            return torch.zeros(len(x), 3)

    data = DataLoader(
        TensorDataset(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
    )
    with pytest.raises(ValueError, match="correction"):
        compare({"m": [_M()]}, data, task="classification", correction="nope")
    assert called["n"] == 0  # no model was evaluated


# --- MCP tolerant of wrong-type sidecar JSON --------------------------------
def test_mcp_get_failures_tolerates_wrong_type_manifest(tmp_path):
    from mushin.mcp.server import _get_failures

    (tmp_path / "mushin_sweep_manifest.json").write_text("[1, 2, 3]")
    out = _get_failures(tmp_path)  # must not raise
    assert out["count"] == 0


def test_mcp_get_provenance_tolerates_wrong_type_record(tmp_path):
    from mushin.mcp.server import _get_provenance

    d = tmp_path / "0"
    d.mkdir()
    (d / "mushin_provenance.json").write_text('"just a string"')
    out = _get_provenance(tmp_path, include_config=True)  # must not raise
    assert out["num_runs"] == 0


def test_plot_with_ax_saves_that_axes_figure(tmp_path, monkeypatch):
    """plot(ax=..., save_filename=...) must save the figure `ax` belongs to,
    not whatever figure happens to be matplotlib's current one."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from mushin.workflows import RobustnessCurve

    class RC(RobustnessCurve):
        @staticmethod
        def task(epsilon):
            return dict(result=100 - epsilon**2)

    wf = RC()
    wf.run(epsilon=[0, 1, 2], working_dir=str(tmp_path / "s"))

    fig1, ax = plt.subplots()
    plt.figure()  # a different, now-current figure

    saved = []
    orig = Figure.savefig

    def spy(self, *args, **kwargs):
        saved.append(self)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    try:
        wf.plot("result", ax=ax, save_filename=str(tmp_path / "p.png"))
    finally:
        plt.close("all")
    assert saved and saved[0] is fig1
