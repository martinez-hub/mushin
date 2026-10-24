# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The Hydra job number and config-group choices `run` records in each cell's
status sidecar, and the reload paths that read them instead of hydra.yaml."""

import json
import os
from pathlib import Path

import pytest

from mushin import multirun
from mushin._hydra_yaml import read_hydra_job
from mushin._resume import STATUS_FILE, read_cell_status, sidecar_job_fields
from mushin.workflows import MultiRunMetricsWorkflow


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        return dict(m=float(a))


def _sweep(root, values=(1, 2, 3), **kwargs):
    wf = W()
    wf.run(a=multirun(list(values)), working_dir=str(root), **kwargs)
    return wf


def _cells(root):
    """The cell directories, in job order (the basic launcher names them so)."""
    return sorted(
        (p for p in Path(root).iterdir() if p.is_dir()), key=lambda p: int(p.name)
    )


def _strip_job_fields(cell):
    """Make the sidecar one from before the fields were recorded."""
    p = cell / STATUS_FILE
    data = json.loads(p.read_text())
    del data["job_num"], data["choices"]
    p.write_text(json.dumps(data))


def _forbid_hydra_yaml(monkeypatch):
    from mushin import _hydra_yaml

    def _read(path):
        raise AssertionError(f"load_from_dir read {path}")

    monkeypatch.setattr(_hydra_yaml, "load_hydra_record", _read)


def _order(root):
    wf = W().load_from_dir(root, metrics_filename=None)
    return [d.name for d in wf.multirun_working_dirs]


def test_run_records_what_hydra_yaml_records(tmp_path):
    """Every cell's sidecar carries the job number and choices of its
    hydra.yaml, so a reload can take either record."""
    _sweep(tmp_path / "s")
    cells = _cells(tmp_path / "s")
    assert len(cells) == 3
    for cell in cells:
        status = read_cell_status(cell)
        num, choices = read_hydra_job(cell / ".hydra" / "hydra.yaml")
        assert status["job_num"] == num == int(cell.name)
        assert status["choices"] == choices
        assert choices  # Hydra records its own groups (hydra/launcher, ...)


def _assert_fields(root, n=4):
    cells = _cells(root)
    assert len(cells) == n
    for cell in cells:
        status = read_cell_status(cell)
        assert status["job_num"] == int(cell.name), (root, status)
        assert isinstance(status["choices"], dict), (root, status)


def test_skipped_cached_and_resumed_cells_record_them_too(tmp_path):
    """The status writes that bypass the task (a cell outside the sample, a
    cache hit, a resumed cell) record the fields as a computed cell does.
    Each is checked before the next sweep rewrites the sidecars."""
    computed = []

    class C(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            computed.append(a)
            return dict(m=float(a))

    def sweep(root, **kwargs):
        C().run(a=multirun([1, 2, 3, 4]), working_dir=str(root), **kwargs)

    cache = str(tmp_path / "cache")
    sweep(tmp_path / "sampled", sample=2, cache_dir=cache)
    assert len(computed) == 2
    _assert_fields(tmp_path / "sampled")
    statuses = [read_cell_status(c)["status"] for c in _cells(tmp_path / "sampled")]
    assert statuses.count("skipped") == 2

    sweep(tmp_path / "cached", cache_dir=cache)
    assert len(computed) == 4  # two cells came from the cache
    _assert_fields(tmp_path / "cached")

    for cell in _cells(tmp_path / "cached"):
        _strip_job_fields(cell)  # the resume must write them back
    sweep(tmp_path / "cached", cache_dir=cache, resume=True)
    assert len(computed) == 4  # every cell was resumed
    _assert_fields(tmp_path / "cached")
    for cell in _cells(tmp_path / "cached"):
        # Not older than the hydra.yaml Hydra rewrote for the resumed job.
        assert sidecar_job_fields(cell, cell / ".hydra" / "hydra.yaml") is not None


def test_a_pre_task_failure_records_them_too(tmp_path, monkeypatch):
    """The setup-phase 'failed' write (pre_task raised, before the task) is
    the reused-directory case the ordering is about: it carries the fields,
    so the cell sorts from its sidecar."""

    class Bad(MultiRunMetricsWorkflow):
        @staticmethod
        def pre_task(a):
            if a == 2:
                raise RuntimeError("setup boom")

        @staticmethod
        def task(a):
            return dict(m=float(a))

    with pytest.warns(UserWarning, match="failed"):
        Bad().run(
            a=multirun([1, 2, 3]), working_dir=str(tmp_path / "s"), on_error="nan"
        )
    _assert_fields(tmp_path / "s", n=3)
    statuses = [read_cell_status(c)["status"] for c in _cells(tmp_path / "s")]
    assert statuses == ["completed", "failed", "completed"]
    for cell in _cells(tmp_path / "s"):
        (cell / ".hydra" / "hydra.yaml").unlink()
    _forbid_hydra_yaml(monkeypatch)
    assert _order(tmp_path / "s") == ["0", "1", "2"]


def test_sample_does_not_depend_on_the_choices(tmp_path, monkeypatch):
    """`sample=` keys on the job number alone. With no choices to record,
    nothing goes into the sidecar, and the sample still applies."""
    from mushin import _hydra_yaml

    monkeypatch.setattr(_hydra_yaml, "hydra_choices", lambda hydra: None)
    with pytest.warns(UserWarning, match="not run"):
        _sweep(tmp_path / "s", values=(1, 2, 3, 4), sample=2)
    statuses = [read_cell_status(c)["status"] for c in _cells(tmp_path / "s")]
    assert statuses.count("skipped") == 2
    assert all("job_num" not in read_cell_status(c) for c in _cells(tmp_path / "s"))


def test_failed_and_budget_skipped_cells_record_them_too(tmp_path, monkeypatch):
    """The fail-soft 'failed' write and the budget 'skipped' write carry the
    fields, so a reload of the sweeps one most often inspects offline (the
    ones that went wrong) skips hydra.yaml like any other."""
    import time

    class Flaky(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            if a == 2:
                raise RuntimeError("boom")
            return dict(m=float(a))

    class Slow(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            time.sleep(0.05)
            return dict(m=float(a))

    with pytest.warns(UserWarning, match="failed"):
        Flaky().run(
            a=multirun([1, 2, 3]), working_dir=str(tmp_path / "f"), on_error="nan"
        )
    with pytest.warns(UserWarning, match="not run"):
        Slow().run(
            a=multirun([1, 2, 3]),
            working_dir=str(tmp_path / "b"),
            max_total_seconds=0.01,
        )
    statuses = {}
    for root in ("f", "b"):
        for cell in _cells(tmp_path / root):
            status = read_cell_status(cell)
            statuses.setdefault(root, []).append(status["status"])
            assert status["job_num"] == int(cell.name), (root, status)
            assert isinstance(status["choices"], dict), (root, status)
            (cell / ".hydra" / "hydra.yaml").unlink()
    assert statuses["f"].count("failed") == 1
    assert "skipped" in statuses["b"]

    _forbid_hydra_yaml(monkeypatch)
    for root in ("f", "b"):
        wf = W().load_from_dir(tmp_path / root, "mushin_metrics.json")
        assert [d.name for d in wf.multirun_working_dirs] == ["0", "1", "2"]


def test_running_is_recorded_before_pre_task_and_provenance(tmp_path):
    """The job's own record, with the fields, is in the directory before the
    hooks that take time, so a job killed in them leaves no prior sweep's
    'completed' status, metrics, job number or choices for a reload to take."""
    seen = []

    class P(MultiRunMetricsWorkflow):
        @staticmethod
        def pre_task(*args, **kwargs):
            seen.append(read_cell_status(Path.cwd()))

        @staticmethod
        def task(a):
            return dict(m=float(a))

    P().run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    assert [s["status"] for s in seen] == ["running", "running"]
    assert sorted(s["job_num"] for s in seen) == [0, 1]
    assert all(isinstance(s["choices"], dict) for s in seen)


def test_a_checkpoint_staged_by_pre_task_is_seen_on_resume(tmp_path):
    """The 'running' write moved ahead of pre_task; checkpoint discovery did
    not: a pre_task that stages a checkpoint into the cell dir (the resilience
    guide's pattern) still hands it to the task as ``last_ckpt``."""
    seen = {}

    class S(MultiRunMetricsWorkflow):
        @staticmethod
        def pre_task(a):
            if read_cell_status(Path.cwd())["attempt"] == 2:
                (Path.cwd() / "last.ckpt").write_text("staged")

        @staticmethod
        def task(a, mushin_resume=None):
            seen[mushin_resume.attempt] = mushin_resume.last_ckpt
            if mushin_resume.attempt == 1:
                raise RuntimeError("first attempt fails")
            return dict(m=float(a))

    with pytest.warns(UserWarning, match="failed"):
        S().run(a=multirun([1]), working_dir=str(tmp_path / "s"), on_error="nan")
    S().run(a=multirun([1]), working_dir=str(tmp_path / "s"), resume=True)
    assert seen[1] is None
    assert seen[2] is not None and seen[2].name == "last.ckpt"


def test_load_from_dir_reads_no_hydra_yaml_for_a_new_sweep(tmp_path, monkeypatch):
    """A sweep run on this version reloads from its sidecars alone: the
    hydra.yaml parse they replace was most of a reload's time."""
    _sweep(tmp_path / "s")
    expected = W().load_from_dir(tmp_path / "s", "mushin_metrics.json").to_xarray()
    for cell in _cells(tmp_path / "s"):
        (cell / ".hydra" / "hydra.yaml").unlink()
    _forbid_hydra_yaml(monkeypatch)
    wf = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    assert [d.name for d in wf.multirun_working_dirs] == ["0", "1", "2"]
    assert wf.to_xarray().identical(expected)


@pytest.mark.config_group("model")
def test_config_group_axis_keys_by_the_sidecar_choices(
    tmp_path, restore_config_group, monkeypatch
):
    """Offline, a config-GROUP axis keys by the option name the sidecar
    records; before, only hydra.yaml held it."""
    from hydra.core.config_store import ConfigStore
    from hydra_zen import make_config

    cs = ConfigStore.instance()
    cs.store(group="model", name="small", node={"width": 4})
    cs.store(group="model", name="big", node={"width": 8})

    class G(MultiRunMetricsWorkflow):
        @staticmethod
        def task(model=None):
            return dict(w=float(model["width"]))

    G(make_config(model=None)).run(
        working_dir=str(tmp_path / "s"), model=multirun(["small", "big"])
    )
    for cell in _cells(tmp_path / "s"):
        assert read_cell_status(cell)["choices"]["model"] in ("small", "big")
        (cell / ".hydra" / "hydra.yaml").unlink()
    _forbid_hydra_yaml(monkeypatch)
    ds = (
        G(make_config(model=None))
        .load_from_dir(tmp_path / "s", "mushin_metrics.json")
        .to_xarray()
    )
    assert float(ds["w"].sel(model="big")) == 8.0
    assert float(ds["w"].sel(model="small")) == 4.0


def test_an_older_sweep_reads_hydra_yaml_as_before(tmp_path):
    """Sidecars from before the fields were recorded send the reload to
    hydra.yaml, so existing sweep directories load as they did (and still
    need their hydra.yaml)."""
    _sweep(tmp_path / "s")
    expected = W().load_from_dir(tmp_path / "s", "mushin_metrics.json").to_xarray()
    for cell in _cells(tmp_path / "s"):
        _strip_job_fields(cell)
    loaded = W().load_from_dir(tmp_path / "s", "mushin_metrics.json").to_xarray()
    assert loaded.identical(expected)
    for cell in _cells(tmp_path / "s"):
        (cell / ".hydra" / "hydra.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        W().load_from_dir(tmp_path / "s", "mushin_metrics.json")


def test_cells_with_and_without_the_fields_sort_together(tmp_path, monkeypatch):
    """A sweep resumed on this version has new sidecars beside old ones: only
    the old cells' hydra.yaml is read, and the order is still by job number."""
    _sweep(tmp_path / "s")
    expected = W().load_from_dir(tmp_path / "s", "mushin_metrics.json").to_xarray()
    _strip_job_fields(_cells(tmp_path / "s")[1])

    from mushin import _hydra_yaml

    read, real = [], _hydra_yaml.load_hydra_record

    def _counting(path):
        read.append(Path(path).parent.parent.name)
        return real(path)

    monkeypatch.setattr(_hydra_yaml, "load_hydra_record", _counting)
    wf = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    assert read == ["1"]
    assert [d.name for d in wf.multirun_working_dirs] == ["0", "1", "2"]
    assert wf.to_xarray().identical(expected)


def test_a_sidecar_older_than_its_hydra_yaml_is_not_used(tmp_path):
    """A job killed before its first status write leaves the previous sweep's
    sidecar beside a new hydra.yaml, and the newer record must win. The two
    cells' sidecars are swapped, so the record read shows in the order."""
    _sweep(tmp_path / "s", values=(1, 2))
    a, b = _cells(tmp_path / "s")
    text_a, text_b = (a / STATUS_FILE).read_text(), (b / STATUS_FILE).read_text()
    (a / STATUS_FILE).write_text(text_b)
    (b / STATUS_FILE).write_text(text_a)
    later = max((c / STATUS_FILE).stat().st_mtime_ns for c in (a, b)) + 10**9

    # The sidecars were just written, after hydra.yaml: they are the record.
    assert _order(tmp_path / "s") == ["1", "0"]
    for c in (a, b):
        os.utime(c / ".hydra" / "hydra.yaml", ns=(later, later))
    assert _order(tmp_path / "s") == ["0", "1"]  # hydra.yaml is newer
    for c in (a, b):
        os.utime(c / STATUS_FILE, ns=(later, later))
    assert _order(tmp_path / "s") == ["1", "0"]  # the same instant: the job's own
