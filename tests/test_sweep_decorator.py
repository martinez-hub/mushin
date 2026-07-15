# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def test_decorated_sweep_returns_labeled_dataset(tmp_path):
    @mushin.sweep
    def experiment(a, b):
        return dict(v=float(a + b))

    ds = experiment.run(
        a=multirun([1, 2]), b=multirun([0, 1]), working_dir=str(tmp_path / "s")
    )
    assert ds.sizes == {"a": 2, "b": 2}
    assert float(ds["v"].sel(a=2, b=1)) == 3.0


def test_handle_exposes_workflow_and_class(tmp_path):
    @mushin.sweep
    def experiment(seed):
        return dict(v=float(seed))

    assert experiment.workflow is None  # before first run
    experiment.run(seed=multirun([0, 1]), working_dir=str(tmp_path / "s"))
    assert isinstance(experiment.workflow, MultiRunMetricsWorkflow)
    assert experiment.workflow.provenance is not None
    assert issubclass(experiment.workflow_cls, MultiRunMetricsWorkflow)
    experiment.workflow_cls().run(seed=multirun([0]), working_dir=str(tmp_path / "s2"))


def test_wraps_preserves_name_and_doc():
    @mushin.sweep
    def experiment(seed):
        "my docstring"
        return dict(v=float(seed))

    assert experiment.__name__ == "experiment"
    assert experiment.__doc__ == "my docstring"


def test_fresh_instance_per_run_no_state_leak(tmp_path):
    import pytest

    @mushin.sweep
    def experiment(seed):
        if seed == 1:
            raise RuntimeError("boom")
        return dict(v=float(seed))

    with pytest.warns(UserWarning, match="fail"):
        experiment.run(
            seed=multirun([0, 1]), working_dir=str(tmp_path / "a"), on_error="nan"
        )
    assert experiment.workflow.failures  # this run failed
    experiment.run(seed=multirun([0]), working_dir=str(tmp_path / "b"))
    assert experiment.workflow.failures == []


def test_decorated_sweep_resilience_and_resume(tmp_path):
    import numpy as np
    import pytest

    FAIL = {"on": True}

    @mushin.sweep
    def experiment(seed):
        if seed == 1 and FAIL["on"]:
            raise RuntimeError("boom")
        return dict(v=float(seed))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="fail"):
        ds = experiment.run(seed=multirun([0, 1, 2]), working_dir=wd, on_error="nan")
    assert np.isnan(float(ds["v"].sel(seed=1)))
    assert ds.attrs["mushin_failures"]  # carried on the dataset

    FAIL["on"] = False
    ds2 = experiment.run(seed=multirun([0, 1, 2]), working_dir=wd, resume=True)
    assert float(ds2["v"].sel(seed=1)) == 1.0  # filled on resume
    assert not ds2.attrs.get("mushin_failures")


def test_decorated_sweep_receives_mushin_resume(tmp_path):
    import pytest

    seen = {}
    FAIL = {"on": True}

    @mushin.sweep
    def experiment(seed, mushin_resume=None):
        seen[seed] = mushin_resume
        if mushin_resume is not None and mushin_resume.dir is not None:
            (mushin_resume.dir / "last.ckpt").write_text("state")
        if seed == 0 and FAIL["on"]:
            raise RuntimeError("boom")
        return dict(v=float(seed))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="fail"):
        experiment.run(seed=multirun([0, 1]), working_dir=wd, on_error="nan")
    assert seen[0].is_resume is False

    FAIL["on"] = False
    seen.clear()
    experiment.run(seed=multirun([0, 1]), working_dir=wd, resume=True)
    assert 1 not in seen  # seed 1 completed -> short-circuited
    assert seen[0].is_resume is True and seen[0].last_ckpt.name == "last.ckpt"
