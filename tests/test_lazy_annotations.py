# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A task whose annotations name a ``TYPE_CHECKING``-only import runs (#330).

On Python 3.14+ a definition may name what is not defined yet (PEP 649), and
evaluating its annotations raises ``NameError``. mushin reads the signatures
it needs (and hands hydra-zen's ``zen``) with those annotations kept as text.
"""

import inspect
import pickle
import sys

import pytest
from hydra_zen import make_config, zen
from hydra_zen.errors import HydraZenValidationError

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow, _prepare_task

lazy = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="PEP 649: annotations evaluated lazily"
)


def _plain(a: int, h: str = "x") -> dict:
    return {"v": float(a)}


def test_a_task_whose_annotations_evaluate_is_not_wrapped():
    from mushin.workflows import _signed

    prepared, wants = _prepare_task(_plain)
    assert prepared is _plain and wants is False
    assert _signed(_plain) is _plain
    assert _signed(None) is None


def _resumable_plain(a: int, mushin_resume) -> dict:
    return {"v": float(a)}


class _ResumableWF(MultiRunMetricsWorkflow):
    task = staticmethod(_resumable_plain)


class _NoSignature:
    __signature__ = 5  # inspect.signature raises TypeError

    def __call__(self, a):
        return {"v": float(a)}


class _NoSignatureWF(MultiRunMetricsWorkflow):
    task = staticmethod(_NoSignature())


def test_validate_leaves_a_task_without_a_signature_to_zen():
    """zen says what is wrong with a task it cannot inspect, as before."""
    wf = _NoSignatureWF(eval_task_cfg=make_config(a=1))
    with pytest.raises(HydraZenValidationError, match="inspectable signatures"):
        wf.validate(include_pre_task=False)


def test_validate_hides_mushin_resume_as_run_does(tmp_path):
    """`mushin_resume` is injected at run time, never a config field: validate
    reports only the swept `a` missing, as run() never asks for it."""
    wf = _ResumableWF()
    wf.run(a=multirun([1]), working_dir=str(tmp_path / "s"))
    with pytest.raises(HydraZenValidationError) as err:
        wf.validate(include_pre_task=False)
    assert "mushin_resume" not in str(err.value)


@lazy
def test_a_decorated_task_naming_a_type_checking_import_runs(tmp_path):
    import _lazy_annotation_tasks as tasks

    ds = mushin.sweep(tasks.task).run(
        a=multirun([1, 2]), working_dir=str(tmp_path / "s")
    )
    assert ds["v"].values.tolist() == [1.0, 2.0]


@lazy
def test_a_workflow_whose_task_and_pre_task_name_one_runs(tmp_path):
    import _lazy_annotation_tasks as tasks

    wf = tasks.LazyWF()
    wf.run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    assert wf.to_xarray()["v"].values.tolist() == [1.0, 2.0]
    # zen reads both signatures with the annotations as text: it reports the
    # swept `a` missing from the base config, not a NameError.
    with pytest.raises(
        HydraZenValidationError, match="missing the following fields: a"
    ):
        wf.validate()
    with pytest.raises(
        HydraZenValidationError, match="missing the following fields: a"
    ):
        wf.validate(include_pre_task=False)


@lazy
def test_a_resumable_task_naming_one_runs(tmp_path):
    import _lazy_annotation_tasks as tasks

    ds = mushin.sweep(tasks.resumable).run(
        a=multirun([1, 2]), working_dir=str(tmp_path / "s")
    )
    assert ds["v"].values.tolist() == [1.0, 2.0]
    assert ds["has_resume"].values.tolist() == [1.0, 1.0]


@lazy
def test_a_parallel_run_of_a_task_naming_one_runs(tmp_path):
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    import _lazy_annotation_tasks as tasks

    wf = tasks.LazyWF()
    wf.run(
        a=multirun([1, 2, 3]),
        working_dir=str(tmp_path / "s"),
        launcher="joblib",
        launcher_config={"n_jobs": 2},
    )
    assert wf.to_xarray()["v"].values.tolist() == [1.0, 2.0, 3.0]


@lazy
def test_the_signed_task_reads_annotations_as_text_and_pickles():
    import _lazy_annotation_tasks as tasks
    from mushin.workflows import _SignedTask

    with pytest.raises(NameError):
        inspect.signature(tasks.task)
    prepared, wants = _prepare_task(tasks.task)
    assert isinstance(prepared, _SignedTask) and wants is False
    assert inspect.signature(prepared).parameters["h"].annotation == "Decimal"
    # Text, so zen's parameters pickle without the module's globals.
    pickle.loads(pickle.dumps(prepared))
    pickle.loads(pickle.dumps(zen(prepared)))
    assert b"TYPE_CHECKING" not in pickle.dumps(zen(prepared))


@lazy
def test_a_type_checking_submodule_in_an_annotation_runs(tmp_path):
    import _lazy_annotation_tasks as tasks

    with pytest.raises(AttributeError):
        inspect.signature(tasks.dotted)
    ds = mushin.sweep(tasks.dotted).run(
        a=multirun([1, 2]), working_dir=str(tmp_path / "s")
    )
    assert ds["v"].values.tolist() == [1.0, 2.0]


@lazy
def test_a_string_stand_in_in_an_annotation_runs(tmp_path):
    import _lazy_annotation_tasks as tasks

    with pytest.raises(TypeError):
        inspect.signature(tasks.standin)
    ds = mushin.sweep(tasks.standin).run(
        a=multirun([1, 2]), working_dir=str(tmp_path / "s")
    )
    assert ds["v"].values.tolist() == [1.0, 2.0]


@lazy
def test_the_pre_launch_guards_still_read_a_lazy_task(tmp_path):
    """The typo guard and the required-parameter check read the signature as
    text: neither gives up on a task whose annotations do not evaluate."""
    import _lazy_annotation_tasks as tasks

    handle = mushin.sweep(tasks.task)
    with pytest.raises(ValueError, match="did you mean 'h'"):
        handle.run(a=multirun([1]), hh=multirun([1]), working_dir=str(tmp_path / "t"))
    with pytest.raises(ValueError, match="requires parameter"):
        handle.run(working_dir=str(tmp_path / "r"))


@lazy
def test_a_parallel_run_of_a_decorated_lazy_task_runs(tmp_path):
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    import _lazy_annotation_tasks as tasks

    ds = tasks.swept.run(
        a=multirun([1, 2, 3]),
        working_dir=str(tmp_path / "s"),
        launcher="joblib",
        launcher_config={"n_jobs": 2},
    )
    assert ds["v"].values.tolist() == [1.0, 2.0, 3.0]


@lazy
def test_the_signed_task_carries_the_tasks_names():
    import _lazy_annotation_tasks as tasks

    prepared = _prepare_task(tasks.task)[0]
    assert prepared.__name__ == "task"
    assert prepared.__qualname__ == tasks.task.__qualname__
    assert prepared.__module__ == tasks.task.__module__
    assert not hasattr(prepared, "__wrapped__")
    assert repr(prepared) == repr(tasks.task)  # zen's messages name the task


@lazy
def test_a_signed_partial_does_not_take_partials_names():
    import functools

    import _lazy_annotation_tasks as tasks

    task = functools.partial(tasks.task, h=None)
    prepared = _prepare_task(task)[0]
    assert prepared.__doc__ != functools.partial.__doc__
    assert prepared.__module__ != "functools"
    assert repr(prepared) == repr(task)


@lazy
def test_a_task_parameter_named_self_reaches_the_task():
    import _lazy_annotation_tasks as tasks

    prepared = _prepare_task(tasks.with_self)[0]
    assert prepared(self=1, a=2) == {"v": 3.0}


@lazy
def test_a_metric_naming_a_type_checking_import_takes_the_seed():
    import _lazy_annotation_tasks as tasks
    from mushin.llm._compare import _accepts_seed

    assert _accepts_seed(tasks.metric) is True


@lazy
def test_an_annotation_the_text_read_cannot_read_raises_its_own_error():
    from mushin.workflows import _task_signature

    def odd(a, h=None):
        return a

    def annotate(format):
        if format == 1:  # VALUE
            return {"h": undefined_name}  # noqa: F821
        raise NotImplementedError

    odd.__annotate__ = annotate
    with pytest.raises(NameError, match="undefined_name"):
        _task_signature(odd)


_SCRIPT = """\
import sys
from typing import TYPE_CHECKING

import mushin
from mushin import multirun

if TYPE_CHECKING:
    from decimal import Decimal


@mushin.sweep
def task(a: int, h: Decimal = None) -> dict:
    return {"v": float(a)}


if __name__ == "__main__":
    try:
        task.run(
            a=multirun([1, 2]),
            working_dir=sys.argv[1],
            launcher="joblib",
            launcher_config={"n_jobs": 2},
        )
    except ValueError as error:
        print("VALUE ERROR:", error)
        raise SystemExit(3) from None
"""


@lazy
def test_a_script_task_a_parallel_launcher_cannot_send_stops_before_launch(tmp_path):
    """A parallel launcher sends a script's task by value, which reads its
    annotations: mushin says so before anything launches, with the fixes."""
    import subprocess

    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    script = tmp_path / "parallel_script.py"
    script.write_text(_SCRIPT)
    wd = tmp_path / "sweep"
    done = subprocess.run(
        [sys.executable, str(script), str(wd)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 3, done.stdout + done.stderr
    assert "from __future__ import annotations" in done.stdout
    assert "task" in done.stdout
    assert not wd.exists()
