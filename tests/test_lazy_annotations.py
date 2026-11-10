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
from hydra_zen import zen

import mushin
from mushin import multirun
from mushin.workflows import _prepare_task

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
    wf.validate()  # zen reads both signatures with the annotations as text


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
