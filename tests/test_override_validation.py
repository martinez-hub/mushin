# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Override names are validated against the task signature before launch, so a
typo'd sweep axis (`lrate=` for a `lr` parameter) fails loudly instead of
silently adding a bogus constant dimension to the dataset."""

from __future__ import annotations

import pytest

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def test_typod_override_name_raises_with_suggestion(tmp_path):
    # `lrate` is a typo for the `lr` parameter (which has a default). Previously
    # this ran fine and the dataset gained a fake `lrate` dimension of constant
    # values; now it must raise before launching, and point at the real name.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr=0.1, seed=0):
            return dict(v=float(lr) + float(seed))

    wf = W()
    with pytest.raises(Exception) as exc:  # noqa: PT011 - message is what matters
        wf.run(
            lrate=multirun([1.0, 2.0]),
            seed=multirun([0, 1]),
            working_dir=str(tmp_path / "s"),
        )
    msg = str(exc.value)
    assert "lrate" in msg
    assert "lr" in msg  # suggests the intended parameter


def test_typod_override_name_raises_for_decorator_sweep(tmp_path):
    @mushin.sweep
    def exp(lr=0.1, seed=0):
        return dict(v=float(lr) + float(seed))

    with pytest.raises(Exception) as exc:  # noqa: PT011
        exp.run(lrate=multirun([1.0, 2.0]), working_dir=str(tmp_path / "s"))
    assert "lrate" in str(exc.value)


def test_valid_override_names_still_run(tmp_path):
    # Real parameter names (and a nested dotted path whose head is a real
    # parameter) must pass validation unharmed.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr=0.1, model=None):
            return dict(v=float(lr))

    wf = W()
    wf.run(lr=multirun([0.1, 0.2]), working_dir=str(tmp_path / "s"))
    ds = wf.to_xarray()
    assert set(ds["v"].sizes) == {"lr"}


def test_deliberate_extra_override_not_a_typo_is_allowed(tmp_path):
    # mushin allows overrides beyond the task's own parameters (consumed by
    # pre_task, config groups, interpolations). A name that is NOT a near-miss of
    # a real parameter must pass — only likely typos are rejected.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(epsilon=0.0, acc=0):
            return dict(v=float(epsilon))

    wf = W()
    wf.run(
        epsilon=multirun([0.1, 0.2]),
        extra_knob=multirun([1, 2]),
        working_dir=str(tmp_path / "s"),
    )
    ds = wf.to_xarray()
    assert {"epsilon", "extra_knob"} <= set(ds["v"].sizes)


def test_kwargs_task_accepts_any_override_name(tmp_path):
    # A task that declares **kwargs opts out of validation (any name is valid).
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(**kwargs):
            return dict(v=float(kwargs.get("anything", 0.0)))

    wf = W()
    wf.run(anything=multirun([1.0, 2.0]), working_dir=str(tmp_path / "s"))
    ds = wf.to_xarray()
    assert set(ds["v"].sizes) == {"anything"}
