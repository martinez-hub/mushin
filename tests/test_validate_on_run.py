# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""run() validates up front that every required task parameter is satisfied by
the base config or an override, so a genuinely missing parameter fails fast with
a clear message instead of an opaque per-job Hydra error mid-sweep."""

from __future__ import annotations

import numpy as np
import pytest
from hydra_zen import builds, make_config

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def test_missing_required_param_raises_before_launch(tmp_path):
    calls = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(epsilon, scale):  # 'scale' provided nowhere
            calls["n"] += 1
            return dict(v=float(epsilon))

    wf = W()
    with pytest.raises(ValueError, match="scale") as exc:
        wf.run(epsilon=multirun([0.0, 1.0]), working_dir=str(tmp_path / "s"))
    assert calls["n"] == 0  # failed before any job launched
    assert "scale" in str(exc.value)


def test_required_param_via_override_is_valid(tmp_path):
    # The common case: the required parameter is supplied by an override, not the
    # base config. Must NOT be a false positive.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(epsilon):
            return dict(v=float(epsilon))

    wf = W()
    wf.run(epsilon=multirun([0.0, 1.0]), working_dir=str(tmp_path / "s"))
    assert wf.is_complete


def test_required_param_via_base_config_is_valid(tmp_path):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(epsilon, scale):
            return dict(v=float(epsilon) * float(scale))

    wf = W(make_config(scale=2.0))  # scale from the base config
    wf.run(epsilon=multirun([0.0, 1.0]), working_dir=str(tmp_path / "s"))
    assert wf.is_complete


def test_param_with_default_is_not_required(tmp_path):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(epsilon, scale=1.0):  # scale has a default
            return dict(v=float(epsilon))

    wf = W()
    wf.run(epsilon=multirun([0.0, 1.0]), working_dir=str(tmp_path / "s"))
    assert wf.is_complete


def test_missing_pre_task_required_param_raises(tmp_path):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def pre_task(seed):  # required, provided nowhere
            np.random.seed(seed)

        @staticmethod
        def task(rand_val):
            return {"rand_val": rand_val}

    wf = W(make_config(rand_val=builds(np.random.rand)))
    with pytest.raises(ValueError, match="seed"):
        wf.run(working_dir=str(tmp_path / "s"))


def test_missing_param_raises_for_decorator_sweep(tmp_path):
    @mushin.sweep
    def exp(epsilon, scale):
        return dict(v=float(epsilon))

    with pytest.raises(ValueError, match="scale"):
        exp.run(epsilon=multirun([0.0, 1.0]), working_dir=str(tmp_path / "s"))
