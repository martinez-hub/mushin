# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`run(launcher=..., launcher_config=...)`: launcher fields without hand-rolled
`hydra.launcher.*` override strings, and an actionable error when the launcher
plugin itself is not installed."""

from __future__ import annotations

import pytest

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


class _W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        return dict(m=float(a))


def test_launcher_config_fields_reach_the_launcher(tmp_path):
    # n_jobs is a real joblib-launcher field: the sweep must complete with it.
    wf = _W()
    wf.run(
        a=multirun([1, 2]),
        launcher="joblib",
        launcher_config={"n_jobs": 1},
        working_dir=str(tmp_path / "s"),
    )
    assert wf.is_complete


def test_launcher_config_bogus_field_errors(tmp_path):
    # Proof the fields actually reach the launcher config: an unknown key is
    # rejected by Hydra's structured launcher config, not silently dropped.
    with pytest.raises(Exception, match="bogus_field"):
        _W().run(
            a=multirun([1]),
            launcher="joblib",
            launcher_config={"bogus_field": 3},
            working_dir=str(tmp_path / "s"),
        )


def test_launcher_config_requires_launcher(tmp_path):
    with pytest.raises(ValueError, match="launcher_config.*launcher"):
        _W().run(
            a=multirun([1]),
            launcher_config={"n_jobs": 1},
            working_dir=str(tmp_path / "s"),
        )


def test_missing_launcher_plugin_names_the_pip_package(tmp_path):
    """The first error every cluster user hits: the launcher plugin isn't
    installed. The message must say WHICH package to pip install, not just
    Hydra's raw 'Could not find' listing."""
    from hydra.errors import MissingConfigException

    with pytest.raises(MissingConfigException, match="hydra-submitit-launcher"):
        _W().run(
            a=multirun([1]),
            launcher="submitit_slurm",
            working_dir=str(tmp_path / "s"),
        )
