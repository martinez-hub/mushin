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


@pytest.mark.parametrize(
    "launcher",
    [
        "submitit_slurm",
        "submitit_local",
        # mushin's own, which run() routes those to (#345)
        "mushin_submitit_slurm",
        "mushin_submitit_local",
    ],
)
def test_missing_submitit_launcher_names_the_slurm_extra(launcher):
    """The first error every cluster user hits: the launcher plugin isn't
    installed. The message must say what to pip install, the `slurm` extra
    (which holds the plugin's floor), not just Hydra's raw 'Could not find'
    listing. The dev group installs the plugin (#344), so this hands the hint
    Hydra's error directly; the rq test below takes the same hint through
    `run()`."""
    from hydra.errors import MissingConfigException

    from mushin.workflows import _with_launcher_hint

    error = MissingConfigException(
        f"Could not find 'hydra/launcher/{launcher}'", f"hydra/launcher/{launcher}"
    )
    hinted = _with_launcher_hint(error, launcher)
    assert isinstance(hinted, MissingConfigException)
    assert 'pip install "mushin-py[slurm]"' in str(hinted)


@pytest.mark.parametrize("values", [[1], ["true", "y"]])
def test_a_missing_launcher_is_named_wherever_hydra_meets_it(tmp_path, values):
    """A swept string Hydra reads back differently has the sweep's config
    composed before launch (#319); a missing launcher fails there, with the
    same install hint as in the launch."""
    from hydra.errors import MissingConfigException

    try:
        import hydra_plugins.hydra_rq_launcher  # noqa: F401

        pytest.skip("hydra-rq-launcher installed; missing-plugin path unreachable")
    except ImportError:
        pass

    with pytest.raises(MissingConfigException, match="pip install hydra-rq-launcher"):
        _W().run(a=multirun(values), launcher="rq", working_dir=str(tmp_path / "s"))


def test_launcher_config_dict_value_emits_hydra_dict_grammar():
    """Dict-valued launcher fields (the documented
    additional_parameters={"requeue": True}) must serialize to Hydra's dict
    grammar and parse back as a dict — not a quoted Python repr string that
    fails the structured launcher config's validation."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    from mushin.workflows import _to_override_element

    s = _to_override_element({"requeue": True, "count": 2, "qos": "low"})
    parsed = OverridesParser.create().parse_override(f"++x={s}").value()
    assert parsed == {"requeue": True, "count": 2, "qos": "low"}


def test_launcher_config_none_value_emits_null():
    """None must serialize to Hydra null, not the string 'None' (which would
    submit e.g. partition="None" to SLURM)."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    from mushin.workflows import _to_override_element

    assert _to_override_element(None) == "null"
    parsed = OverridesParser.create().parse_override("++x=null").value()
    assert parsed is None


def test_robustness_curve_forwards_launcher_config(tmp_path):
    """RobustnessCurve.run must forward launcher_config like the other
    workflows — a bogus field must be rejected by the launcher config, not
    silently dropped (which would submit with plugin-default resources)."""
    from mushin.workflows import RobustnessCurve

    class RC(RobustnessCurve):
        @staticmethod
        def task(epsilon):
            return dict(result=float(epsilon))

    with pytest.raises(Exception, match="bogus_field"):
        RC().run(
            epsilon=[0.0, 1.0],
            launcher="joblib",
            launcher_config={"bogus_field": 3},
            working_dir=str(tmp_path / "s"),
        )
