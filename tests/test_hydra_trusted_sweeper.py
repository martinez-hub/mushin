# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Alarm for the private-API workaround in ``mushin.workflows``.

hydra-core 1.3.7 refuses to instantiate ``hydra._internal.*`` targets named by
declarative config. Hydra exempts its own sweeper via
``Plugins.instantiate_sweeper``; hydra-zen's ``launch`` does not, so the stock
``hydra/sweeper=basic`` target is rejected and every mushin multirun raises.
``_trusted_sweeper_target`` marks that one target trusted around our launch.

That leans on two things upstream: a PRIVATE context manager
(``_trusted_internal_target`` — in ``hydra._internal.target_policy`` on 1.3.7,
``hydra._internal.execution_policy`` on 1.4) and the exact target string the
stock sweeper config uses. If either moves, the wrap
silently stops applying and every sweep breaks again — so these tests assert
the wrap still *works*, not that some source line still looks a certain way.
And once hydra-zen no longer needs it (hydra-zen#885), the wrap must go:
``test_the_wrap_is_still_needed`` runs a sweep with it disabled and fails the
day that sweep succeeds.

The previous version of this file asserted a substring of hydra-zen's source
and the sweeper conf's ``_target_``. Both were satisfied by the very
configuration that is broken, so the alarm was green on hydra-core 1.3.7 itself
and could not see a fix arriving from hydra's side.
"""

from __future__ import annotations

import hydra
import pytest
from hydra_zen import make_config

import mushin
from mushin.workflows import (
    BASIC_SWEEPER_TARGET,
    _hydra_trusted_internal_target,
    _trusted_sweeper_target,
)


def _hydra_policy_module():
    """Ask hydra directly, never mushin's captured import.

    Reading mushin's own `_hydra_trusted_internal_target` would make these
    tests SKIP when the guard goes stale — exactly the case they exist to
    catch. 1.3.7 keeps the policy in `target_policy`; 1.4 moved it to
    `execution_policy`. Returns the module, or None when hydra has no policy.
    """
    import importlib

    for name in ("hydra._internal.target_policy", "hydra._internal.execution_policy"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "_trusted_internal_target"):
            return mod
    return None


_POLICY = _hydra_policy_module()
_HAS_POLICY = _POLICY is not None


def test_wrap_is_active_exactly_when_hydra_has_the_policy():
    """The guard must track hydra, not a version string we hard-code."""
    import contextlib

    ctx = _trusted_sweeper_target()
    if _HAS_POLICY:
        assert _hydra_trusted_internal_target is not None, (
            f"hydra-core {hydra.__version__} exposes the target policy, but "
            "mushin's guarded import did not pick it up — sweeps will fail."
        )
        assert not isinstance(ctx, contextlib.nullcontext), (
            f"hydra-core {hydra.__version__} exposes the target policy, but the "
            "wrap fell back to a no-op — sweeps will fail."
        )
    else:
        assert isinstance(ctx, contextlib.nullcontext)
    with ctx:  # must be usable either way
        pass


@pytest.mark.skipif(not _HAS_POLICY, reason="hydra-core < 1.3.7 has no target policy")
def test_the_trusted_target_is_the_one_hydra_rejects():
    """The string we trust must be the string the stock sweeper config names.

    If Hydra renames the sweeper or gives it a public target, trusting the old
    path is a silent no-op and every sweep raises again.
    """
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeperConf

    assert BasicSweeperConf._target_ == BASIC_SWEEPER_TARGET, (
        f"stock sweeper target is {BasicSweeperConf._target_!r}, but mushin "
        f"trusts {BASIC_SWEEPER_TARGET!r}. Update BASIC_SWEEPER_TARGET, or drop "
        "the wrap if the target is no longer private."
    )


@pytest.mark.skipif(not _HAS_POLICY, reason="hydra-core < 1.3.7 has no target policy")
def test_hydra_still_rejects_the_sweeper_target_by_config():
    """The rule the wrap exists for. If this stops raising, *hydra* relaxed it.

    This asks hydra directly, so it cannot see hydra-zen change — that is what
    `test_the_wrap_is_still_needed` is for.
    """
    from hydra_zen import instantiate

    with pytest.raises(Exception, match="cannot be selected by declarative"):
        instantiate({"_target_": BASIC_SWEEPER_TARGET, "max_batch_size": None})


@pytest.mark.skipif(not _HAS_POLICY, reason="hydra-core < 1.3.7 has no target policy")
def test_the_wrap_is_still_needed(tmp_path, monkeypatch):
    """Run a real sweep with the wrap disabled. It must FAIL — or the wrap is dead.

    hydra-zen#885 moves `launch` onto `Plugins.instantiate_sweeper`, which
    carries Hydra's own exemption; once that ships, a sweep succeeds without our
    wrap and this test fails on purpose. The failure message says what to do —
    including raising the hydra-zen floor, without which deleting the wrap
    re-breaks every older hydra-zen on hydra-core >= 1.3.7.
    """
    import mushin.workflows as workflows

    monkeypatch.setattr(workflows, "_hydra_trusted_internal_target", None)

    @mushin.sweep
    def task(x: int) -> dict:
        return {"y": x}

    try:
        task.run(x=mushin.multirun([1]), working_dir=str(tmp_path))
    except Exception as exc:  # the expected outcome on today's hydra-zen
        assert "cannot be selected by declarative" in f"{exc}\n{exc.__cause__}", exc
        return
    pytest.fail(
        "A sweep succeeded with the trusted-sweeper wrap disabled: hydra-zen now "
        "constructs the sweeper through Hydra's plugin registry. In ONE change: "
        "raise the hydra-zen floor in pyproject.toml to the release that ships "
        "hydra-zen#885 (older hydra-zen on hydra-core >= 1.3.7 still needs the "
        "wrap, and no CI job resolves that pairing), then delete "
        "`_trusted_sweeper_target` in src/mushin/workflows.py and the "
        "policy-gated tests in this file."
    )


def test_a_real_sweep_runs_under_the_installed_hydra(tmp_path):
    """End to end: the thing users actually do, on whatever hydra is installed."""

    @mushin.sweep
    def task(x: int) -> dict:
        return {"y": x * 2}

    ds = task.run(x=mushin.multirun([1, 2]), working_dir=str(tmp_path))
    assert dict(ds.sizes) == {"x": 2}
    assert [float(v) for v in ds["y"].values] == [2.0, 4.0]


@pytest.mark.skipif(not _HAS_POLICY, reason="hydra-core < 1.3.7 has no target policy")
def test_trust_does_not_leak_past_the_sweep(tmp_path):
    """The widening must be scoped to our launch and no wider."""
    from hydra_zen import instantiate

    @mushin.sweep
    def task(x: int) -> dict:
        return {"y": x}

    task.run(x=mushin.multirun([1]), working_dir=str(tmp_path))

    assert _POLICY._TRUSTED_INTERNAL_TARGET_CONTEXT.get() is None
    # a different hydra-internal target is still refused afterwards
    with pytest.raises(Exception, match="cannot be selected by declarative"):
        instantiate({"_target_": "hydra._internal.utils._locate", "_partial_": True})
    # and so is the sweeper itself, outside the launch window
    with pytest.raises(Exception, match="cannot be selected by declarative"):
        instantiate({"_target_": BASIC_SWEEPER_TARGET, "max_batch_size": None})


def test_config_still_builds(tmp_path):
    """Guards against the wrap swallowing hydra-zen's own config handling."""
    cfg = make_config(x=1)
    assert cfg is not None
