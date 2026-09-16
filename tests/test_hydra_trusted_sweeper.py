# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Alarm for the private-API workaround in ``mushin.workflows``.

hydra-core 1.3.7 refuses to instantiate ``hydra._internal.*`` targets named by
declarative config. Hydra exempts its own sweeper via
``Plugins.instantiate_sweeper``; hydra-zen's ``launch`` does not, so the stock
``hydra/sweeper=basic`` target is rejected and every mushin multirun raises.
``_trusted_sweeper_target`` marks that one target trusted around our launch.

That leans on two things upstream: a PRIVATE context manager
(``hydra._internal.target_policy._trusted_internal_target``, 1.3.7+) and the
exact target string the stock sweeper config uses. If either moves, the wrap
silently stops applying and every sweep breaks again — so these tests assert
the wrap still *works*, not that some source line still looks a certain way.

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


def _hydra_exposes_policy() -> bool:
    """Ask hydra directly, never mushin's captured import.

    Reading mushin's own `_hydra_trusted_internal_target` would make this test
    SKIP when the guard goes stale — exactly the case it exists to catch.
    """
    try:
        from hydra._internal.target_policy import (  # noqa: F401
            _trusted_internal_target,
        )
    except ImportError:
        return False
    return True


_HAS_POLICY = _hydra_exposes_policy()


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
def test_sweeper_is_still_rejected_without_the_wrap():
    """If this stops raising, the wrap is dead weight and should be deleted.

    This is the check the old canary could not make: it observes the actual
    policy decision, so it fires when *hydra* relaxes the rule or hydra-zen
    starts routing through `Plugins` — not only when hydra-zen's source moves.
    """
    from hydra_zen import instantiate

    with pytest.raises(Exception, match="cannot be selected by declarative"):
        instantiate({"_target_": BASIC_SWEEPER_TARGET, "max_batch_size": None})


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
    from hydra._internal.target_policy import _TRUSTED_INTERNAL_TARGET_CONTEXT
    from hydra_zen import instantiate

    @mushin.sweep
    def task(x: int) -> dict:
        return {"y": x}

    task.run(x=mushin.multirun([1]), working_dir=str(tmp_path))

    assert _TRUSTED_INTERNAL_TARGET_CONTEXT.get() is None
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
