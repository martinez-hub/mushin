# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Canary for the ``hydra-core < 1.3.7`` ceiling in pyproject.toml.

hydra-core 1.3.7 refuses to instantiate ``hydra._internal.*`` targets named by
declarative config. Hydra builds its own sweeper through
``Plugins.instantiate_sweeper``, which marks the target trusted; hydra-zen's
``launch`` deliberately does not, so the stock ``hydra/sweeper=basic`` target is
rejected and every mushin multirun raises. Hence the ceiling.

A ceiling that nobody revisits is the failure mode we were trying to avoid when
we chose not to cap hydra-core in #204, and the scheduled latest-deps job cannot
warn us here: it resolves within our declared constraints, so it will now never
see 1.3.7 either. This test is the replacement alarm. It watches the upstream
condition the ceiling is premised on, and fails when that changes so somebody
re-reads the pin instead of carrying it forever.
"""

from __future__ import annotations

import inspect

import hydra_zen
from hydra_zen._launch import launch

# The exact line that the 1.3.7 policy rejects: a bare `instantiate` on the
# sweeper config rather than a trip through Hydra's plugin discovery.
_BYPASS = "instantiate(cfg.hydra.sweeper)"


def test_hydra_zen_still_bypasses_plugin_discovery():
    """Fails when hydra-zen stops instantiating the sweeper directly.

    When that happens, hydra-zen has (probably) adopted
    ``Plugins.instantiate_sweeper`` and the ceiling can likely be lifted: drop
    ``< 1.3.7`` from pyproject.toml, run the suite against hydra-core 1.3.7+,
    and delete this file if it passes.
    """
    source = inspect.getsource(launch)
    assert _BYPASS in source, (
        f"hydra-zen {hydra_zen.__version__} no longer contains "
        f"{_BYPASS!r} in `launch`. The `hydra-core < 1.3.7` ceiling in "
        "pyproject.toml was added because that call is rejected by hydra-core's "
        "target policy. Re-check whether the ceiling can be lifted."
    )


def test_sweeper_target_is_still_the_private_hydra_path():
    """The other half of the premise: the config names a rejected path.

    If Hydra ever gives the basic sweeper a public target, the ceiling is moot
    regardless of what hydra-zen does.
    """
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeperConf

    assert BasicSweeperConf._target_.startswith("hydra._internal."), (
        f"BasicSweeperConf now targets {BasicSweeperConf._target_!r}, which is "
        "no longer a private `hydra._internal` path. The `hydra-core < 1.3.7` "
        "ceiling in pyproject.toml may no longer be needed."
    )
