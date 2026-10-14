# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Canary for sweeper construction breaking under the installed Hydra.

Every ``mushin.sweep().run()`` is a Hydra multirun, and a multirun only happens
if hydra-zen can build a sweeper from the installed hydra-core. That pairing has
broken before: hydra-core 1.3.6 began refusing to instantiate the
``hydra._internal`` sweeper target that hydra-zen named directly, which took out
every sweep on a fresh install. hydra-zen 0.17.0 builds the sweeper through
Hydra's plugin registry instead, and ``pyproject.toml`` floors hydra-zen there.

This file keeps one end-to-end sweep, so the next break of that kind fails a
test named for the cause instead of scattering failures across the suite. The
``latest-deps`` CI job, which resolves new releases of both packages, is where
it is expected to fire first.
"""

from __future__ import annotations

import mushin


def test_a_real_sweep_runs_under_the_installed_hydra(tmp_path):
    """End to end: the thing users actually do, on whatever hydra is installed."""

    @mushin.sweep
    def task(x: int) -> dict:
        return {"y": x * 2}

    ds = task.run(x=mushin.multirun([1, 2]), working_dir=str(tmp_path))
    assert dict(ds.sizes) == {"x": 2}
    assert [float(v) for v in ds["y"].values] == [2.0, 4.0]
