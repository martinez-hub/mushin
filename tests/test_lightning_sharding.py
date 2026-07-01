# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest


def test_fsdp_strategy_builds_via_hydra_zen():
    # FSDP needs torch>=1.12 with the FSDP API; skip cleanly if unavailable.
    FSDPStrategy = pytest.importorskip("pytorch_lightning.strategies").FSDPStrategy
    from hydra_zen import builds, instantiate

    # The exact shape documented in the sharding guide. Construction must not
    # initialize distributed (CPU-only), so this is hermetic. Note: no
    # populate_full_signature — it would pull FSDPStrategy's ``timeout: timedelta``
    # default, which the lowest-supported hydra-zen (0.10) cannot serialize
    # (HydraZenUnsupportedPrimitiveError). Passing only the args we document keeps
    # the guard valid across the dependency floor.
    cfg = builds(FSDPStrategy, sharding_strategy="FULL_SHARD")
    strategy = instantiate(cfg)
    assert isinstance(strategy, FSDPStrategy)
