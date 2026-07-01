# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest


def test_fsdp_strategy_builds_via_hydra_zen():
    # FSDP needs torch>=1.12 with the FSDP API; skip cleanly if unavailable.
    FSDPStrategy = pytest.importorskip("pytorch_lightning.strategies").FSDPStrategy
    from hydra_zen import builds, instantiate

    # The exact shape documented in the sharding guide. Construction must not
    # initialize distributed (CPU-only), so this is hermetic.
    cfg = builds(
        FSDPStrategy,
        sharding_strategy="FULL_SHARD",
        populate_full_signature=True,
    )
    strategy = instantiate(cfg)
    assert isinstance(strategy, FSDPStrategy)
