# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Unit tests for HydraDDP launcher helpers that don't need GPUs (the full
HydraDDP integration test is GPU-gated and skipped in CI, so these cover the
env setup/teardown logic that keeps consecutive multirun jobs clean)."""

import os

from mushin.lightning.launchers import _setup_environment, _teardown

_PL_DDP_ENV = (
    "LOCAL_RANK",
    "NODE_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "PL_GLOBAL_SEED",
)


def test_teardown_clears_pl_ddp_env_and_preserves_others(monkeypatch):
    for k in _PL_DDP_ENV:
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("MUSHIN_UNRELATED", "keep")

    _teardown()

    for k in _PL_DDP_ENV:
        assert k not in os.environ  # the DDP vars are cleared between jobs
    assert os.environ["MUSHIN_UNRELATED"] == "keep"  # unrelated vars untouched


def test_teardown_is_idempotent_when_env_absent(monkeypatch):
    for k in _PL_DDP_ENV:
        monkeypatch.delenv(k, raising=False)
    _teardown()  # must not raise when the vars are already gone


def test_setup_environment_noop_when_distributed_uninitialized():
    # On a plain CPU test run the process group is not initialized, so this is a
    # no-op — it must run without raising.
    _setup_environment()
