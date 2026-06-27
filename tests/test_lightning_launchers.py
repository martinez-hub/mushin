# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Unit tests for HydraDDP launcher helpers that don't need GPUs."""

import os

from mushin.lightning import launchers
from mushin.lightning.launchers import _set_env, _setup_environment, _teardown


def test_teardown_clears_only_mushin_set_vars(monkeypatch):
    # scheduler-owned vars present but NOT set by mushin -> must be preserved
    monkeypatch.setenv("MASTER_ADDR", "scheduler-host")
    monkeypatch.setenv("WORLD_SIZE", "8")
    launchers._MUSHIN_SET_ENV.clear()

    # mushin sets a couple of its own
    _set_env("LOCAL_RANK", "1")
    _set_env("NODE_RANK", "0")

    _teardown()

    assert "LOCAL_RANK" not in os.environ  # mushin-set -> cleared
    assert "NODE_RANK" not in os.environ
    assert os.environ["MASTER_ADDR"] == "scheduler-host"  # scheduler-owned -> kept
    assert os.environ["WORLD_SIZE"] == "8"
    assert launchers._MUSHIN_SET_ENV == set()  # tracking reset


def test_teardown_resets_pl_global_seed(monkeypatch):
    monkeypatch.setenv("PL_GLOBAL_SEED", "123")
    launchers._MUSHIN_SET_ENV.clear()
    _teardown()
    assert "PL_GLOBAL_SEED" not in os.environ  # PL's, safe to reset between jobs


def test_teardown_is_idempotent_when_nothing_set(monkeypatch):
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    launchers._MUSHIN_SET_ENV.clear()
    _teardown()  # must not raise when mushin set nothing


def test_setup_environment_noop_when_distributed_uninitialized():
    _setup_environment()  # process group not initialized -> no-op, must not raise


def test_validate_external_world_size_mismatch_raises():
    import pytest

    from mushin.lightning.launchers import _validate_external_world_size

    class _Env:
        creates_processes_externally = True

        def world_size(self):
            return 4  # launcher started 4 (e.g. ntasks_per_node=2 over 2 nodes)

    # Trainer expects num_nodes=2 x devices=4 = 8
    with pytest.raises(RuntimeError, match="world size"):
        _validate_external_world_size(
            num_nodes=2, num_processes=4, cluster_environment=_Env()
        )


def test_validate_external_world_size_match_ok():
    from mushin.lightning.launchers import _validate_external_world_size

    class _Env:
        creates_processes_externally = True

        def world_size(self):
            return 8

    # 2 nodes x 4 devices == 8 -> no error
    _validate_external_world_size(
        num_nodes=2, num_processes=4, cluster_environment=_Env()
    )


def test_validate_skips_when_not_external():
    from mushin.lightning.launchers import _validate_external_world_size

    class _Env:
        creates_processes_externally = False  # single-node subprocess path

        def world_size(self):
            return 999  # mismatched, but must be ignored

    _validate_external_world_size(
        num_nodes=1, num_processes=2, cluster_environment=_Env()
    )
