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
