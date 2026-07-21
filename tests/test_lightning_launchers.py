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


def test_global_rank_computation():
    from mushin.lightning.launchers import _global_rank

    # node 0: local 0,1 -> global 0,1 ; node 1 with 2 GPUs/node: local 0,1 -> global 2,3
    assert _global_rank(node_rank=0, num_processes=2, local_rank=1) == 1
    assert _global_rank(node_rank=1, num_processes=2, local_rank=0) == 2
    assert _global_rank(node_rank=1, num_processes=2, local_rank=1) == 3


def test_call_children_scripts_tracks_procs_and_launch_wires_observer(monkeypatch):
    """The launcher must own its children like PL's base class: track Popen
    handles in self.procs (so kill()/signal forwarding work) and start the
    process observer (so children are reaped if rank 0 dies)."""
    import mushin.lightning.launchers as L

    fake_procs = []

    def fake_subprocess_call(local_rank, global_rank, testing, predicting):
        proc = object()
        fake_procs.append(proc)
        return proc

    observed = {}
    monkeypatch.setattr(L, "_subprocess_call", fake_subprocess_call)
    monkeypatch.setattr(L, "sleep", lambda *_: None)
    monkeypatch.setattr(
        L, "_launch_process_observer", lambda procs: observed.setdefault("obs", procs)
    )
    monkeypatch.setattr(
        L,
        "_set_num_threads_if_needed",
        lambda num_processes: observed.setdefault("threads", num_processes),
    )

    from lightning_fabric.plugins.environments import LightningEnvironment

    launcher = L._HydraReattachLauncher(
        cluster_environment=LightningEnvironment(), num_processes=3, num_nodes=1
    )
    result = launcher.launch(lambda: "ran", trainer=None)

    assert result == "ran"
    assert launcher.procs == fake_procs and len(launcher.procs) == 2
    assert observed["obs"] is launcher.procs
    assert observed["threads"] == 3


def test_hydra_run_dir_override_is_windows_safe():
    """Backslash is an escape character inside Hydra's quoted override
    grammar; a raw Windows cwd would be corrupted. Forward slashes are valid
    on every platform."""
    from pathlib import PurePosixPath, PureWindowsPath

    from mushin.lightning.launchers import _hydra_run_dir_override

    assert (
        _hydra_run_dir_override(PureWindowsPath(r"C:\Users\me\out"))
        == '"C:/Users/me/out"'
    )
    # '=' in the dir name still relies on the quoting
    assert _hydra_run_dir_override(PurePosixPath("/tmp/lr=0.1")) == '"/tmp/lr=0.1"'


def test_interrank_delay_env_override(monkeypatch):
    from mushin.lightning.launchers import _interrank_delay

    monkeypatch.delenv("MUSHIN_DDP_LAUNCH_DELAY", raising=False)
    assert _interrank_delay() == 1.0  # deterministic default
    monkeypatch.setenv("MUSHIN_DDP_LAUNCH_DELAY", "0")
    assert _interrank_delay() == 0.0
    monkeypatch.setenv("MUSHIN_DDP_LAUNCH_DELAY", "2.5")
    assert _interrank_delay() == 2.5
    monkeypatch.setenv("MUSHIN_DDP_LAUNCH_DELAY", "-3")
    assert _interrank_delay() == 0.0  # clamped
    monkeypatch.setenv("MUSHIN_DDP_LAUNCH_DELAY", "garbage")
    assert _interrank_delay() == 1.0  # falls back to default
