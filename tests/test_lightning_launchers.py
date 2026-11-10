# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Unit tests for HydraDDP launcher helpers that don't need GPUs."""

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

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

    # With nothing mushin-set, PL_GLOBAL_SEED is still cleared...
    monkeypatch.setenv("PL_GLOBAL_SEED", "123")
    _teardown()
    assert "PL_GLOBAL_SEED" not in os.environ  # PL's, safe to reset between jobs
    # ...and a call with nothing set at all must not raise.
    _teardown()


def test_setup_environment_noop_when_distributed_uninitialized():
    _setup_environment()  # process group not initialized -> no-op, must not raise


@pytest.mark.parametrize(
    ("external", "launched", "num_nodes", "num_processes", "raises"),
    [
        # the launcher started 4 (e.g. ntasks_per_node=2 over 2 nodes), but the
        # Trainer expects num_nodes=2 x devices=4 = 8
        pytest.param(True, 4, 2, 4, True, id="mismatch-raises"),
        pytest.param(True, 8, 2, 4, False, id="match-ok"),  # 2 x 4 == 8
        # the single-node subprocess path: mismatched, but must be ignored
        pytest.param(False, 999, 1, 2, False, id="not-external-skips"),
    ],
)
def test_validate_external_world_size(
    external, launched, num_nodes, num_processes, raises
):
    from mushin.lightning.launchers import _validate_external_world_size

    class _Env:
        creates_processes_externally = external

        def world_size(self):
            return launched

    check = pytest.raises(RuntimeError, match="world size") if raises else nullcontext()
    with check:
        _validate_external_world_size(
            num_nodes=num_nodes, num_processes=num_processes, cluster_environment=_Env()
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

    def fake_subprocess_call(
        local_rank, global_rank, testing, predicting, validating=False
    ):
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
    """A re-launched rank's run dir is written with forward slashes, valid on
    every platform, and quoted as a swept string is (#321), so Hydra reads it
    back as written, '=' and all."""
    from pathlib import PurePosixPath, PureWindowsPath

    from hydra.core.override_parser.overrides_parser import OverridesParser

    from mushin.lightning.launchers import _hydra_run_dir_override

    def read_back(override):
        line = f"hydra.run.dir={override}"
        return OverridesParser.create().parse_overrides([line])[0].value()

    windows = _hydra_run_dir_override(PureWindowsPath(r"C:\Users\me\out"))
    assert read_back(windows) == "C:/Users/me/out"
    # '=' in the dir name still relies on the quoting
    assert read_back(_hydra_run_dir_override(PurePosixPath("/tmp/lr=0.1"))) == (
        "/tmp/lr=0.1"
    )


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


def test_launch_dispatches_validate_as_validating_not_fit(monkeypatch):
    """trainer.validate() dispatches as `_validate_impl`; the child ranks must
    be told to VALIDATE, not fall through to fit (which would train while rank 0
    validates -> collective mismatch/hang)."""
    from lightning_fabric.plugins.environments import LightningEnvironment

    import mushin.lightning.launchers as L

    # LightningEnvironment.creates_processes_externally is True when LOCAL_RANK
    # is set; ensure a clean env so the child-spawn branch actually runs (other
    # launcher tests leak these vars into os.environ).
    for _v in ("LOCAL_RANK", "NODE_RANK", "WORLD_SIZE", "RANK", "MASTER_PORT"):
        monkeypatch.delenv(_v, raising=False)

    captured = []

    def fake_subprocess_call(local_rank, global_rank, testing, predicting, validating):
        captured.append(
            dict(testing=testing, predicting=predicting, validating=validating)
        )
        return object()

    monkeypatch.setattr(L, "_subprocess_call", fake_subprocess_call)
    monkeypatch.setattr(L, "sleep", lambda *_: None)
    monkeypatch.setattr(L, "_launch_process_observer", lambda procs: None)
    monkeypatch.setattr(L, "_set_num_threads_if_needed", lambda num_processes: None)

    launcher = L._HydraReattachLauncher(
        cluster_environment=LightningEnvironment(), num_processes=2, num_nodes=1
    )

    def _validate_impl():
        return "validated"

    assert launcher.launch(_validate_impl, trainer=None) == "validated"
    assert captured and all(
        c == dict(testing=False, predicting=False, validating=True) for c in captured
    )


def test_pl_main_pl_validating_calls_trainer_validate():
    from unittest.mock import MagicMock

    from mushin.lightning._pl_main import task

    trainer = MagicMock()
    module = object()
    task(trainer, module, datamodule=None, pl_validating=True)
    trainer.validate.assert_called_once()
    trainer.fit.assert_not_called()
    trainer.test.assert_not_called()


def _run_subprocess_call_in_hydra_job(monkeypatch, tmp_path, overrides, **entry):
    """hydra-zen-launch a job and invoke the REAL _subprocess_call inside it
    (local rank 1, global rank 5), with Popen captured. Returns the captured
    dict: ``command``, ``cwd`` (Popen's), ``task_cwd`` and ``output_dir``.
    ``entry`` sets the testing/predicting/validating flags (default: fit)."""
    import subprocess

    from hydra_zen import launch, make_config

    captured = {}

    class _FakeProc:
        pass

    def fake_popen(command, env=None, cwd=None):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        return _FakeProc()

    def task(cfg):
        from hydra.core.hydra_config import HydraConfig

        captured["task_cwd"] = os.getcwd()
        captured["output_dir"] = HydraConfig.get().runtime.output_dir
        flags = {"testing": False, "predicting": False, **entry}
        launchers._subprocess_call(local_rank=1, global_rank=5, **flags)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # config.yaml saved by Hydra must contain `trainer` and `module` keys
    # (the launcher validates them before spawning child ranks).
    Config = make_config(trainer=1, module=2)
    jr = launch(Config, task, overrides=overrides, version_base="1.3")
    assert jr.return_value is None  # surfaces any exception raised in the task
    return captured


def test_subprocess_call_locates_job_dir_without_chdir(monkeypatch, tmp_path):
    # Under version_base >= 1.2 semantics Hydra does NOT chdir into the job
    # dir. The rank re-launcher must locate the job's saved config.yaml and
    # spawn children via HydraConfig's runtime output dir, not os.getcwd().
    captured = _run_subprocess_call_in_hydra_job(monkeypatch, tmp_path, [])

    job_dir = Path(captured["output_dir"]).resolve()
    assert Path(captured["task_cwd"]).resolve() != job_dir  # no chdir happened
    assert Path(captured["cwd"]).resolve() == job_dir  # child spawns in job dir
    # the child is pointed at the job's saved config, which really exists
    cp = captured["command"][captured["command"].index("-cp") + 1]
    assert Path(cp).resolve() == (job_dir / ".hydra").resolve()
    assert (Path(cp) / "config.yaml").exists()
    # and its own hydra.run.dir is the job dir, not the launch dir
    run_dir = next(
        o for o in captured["command"] if o.startswith("hydra.run.dir=")
    ).split("=", 1)[1]
    assert Path(run_dir.strip("'\"")).resolve() == job_dir


@pytest.mark.parametrize("entry", [None, "testing", "predicting", "validating"])
def test_subprocess_call_argv(monkeypatch, tmp_path, entry):
    """The child rank's argv: the entry module, the job's saved config, one
    flag per Trainer entry point (only ``entry`` true; none for fit), the
    local rank, then the job's Hydra settings."""
    captured = _run_subprocess_call_in_hydra_job(
        monkeypatch, tmp_path, [], **({entry: True} if entry else {})
    )
    cmd = captured["command"]
    assert cmd[:4] == [sys.executable, "-m", "mushin.lightning._pl_main", "-cp"]
    assert cmd[5:11] == [
        "-cn",
        "config.yaml",
        f"++pl_testing={'true' if entry == 'testing' else 'false'}",
        f"++pl_predicting={'true' if entry == 'predicting' else 'false'}",
        f"++pl_validating={'true' if entry == 'validating' else 'false'}",
        "++pl_local_rank=1",
    ]
    assert cmd[11].startswith("hydra.run.dir=")
    assert cmd[12] == "hydra.output_subdir=.pl_hydra_rank_5"
    assert cmd[13].startswith("hydra.job.name=")
    assert len(cmd) == 14


def test_subprocess_call_unchanged_when_chdir_is_on(monkeypatch, tmp_path):
    # With hydra.job.chdir=True (what mushin's run() always sets) the job dir
    # and the CWD coincide — the launcher must resolve to the same place.
    captured = _run_subprocess_call_in_hydra_job(
        monkeypatch, tmp_path, ["hydra.job.chdir=True"]
    )

    job_dir = Path(captured["output_dir"]).resolve()
    assert Path(captured["task_cwd"]).resolve() == job_dir  # chdir happened
    assert Path(captured["cwd"]).resolve() == job_dir
