# Multi-node DDP Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `mushin` correct and ergonomic for multi-node DDP delegated to `hydra-submitit-launcher` + Lightning's `SLURMEnvironment`/`TorchElasticEnvironment`, by fixing the rank-safety bugs that fire under multi-node, adding a fail-fast tasks↔ranks validation, and shipping config/seed helpers + docs.

**Architecture:** Under an external launcher (`creates_processes_externally=True`), `HydraDDP` already steps aside and Lightning uses the externally-launched ranks — so the work is NOT process spawning. It is: save metrics only on global rank 0, clear only mushin-set env vars on teardown, validate `world_size == num_nodes × devices_per_node` (fail fast), and provide a submitit-config helper (derives `tasks_per_node` from `gpus_per_node`) and a per-rank seeding helper. All unit-testable in CI; an end-to-end multi-node run is the (deferred) merge gate.

**Tech Stack:** Python 3.9+, PyTorch Lightning (≥2.4), hydra-zen, Hydra, `hydra-submitit-launcher` (optional, user-installed), pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-06-27-multinode-ddp-design.md`

**MERGE GATE:** Do NOT merge this branch until the cluster smoke test (Task 6) is run on a real multi-node cluster and passes. Everything below is unit-testable without a cluster.

---

## File Structure

- `src/mushin/lightning/callbacks.py` — `MetricsCallback`: guard `torch.save` on `trainer.is_global_zero`.
- `src/mushin/lightning/launchers.py` — scoped `_teardown` + `_set_env` tracking; `_validate_external_world_size` (called from `HydraDDP.setup_environment`); `_global_rank` helper + global-rank output subdir in `_subprocess_call`/`_call_children_scripts`.
- `src/mushin/lightning/_cluster.py` (new) — `submitit_slurm_config`, `seed_everything_per_rank`.
- `src/mushin/lightning/__init__.py`, `src/mushin/__init__.py` — export the two new helpers.
- `tests/test_lightning_callbacks.py`, `tests/test_lightning_launchers.py`, `tests/test_cluster.py` (new) — unit tests; gated cluster smoke test.
- `pyproject.toml` — register a `cluster` pytest marker, deselected by default.
- `docs/guides/multinode.md` (new) + `mkdocs.yml` nav + `changes/+multinode-ddp.added.md`.

The dead PL `<1.6` branch in `launchers.py` (mushin requires PL ≥2.4) is left untouched; only the `PL_VERSION >= Version(1, 6, 0)` branch matters.

---

### Task 1: `MetricsCallback` saves only on global rank 0

**Files:**
- Modify: `src/mushin/lightning/callbacks.py` (`on_validation_end`, `on_test_end`)
- Test: `tests/test_lightning_callbacks.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lightning_callbacks.py` (this file already has `_make_fake_trainer`/`_make_fake_module` MagicMock helpers from the audit work — reuse them; they let you set `trainer.is_global_zero`, `trainer.sanity_checking`, `trainer.callback_metrics`, and `module.current_epoch`):

```python
def test_metrics_callback_saves_only_on_global_zero(tmp_path):
    import torch

    from mushin.lightning.callbacks import MetricsCallback

    cb = MetricsCallback(save_dir=tmp_path)
    trainer = _make_fake_trainer(
        callback_metrics={"val_acc": torch.tensor(0.5)}, is_global_zero=False
    )
    module = _make_fake_module(current_epoch=0)

    cb.on_validation_end(trainer, module)
    # non-zero rank records in memory but writes NO file (avoids N ranks clobbering)
    assert not (tmp_path / "fit_metrics.pt").exists()
    assert cb.val_metrics["val_acc"] == [0.5]

    trainer0 = _make_fake_trainer(
        callback_metrics={"val_acc": torch.tensor(0.6)}, is_global_zero=True
    )
    cb.on_validation_end(trainer0, _make_fake_module(current_epoch=1))
    assert (tmp_path / "fit_metrics.pt").exists()  # rank 0 writes


def test_metrics_callback_test_end_saves_only_on_global_zero(tmp_path):
    import torch

    from mushin.lightning.callbacks import MetricsCallback

    cb = MetricsCallback(save_dir=tmp_path)
    cb.on_test_end(
        _make_fake_trainer(callback_metrics={"acc": torch.tensor(1.0)}, is_global_zero=False),
        _make_fake_module(current_epoch=0),
    )
    assert not (tmp_path / "test_metrics.pt").exists()
```

If `_make_fake_trainer` does not yet accept an `is_global_zero` kwarg, extend it to set that attribute on the mock (default `True` so existing tests are unaffected). Read the existing helper first and adapt minimally.

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_lightning_callbacks.py -k global_zero -v`
Expected: FAIL — the callback currently saves unconditionally, so the `not exists()` assertion fails.

- [ ] **Step 3: Implement the rank-0 guard**

In `src/mushin/lightning/callbacks.py`, change the two save sites. `on_validation_end`:

```python
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        # Make sure PL is not doing its sanity check run
        if trainer.sanity_checking:
            return self.val_metrics
        self._record(
            self.val_metrics, trainer.callback_metrics, pl_module.current_epoch
        )
        # Under (multi-node) DDP every rank fires this callback; only rank 0 writes
        # so N ranks don't clobber the same file on a shared filesystem.
        if trainer.is_global_zero:
            torch.save(self.val_metrics, self._get_filename("fit"))
        return self.val_metrics

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        self._record(self.test_metrics, trainer.callback_metrics)
        if trainer.is_global_zero:
            torch.save(self.test_metrics, self._get_filename("test"))
        return self.test_metrics
```

- [ ] **Step 4: Run, expect PASS** — `uv run pytest tests/test_lightning_callbacks.py -v` (new + existing all pass; existing tests use `is_global_zero=True` default).

- [ ] **Step 5: Style + commit**

```bash
uv run ruff check src/mushin/lightning/callbacks.py tests/test_lightning_callbacks.py
uv run ruff format src/mushin/lightning/callbacks.py tests/test_lightning_callbacks.py
git add src/mushin/lightning/callbacks.py tests/test_lightning_callbacks.py
git commit -m "fix(lightning): MetricsCallback saves only on global rank 0"
```

---

### Task 2: Scoped `_teardown` (clear only mushin-set env vars)

**Files:**
- Modify: `src/mushin/lightning/launchers.py` (`_teardown`, add `_set_env` + `_MUSHIN_SET_ENV`, use it in `_call_children_scripts`)
- Test: `tests/test_lightning_launchers.py`

Context: `_call_children_scripts` (the single-node subprocess launcher) sets `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, `LOCAL_RANK`, `WORLD_SIZE`. Under an external launcher (SLURM/torchrun) those vars are scheduler-owned and `_call_children_scripts` never runs, so `_teardown` must NOT pop them. `PL_GLOBAL_SEED` is set by Lightning (not mushin) but is safe to reset between jobs.

- [ ] **Step 1: Rewrite the teardown tests**

Replace the body of `tests/test_lightning_launchers.py` (it currently asserts the OLD unconditional behavior). New content:

```python
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
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_lightning_launchers.py -v` (FAIL: `_set_env` / `_MUSHIN_SET_ENV` don't exist yet).

- [ ] **Step 3: Implement scoped teardown**

In `src/mushin/lightning/launchers.py`, replace the `_teardown` function and add the tracking helper near the top (after the imports, before `_setup_environment`):

```python
# Env vars mushin itself set (single-node subprocess launcher). Under an external
# launcher (SLURM/torchrun) these are scheduler-owned and mushin sets none, so
# teardown leaves them alone.
_MUSHIN_SET_ENV: set[str] = set()


def _set_env(name: str, value: str) -> None:
    os.environ[name] = value
    _MUSHIN_SET_ENV.add(name)
```

Replace `_teardown`:

```python
def _teardown() -> None:
    # Remove only the env vars mushin set itself, so consecutive multirun jobs
    # start fresh without stomping scheduler-owned vars under SLURM/torchrun.
    for name in list(_MUSHIN_SET_ENV):
        os.environ.pop(name, None)
    _MUSHIN_SET_ENV.clear()
    # PL_GLOBAL_SEED is Lightning's, not scheduler-owned; safe to reset each job.
    os.environ.pop("PL_GLOBAL_SEED", None)
```

In `_HydraDDPLauncher._call_children_scripts` (the `PL_VERSION >= Version(1, 6, 0)` branch), replace the five `os.environ[...] = ...` assignments with `_set_env(...)`:

```python
            # DDP Environment variables (tracked so _teardown clears only these)
            _set_env("MASTER_ADDR", self.cluster_environment.main_address)
            _set_env("MASTER_PORT", str(self.cluster_environment.main_port))
            _set_env("NODE_RANK", str(self.cluster_environment.node_rank()))
            _set_env("LOCAL_RANK", str(self.cluster_environment.local_rank()))
            _set_env("WORLD_SIZE", f"{self.num_processes * self.num_nodes}")
```

- [ ] **Step 4: Run, expect PASS** — `uv run pytest tests/test_lightning_launchers.py -v`.

- [ ] **Step 5: Style + commit**

```bash
uv run ruff check src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
uv run ruff format src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
git add src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
git commit -m "fix(lightning): scoped _teardown clears only mushin-set env vars"
```

---

### Task 3: Fail-fast tasks↔ranks validation

**Files:**
- Modify: `src/mushin/lightning/launchers.py` (add `_validate_external_world_size`; call it from `HydraDDP.setup_environment`)
- Test: `tests/test_lightning_launchers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lightning_launchers.py`:

```python
def test_validate_external_world_size_mismatch_raises():
    import pytest

    from mushin.lightning.launchers import _validate_external_world_size

    class _Env:
        creates_processes_externally = True

        def world_size(self):
            return 4  # launcher started 4 (e.g. ntasks_per_node=2 over 2 nodes)

    # Trainer expects num_nodes=2 x devices=4 = 8
    with pytest.raises(RuntimeError, match="world size"):
        _validate_external_world_size(num_nodes=2, num_processes=4, cluster_environment=_Env())


def test_validate_external_world_size_match_ok():
    from mushin.lightning.launchers import _validate_external_world_size

    class _Env:
        creates_processes_externally = True

        def world_size(self):
            return 8

    # 2 nodes x 4 devices == 8 -> no error
    _validate_external_world_size(num_nodes=2, num_processes=4, cluster_environment=_Env())


def test_validate_skips_when_not_external():
    from mushin.lightning.launchers import _validate_external_world_size

    class _Env:
        creates_processes_externally = False  # single-node subprocess path

        def world_size(self):
            return 999  # mismatched, but must be ignored

    _validate_external_world_size(num_nodes=1, num_processes=2, cluster_environment=_Env())
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_lightning_launchers.py -k validate -v` (no such function).

- [ ] **Step 3: Implement the validator + wire it into `setup_environment`**

Add the free function to `src/mushin/lightning/launchers.py` (top level, near `_setup_environment`):

```python
def _validate_external_world_size(num_nodes, num_processes, cluster_environment) -> None:
    """Under an external launcher (SLURM/torchrun), fail fast if the number of
    launched processes doesn't match num_nodes x devices-per-node — the #1
    multi-node footgun (a mismatch otherwise hangs at rendezvous, OOMs, or
    silently runs single-GPU). No-op for the single-node subprocess path."""
    if cluster_environment is None or not cluster_environment.creates_processes_externally:
        return
    expected = int(num_nodes) * int(num_processes)
    actual = int(cluster_environment.world_size())
    if actual != expected:
        raise RuntimeError(
            f"DDP world-size mismatch: the launcher started {actual} process(es), "
            f"but the Trainer expects num_nodes={num_nodes} x devices={num_processes} "
            f"= {expected}. For DDP, set the launcher's tasks-per-node equal to "
            f"GPUs-per-node (== Trainer `devices`). See the multi-node guide."
        )
```

In `HydraDDP.setup_environment` (the `PL_VERSION >= Version(1, 6, 0)` branch), call it after `super().setup_environment()`:

```python
        def setup_environment(self) -> None:
            _setup_environment()
            super().setup_environment()
            _validate_external_world_size(
                self.num_nodes, self.num_processes, self.cluster_environment
            )
```

- [ ] **Step 4: Run, expect PASS** — `uv run pytest tests/test_lightning_launchers.py -k validate -v`.

- [ ] **Step 5: Style + commit**

```bash
uv run ruff check src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
uv run ruff format src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
git add src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
git commit -m "feat(lightning): fail-fast validation of SLURM tasks vs DDP ranks"
```

---

### Task 4: Global-rank output subdir (defensive)

**Files:**
- Modify: `src/mushin/lightning/launchers.py` (`_global_rank` helper; `_subprocess_call` signature; `_call_children_scripts` call site)
- Test: `tests/test_lightning_launchers.py`

Context: `_subprocess_call` writes `hydra.output_subdir=.pl_hydra_rank_{local_rank}`. This path runs only in the single-node subprocess launcher, but for the rare multi-node-without-external-launcher case two nodes' local-rank-1 collide on a shared FS. Key the subdir on global rank.

- [ ] **Step 1: Write the failing test**

```python
def test_global_rank_computation():
    from mushin.lightning.launchers import _global_rank

    # node 0: local 0,1 -> global 0,1 ; node 1 with 2 GPUs/node: local 0,1 -> global 2,3
    assert _global_rank(node_rank=0, num_processes=2, local_rank=1) == 1
    assert _global_rank(node_rank=1, num_processes=2, local_rank=0) == 2
    assert _global_rank(node_rank=1, num_processes=2, local_rank=1) == 3
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_lightning_launchers.py -k global_rank_computation -v`.

- [ ] **Step 3: Implement**

Add to `src/mushin/lightning/launchers.py`:

```python
def _global_rank(node_rank: int, num_processes: int, local_rank: int) -> int:
    """Global rank = node_rank * (GPUs per node) + local_rank."""
    return int(node_rank) * int(num_processes) + int(local_rank)
```

Change `_subprocess_call` to take a `global_rank` and use it for the output subdir. Update its signature `def _subprocess_call(local_rank, global_rank, testing, predicting):` and the override line:

```python
        f"hydra.output_subdir=.pl_hydra_rank_{global_rank}",
```

(Keep `local_rank` for the `LOCAL_RANK` env and the `++pl_local_rank` flag.) Update the call in `_call_children_scripts` to compute and pass it:

```python
            node_rank = self.cluster_environment.node_rank()
            for local_rank in range(1, self.num_processes):
                _subprocess_call(
                    local_rank,
                    _global_rank(node_rank, self.num_processes, local_rank),
                    testing,
                    predicting,
                )
                delay = np.random.uniform(1, 5, 1)[0]
                sleep(delay)
```

- [ ] **Step 4: Run, expect PASS** — `uv run pytest tests/test_lightning_launchers.py -k global_rank -v` (the computation test; the subprocess path itself is GPU/cluster-gated).

- [ ] **Step 5: Style + commit**

```bash
uv run ruff check src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
uv run ruff format src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
git add src/mushin/lightning/launchers.py tests/test_lightning_launchers.py
git commit -m "fix(lightning): key per-rank Hydra output subdir on global rank"
```

---

### Task 5: `submitit_slurm_config` + `seed_everything_per_rank` helpers

**Files:**
- Create: `src/mushin/lightning/_cluster.py`
- Modify: `src/mushin/lightning/__init__.py`, `src/mushin/__init__.py`
- Test: `tests/test_cluster.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cluster.py`:

```python
import pytest


def test_submitit_slurm_config_derives_tasks_per_node():
    from mushin.lightning import submitit_slurm_config

    cfg = submitit_slurm_config(nodes=2, gpus_per_node=4, partition="gpu", cpus_per_task=8)
    # DDP contract: one SLURM task per GPU
    assert cfg["tasks_per_node"] == 4
    assert cfg["gpus_per_node"] == 4
    assert cfg["nodes"] == 2
    assert cfg["cpus_per_task"] == 8
    assert cfg["partition"] == "gpu"


def test_submitit_slurm_config_passthrough_and_optional():
    from mushin.lightning import submitit_slurm_config

    cfg = submitit_slurm_config(nodes=1, gpus_per_node=2, mem_gb=64, account="proj")
    assert cfg["mem_gb"] == 64
    assert cfg["account"] == "proj"  # extra kwargs pass through
    assert "partition" not in cfg  # omitted when not given


def test_submitit_slurm_config_rejects_bad_inputs():
    from mushin.lightning import submitit_slurm_config

    with pytest.raises(ValueError):
        submitit_slurm_config(nodes=0, gpus_per_node=4)
    with pytest.raises(ValueError):
        submitit_slurm_config(nodes=1, gpus_per_node=0)


def test_seed_everything_per_rank_offsets_by_global_rank(monkeypatch):
    from mushin.lightning import seed_everything_per_rank

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert seed_everything_per_rank(1000) == 1003  # base + global rank

    monkeypatch.setenv("RANK", "5")  # RANK takes precedence
    assert seed_everything_per_rank(1000) == 1005


def test_seed_everything_per_rank_defaults_rank_zero(monkeypatch):
    from mushin.lightning import seed_everything_per_rank

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    assert seed_everything_per_rank(42) == 42
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_cluster.py -v` (module/functions don't exist).

- [ ] **Step 3: Implement `_cluster.py`**

Create `src/mushin/lightning/_cluster.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Helpers for multi-node training on SLURM/Elastic clusters."""

from __future__ import annotations

import os
from typing import Any

from pytorch_lightning import seed_everything


def submitit_slurm_config(
    *,
    nodes: int,
    gpus_per_node: int,
    cpus_per_task: int = 1,
    partition: str | None = None,
    timeout_min: int = 60,
    mem_gb: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a ``hydra-submitit-launcher`` SLURM config for multi-node DDP.

    ``tasks_per_node`` is derived as ``gpus_per_node`` so the two can never desync
    (DDP needs one SLURM task per GPU). Returns a plain dict you wire into your
    Hydra ``hydra/launcher`` config (see the multi-node guide); it submits nothing.
    Extra keyword args (e.g. ``account``, ``qos``) pass through verbatim.
    """
    if int(nodes) < 1 or int(gpus_per_node) < 1:
        raise ValueError(
            f"`nodes` and `gpus_per_node` must be >= 1; got nodes={nodes}, "
            f"gpus_per_node={gpus_per_node}"
        )
    cfg: dict[str, Any] = {
        "nodes": int(nodes),
        "gpus_per_node": int(gpus_per_node),
        "tasks_per_node": int(gpus_per_node),  # one DDP rank per GPU
        "cpus_per_task": int(cpus_per_task),
        "timeout_min": int(timeout_min),
    }
    if partition is not None:
        cfg["partition"] = partition
    if mem_gb is not None:
        cfg["mem_gb"] = int(mem_gb)
    cfg.update(extra)
    return cfg


def seed_everything_per_rank(base: int, workers: bool = True) -> int:
    """Seed each process with ``base + global_rank`` so a multi-GPU/-node run is as
    reproducible as a single-GPU run (each rank gets a distinct but deterministic
    seed). Reads the global rank from ``RANK`` (preferred) or ``SLURM_PROCID``,
    defaulting to 0. Returns the seed used."""
    rank_str = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"
    seed = int(base) + int(rank_str)
    seed_everything(seed, workers=workers)
    return seed
```

- [ ] **Step 4: Export the helpers**

In `src/mushin/lightning/__init__.py`:

```python
from ._cluster import seed_everything_per_rank, submitit_slurm_config
from .callbacks import MetricsCallback
from .launchers import HydraDDP

__all__ = [
    "MetricsCallback",
    "HydraDDP",
    "submitit_slurm_config",
    "seed_everything_per_rank",
]
```

In `src/mushin/__init__.py`, add to the lightning import and `__all__`:

```python
from .lightning import (
    HydraDDP,
    MetricsCallback,
    seed_everything_per_rank,
    submitit_slurm_config,
)
```
and add `"submitit_slurm_config"` and `"seed_everything_per_rank"` to the `__all__` list.

- [ ] **Step 5: Run, expect PASS** — `uv run pytest tests/test_cluster.py -v`, then `uv run python -c "import mushin; print(mushin.submitit_slurm_config, mushin.seed_everything_per_rank)"`.

- [ ] **Step 6: Style + commit**

```bash
uv run ruff check src/mushin/lightning tests/test_cluster.py
uv run ruff format src/mushin/lightning tests/test_cluster.py
git add src/mushin/lightning/_cluster.py src/mushin/lightning/__init__.py src/mushin/__init__.py tests/test_cluster.py
git commit -m "feat(lightning): submitit_slurm_config + seed_everything_per_rank helpers"
```

---

### Task 6: Gated cluster smoke test (the MERGE GATE)

**Files:**
- Modify: `pyproject.toml` (register the `cluster` marker, deselect by default)
- Test: `tests/test_cluster.py`

- [ ] **Step 1: Register the marker**

In `pyproject.toml` under `[tool.pytest.ini_options]`, extend the existing `addopts` and `markers` (the `real_data` marker is already there from the detection work). The `addopts` currently reads `-m 'not real_data'`; change it to deselect `cluster` too:

```toml
addopts = "-m 'not real_data and not cluster'"
markers = [
    "real_data: end-to-end checks that download a real dataset/model (deselected by default; run with -m real_data)",
    "cluster: end-to-end multi-node DDP checks that need a real SLURM/Elastic cluster (deselected by default; run with -m cluster)",
]
```

(If `addopts`/`markers` already exist, edit them in place — do not add duplicate keys.)

- [ ] **Step 2: Write the gated smoke test**

Append to `tests/test_cluster.py` (it is collected only under `-m cluster`, so it never runs in normal CI):

```python
@pytest.mark.cluster
def test_multinode_ddp_end_to_end(tmp_path):
    """MERGE GATE: run on a real multi-node SLURM allocation. Launches a tiny DDP
    job via submitit + SLURMEnvironment, then asserts it completed, metrics were
    written exactly once (rank 0), and results load back. Run with:
        pytest -m cluster tests/test_cluster.py
    on a node with `hydra-submitit-launcher` installed and a SLURM allocation.
    See docs/guides/multinode.md for the runbook and required env."""
    pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    # The concrete launch is documented in the runbook; this test is a placeholder
    # the cluster operator fleshes out against their partition/account. It exists
    # to (a) reserve the marker and (b) be the named gate. Keep it skipping cleanly
    # until wired to a real allocation:
    pytest.skip(
        "Provide a SLURM allocation + partition/account, then implement the launch "
        "per docs/guides/multinode.md (this is the merge gate)."
    )
```

> **Implementer note:** This task intentionally ships the *gate* (marker + deselection + a clearly-skipping placeholder + the runbook in Task 7), not a fabricated cluster run. The real launch can only be authored/validated against an actual cluster; do NOT fake it. The branch's merge condition is the human running this on their cluster.

- [ ] **Step 3: Verify deselection** — `uv run pytest tests/test_cluster.py -q` (the unit tests run; `test_multinode_ddp_end_to_end` is deselected). `uv run pytest tests/test_cluster.py -m cluster --collect-only -q` lists it.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/test_cluster.py
git commit -m "test(cluster): gated multi-node merge-gate marker + smoke placeholder"
```

---

### Task 7: Docs — multi-node guide + runbook + changelog

**Files:**
- Create: `docs/guides/multinode.md`
- Modify: `mkdocs.yml` (nav)
- Create: `changes/+multinode-ddp.added.md`

- [ ] **Step 1: Write the guide**

Create `docs/guides/multinode.md`:

````markdown
# Multi-node training (SLURM / Elastic)

`HydraDDP` runs single-node multi-GPU out of the box. For **multi-node**, delegate
process launching to a cluster scheduler — Hydra's
[`hydra-submitit-launcher`](https://hydra.cc/docs/plugins/submitit_launcher/) +
Lightning's auto-detected `SLURMEnvironment`. Under an external launcher,
`HydraDDP` steps aside and Lightning uses the scheduler-launched ranks directly.

## The one contract: one SLURM task per GPU

For DDP there must be exactly one process (rank) per GPU:

```
tasks_per_node == gpus_per_node == Trainer `devices` per node
world_size     == nodes * gpus_per_node
```

Get this wrong and you hang at rendezvous (too few tasks), OOM (too many), or
silently run single-GPU. `mushin.submitit_slurm_config` derives `tasks_per_node`
from `gpus_per_node` so they cannot desync, and `HydraDDP` **fails fast** with a
clear error if the launched world size doesn't match `num_nodes x devices`.

## Building the launcher config

```python
from mushin import submitit_slurm_config

slurm = submitit_slurm_config(
    nodes=2,
    gpus_per_node=4,        # -> tasks_per_node=4, one rank per GPU
    cpus_per_task=8,
    partition="gpu",
    timeout_min=120,
    mem_gb=64,
)
```

Wire these values into your Hydra submitit launcher (install the plugin:
`pip install hydra-submitit-launcher`). Set `hydra/launcher=submitit_slurm` and
override its fields from the dict, e.g. as launch overrides:
`hydra.launcher.nodes=2 hydra.launcher.tasks_per_node=4 hydra.launcher.gpus_per_node=4 ...`.
The Trainer must use a matching `devices`/`num_nodes`:

```python
import pytorch_lightning as pl
from hydra_zen import builds
from mushin import HydraDDP

TrainerConfig = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=4,        # == gpus_per_node
    num_nodes=2,      # == nodes
    strategy=builds(HydraDDP),
    populate_full_signature=True,
)
```

## Reproducibility

Seed each rank deterministically so a 64-GPU run reproduces a 1-GPU run:

```python
from mushin import seed_everything_per_rank

seed_everything_per_rank(1234)   # each rank: 1234 + global_rank
```

## Metrics

`MetricsCallback` writes `metrics.pt` only on global rank 0, so the N ranks don't
clobber the file on a shared filesystem; `load_experiment` reads it back as usual.

## Runbook (the merge gate)

To validate a real multi-node run on your cluster:

1. `pip install hydra-submitit-launcher` on the cluster.
2. Build the launcher config with `submitit_slurm_config(nodes=2, gpus_per_node=<G>, partition=<P>, account=<A>)`.
3. Configure the Trainer with `devices=<G>`, `num_nodes=2`, `strategy=builds(HydraDDP)`.
4. Launch your hydra-zen workflow with `hydra/launcher=submitit_slurm` and `--multirun`.
5. Confirm: the job completes, `metrics.pt` exists exactly once per job dir,
   `load_experiment` aggregates the results, and a mismatched `tasks_per_node`
   raises the fail-fast world-size error rather than hanging.

This is the condition for merging the multi-node branch.
````

- [ ] **Step 2: Add to nav** — in `mkdocs.yml`, add `- Multi-node training: guides/multinode.md` under the guides nav section (match the existing indentation/format).

- [ ] **Step 3: Changelog fragment** — create `changes/+multinode-ddp.added.md`:

```markdown
Multi-node DDP support: `submitit_slurm_config` (derives `tasks_per_node` from
`gpus_per_node`) and `seed_everything_per_rank` helpers, a fail-fast check that the
launched world size matches `num_nodes x devices`, `MetricsCallback` now writes only
on global rank 0, and `_teardown` clears only mushin-set env vars (leaving
scheduler-owned vars alone under SLURM/torchrun). See the new multi-node guide.
```

- [ ] **Step 4: Build docs strict** — `uv run --group docs mkdocs build --strict` (no warnings; the trailing mkdocs-material banner is not an error). Then `rm -rf site/`.

- [ ] **Step 5: Commit**

```bash
git add docs/guides/multinode.md mkdocs.yml changes/+multinode-ddp.added.md
git commit -m "docs(lightning): multi-node training guide + runbook + changelog"
```

---

### Task 8: Full verification sweep

- [ ] **Step 1: Lint / format / spell**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run codespell src/mushin/lightning tests/test_cluster.py docs/guides/multinode.md changes/+multinode-ddp.added.md`
Expected: all clean. (Fix and re-run if not.)

- [ ] **Step 2: Full suite** — `uv run pytest -q` — all pass; the `cluster` and `real_data` tests are deselected.

- [ ] **Step 3: Final fixup commit (if any)**

```bash
git add -A && git commit -m "chore(lightning): lint/format fixups for multi-node support"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** rank-0 saves (Task 1), scoped teardown (Task 2), tasks↔ranks validation (Task 3), global-rank subdir (Task 4), submitit helper + per-rank seed (Task 5), cluster merge-gate (Task 6), docs/runbook (Task 7). Every spec component maps to a task. The deferred rendezvous rewrite is explicitly out of scope.
- **Placeholder scan:** no TBD/TODO; every code step has complete code. Task 6's placeholder skip is intentional and explained (the real cluster launch can't be authored without a cluster — it's the human-run merge gate, not deferred work).
- **Type consistency:** `_set_env`/`_MUSHIN_SET_ENV`, `_validate_external_world_size(num_nodes, num_processes, cluster_environment)`, `_global_rank(node_rank, num_processes, local_rank)`, `submitit_slurm_config(*, nodes, gpus_per_node, ...)`, `seed_everything_per_rank(base, workers=True)` are used identically wherever referenced.
- **Merge gate** is called out in the header and Task 6.
