# HydraFSDP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `HydraFSDP` — an `FSDPStrategy` subclass that makes sharded training work under Hydra `--multirun`, by sharing `HydraDDP`'s Hydra-aware reattach launcher via a mixin.

**Architecture:** Extract `HydraDDP`'s three strategy-agnostic overrides (`setup_environment`, `_configure_launcher`, `teardown`) into a `_HydraReattachMixin`; rename `_HydraDDPLauncher` → `_HydraReattachLauncher`; then `HydraDDP(_HydraReattachMixin, DDPStrategy)` and `HydraFSDP(_HydraReattachMixin, FSDPStrategy)`. Built on the `multinode-ddp` branch (reuses its scoped `_teardown`, `_validate_external_world_size`). **Cluster-gated — do not merge until multi-GPU validated.**

**Tech Stack:** Python 3.10+, pytorch_lightning (DDPStrategy, FSDPStrategy, `_SubprocessScriptLauncher`), hydra-zen, pytest, uv.

**Reference (read once before starting):**
- Spec: `docs/superpowers/specs/2026-07-01-hydra-fsdp-design.md`
- The file to refactor: `src/mushin/lightning/launchers.py`. The live definitions live inside `if PL_VERSION >= Version(1, 6, 0):` (line ~37), which imports `DDPStrategy` and `_SubprocessScriptLauncher` and then defines `HydraDDP(DDPStrategy)` (~line 43) and `_HydraDDPLauncher(_SubprocessScriptLauncher)` (~line 239). A dead PL<1.6 branch further down defines `HydraDDP(DDPPlugin)` — **leave it untouched**.
- Module-level helpers already present (from PR #50): `_setup_environment()`, `_validate_external_world_size(num_nodes, num_processes, cluster_environment)`, `_teardown()` (scoped to `_MUSHIN_SET_ENV`), `_set_env`, `_global_rank`, `_subprocess_call(local_rank, global_rank, testing, predicting)`.
- Current `HydraDDP` methods (verbatim, to extract):
  ```python
  def setup_environment(self) -> None:
      _setup_environment()
      _validate_external_world_size(self.num_nodes, self.num_processes, self.cluster_environment)
      super().setup_environment()

  def _configure_launcher(self) -> None:
      if self.cluster_environment is None:  # pragma: no cover
          raise TypeError("HydraDDP.cluster_environment is None")
      if not self.cluster_environment.creates_processes_externally:
          self._launcher = _HydraDDPLauncher(
              self.cluster_environment, self.num_processes, self.num_nodes
          )
          self._rank_0_will_call_children_scripts = True

  def teardown(self) -> None:
      super().teardown()
      _teardown()
  ```
- Exports: `src/mushin/lightning/__init__.py` (`from .launchers import HydraDDP`), `src/mushin/__init__.py`.
- Tests: `tests/test_lightning_hydra_ddp.py`, `tests/test_lightning_launchers.py`. The `cluster` pytest marker is registered on this branch (`pyproject.toml`; `addopts = "-m 'not real_data and not cluster'"`).

**Conventions:**
- Source files keep the MIT/FAR copyright header.
- After edits: `uv run ruff check <paths>` + `uv run ruff format <paths>` (target py39 on this branch — do NOT rely on py310-only lint being on).
- Commit messages imperative; **no Claude attribution / no `Co-Authored-By`**.
- Tests via `uv run pytest`. Do NOT `git add -A` (there's an untracked `.worktrees/`); stage named files.

---

### Task 1: Refactor — rename launcher + extract `_HydraReattachMixin` (HydraDDP regression-safe)

Make `HydraDDP`'s behavior identical while extracting the shared pieces. No new feature yet.

**Files:**
- Modify: `src/mushin/lightning/launchers.py`
- Modify: `tests/test_lightning_hydra_ddp.py`, `tests/test_lightning_launchers.py` (only if they reference `_HydraDDPLauncher` by name)

- [ ] **Step 1: Baseline — existing HydraDDP tests green**

Run: `uv run pytest tests/test_lightning_hydra_ddp.py tests/test_lightning_launchers.py -v`
Expected: PASS. (Records the behavior we must preserve.)

- [ ] **Step 2: Find any references to the launcher name**

Run: `grep -rn "_HydraDDPLauncher" src tests`
Note every hit — they all get renamed to `_HydraReattachLauncher` in this task.

- [ ] **Step 3: Rename the launcher class + its uses**

In `src/mushin/lightning/launchers.py` (inside the `PL_VERSION >= Version(1, 6, 0)` block), rename the class `_HydraDDPLauncher` to `_HydraReattachLauncher` (the `class _HydraDDPLauncher(_SubprocessScriptLauncher):` line). Update every reference found in Step 2 (in `launchers.py` the reference is inside `_configure_launcher`; also update any test references).

- [ ] **Step 4: Extract the mixin and re-parent HydraDDP**

Inside the `PL_VERSION >= Version(1, 6, 0)` block, **before** the `class HydraDDP(...)` definition, add the mixin:

```python
    class _HydraReattachMixin:
        """Hydra-aware launcher behavior shared by ``HydraDDP`` and ``HydraFSDP``:
        reattach each rank via the job's saved ``config.yaml`` (not ``sys.argv``,
        which in a Hydra sweep would re-run the wrong job), fail fast on an external
        world-size mismatch, and scope env-var teardown to what mushin set so
        consecutive multirun jobs start clean without stomping scheduler-owned vars.
        """

        def setup_environment(self) -> None:
            _setup_environment()
            # Validate BEFORE super().setup_environment(): that is where Lightning
            # initializes the process group / rendezvous, which is exactly what
            # hangs when the launcher started the wrong number of ranks.
            _validate_external_world_size(
                self.num_nodes, self.num_processes, self.cluster_environment
            )
            super().setup_environment()

        def _configure_launcher(self) -> None:
            if self.cluster_environment is None:  # pragma: no cover
                raise TypeError("cluster_environment is None")
            if not self.cluster_environment.creates_processes_externally:
                self._launcher = _HydraReattachLauncher(
                    self.cluster_environment, self.num_processes, self.num_nodes
                )
                self._rank_0_will_call_children_scripts = True

        def teardown(self) -> None:
            """Additional teardown so consecutive Hydra multirun jobs start fresh."""
            super().teardown()
            _teardown()
```

Then change the `HydraDDP` class to inherit the mixin and DROP its now-duplicated methods, keeping its docstring:

```python
    class HydraDDP(_HydraReattachMixin, DDPStrategy):  # type: ignore
        """<KEEP THE EXISTING HydraDDP DOCSTRING VERBATIM>"""
```

Delete the three method definitions (`setup_environment`, `_configure_launcher`, `teardown`) from `HydraDDP`'s body — they now come from the mixin. The class body is just the docstring. (Note: `_HydraReattachLauncher` must be defined in the module before the mixin *executes* its `_configure_launcher`, but since that only runs at call time, defining the launcher later in the block is fine — but to be safe and readable, ensure `_HydraReattachLauncher` is still defined in the same block; order of class defs doesn't matter for runtime since the name is resolved when `_configure_launcher` is called.)

- [ ] **Step 5: Run the regression tests**

Run: `uv run pytest tests/test_lightning_hydra_ddp.py tests/test_lightning_launchers.py -v`
Expected: PASS — identical to Step 1. If a test referenced `_HydraDDPLauncher` and now fails on import, update it to `_HydraReattachLauncher` and re-run.

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check src/mushin/lightning/launchers.py tests/test_lightning_hydra_ddp.py tests/test_lightning_launchers.py
uv run ruff format src/mushin/lightning/launchers.py tests/test_lightning_hydra_ddp.py tests/test_lightning_launchers.py
git add src/mushin/lightning/launchers.py tests/test_lightning_hydra_ddp.py tests/test_lightning_launchers.py
git commit -m "refactor: extract _HydraReattachMixin + rename launcher (shared by HydraDDP/HydraFSDP)"
```

---

### Task 2: Add `HydraFSDP` + genericize the validation message

**Files:**
- Modify: `src/mushin/lightning/launchers.py`
- Test: `tests/test_lightning_hydra_fsdp.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lightning_hydra_fsdp.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest


def test_hydra_fsdp_is_fsdp_strategy_with_mixin():
    from pytorch_lightning.strategies import FSDPStrategy

    from mushin.lightning.launchers import HydraFSDP, _HydraReattachMixin

    assert issubclass(HydraFSDP, FSDPStrategy)
    assert issubclass(HydraFSDP, _HydraReattachMixin)


def test_hydra_fsdp_configures_reattach_launcher_single_node():
    from mushin.lightning.launchers import HydraFSDP, _HydraReattachLauncher

    strat = HydraFSDP.__new__(HydraFSDP)  # avoid full __init__/accelerator setup

    class _Env:
        creates_processes_externally = False

    strat.cluster_environment = _Env()
    strat.num_processes = 2
    strat.num_nodes = 1
    strat._configure_launcher()
    assert isinstance(strat._launcher, _HydraReattachLauncher)
    assert strat._rank_0_will_call_children_scripts is True


def test_hydra_fsdp_steps_aside_under_external_launcher():
    from mushin.lightning.launchers import HydraFSDP

    strat = HydraFSDP.__new__(HydraFSDP)

    class _Env:
        creates_processes_externally = True

    strat.cluster_environment = _Env()
    strat.num_processes = 4
    strat.num_nodes = 2
    # external launcher: do not install mushin's launcher
    strat._launcher = None
    strat._configure_launcher()
    assert strat._launcher is None
```

- [ ] **Step 2: Confirm failure**

Run: `uv run pytest tests/test_lightning_hydra_fsdp.py -v`
Expected: FAIL — `cannot import name 'HydraFSDP'`.

- [ ] **Step 3: Add `HydraFSDP` and the FSDP import**

In `src/mushin/lightning/launchers.py`, inside the `PL_VERSION >= Version(1, 6, 0)` block, add the FSDP import next to the DDP import:

```python
    from pytorch_lightning.strategies.fsdp import FSDPStrategy
```

After the `HydraDDP` class (still inside the block), add:

```python
    class HydraFSDP(_HydraReattachMixin, FSDPStrategy):  # type: ignore
        """Fully-Sharded Data Parallel strategy that works under Hydra ``--multirun``.

        Like :class:`HydraDDP`, but for sharded training: it replaces Lightning's
        stock ``FSDPStrategy`` subprocess launcher (which re-execs the script with
        ``sys.argv`` and, in a sweep, spawns the wrong job) with mushin's launcher,
        which reattaches each rank via the job's saved ``config.yaml``. FSDP shards
        parameters/gradients/optimizer state across ranks; mushin's results and
        significance analysis are unchanged from a single-GPU run.

        Requires Hydra to save a ``config.yaml`` (with ``trainer`` and ``module``
        keys) in the job's output dir — the same contract as :class:`HydraDDP`.
        Configure it with hydra-zen, e.g. ``strategy=builds(HydraFSDP)`` on a
        ``builds(pl.Trainer, ...)`` config.
        """
```

(The class body is just the docstring — all behavior comes from `_HydraReattachMixin`.)

- [ ] **Step 4: Genericize the validation message**

`_validate_external_world_size` raises a message starting `"DDP world size mismatch: ..."`. Change `"DDP world size mismatch"` to `"distributed world size mismatch"` (it now serves FSDP too). Update any test in `tests/test_lightning_launchers.py` that matches the old string (grep `world size mismatch`).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_lightning_hydra_fsdp.py tests/test_lightning_launchers.py tests/test_lightning_hydra_ddp.py -v`
Expected: PASS (new HydraFSDP tests + the DDP/launcher tests still green, including the updated message match).

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check src/mushin/lightning/launchers.py tests/test_lightning_hydra_fsdp.py tests/test_lightning_launchers.py
uv run ruff format src/mushin/lightning/launchers.py tests/test_lightning_hydra_fsdp.py tests/test_lightning_launchers.py
git add src/mushin/lightning/launchers.py tests/test_lightning_hydra_fsdp.py tests/test_lightning_launchers.py
git commit -m "feat: add HydraFSDP (FSDP sharded training under Hydra multirun)"
```

---

### Task 3: Export `HydraFSDP`

**Files:**
- Modify: `src/mushin/lightning/__init__.py`, `src/mushin/__init__.py`
- Test: `tests/test_lightning_hydra_fsdp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lightning_hydra_fsdp.py`:

```python
def test_hydra_fsdp_exported():
    import mushin
    from mushin import HydraFSDP as A
    from mushin.lightning import HydraFSDP as B

    assert A is B
    assert "HydraFSDP" in mushin.__all__
    assert "HydraFSDP" in mushin.lightning.__all__
```

- [ ] **Step 2: Confirm failure**

Run: `uv run pytest tests/test_lightning_hydra_fsdp.py::test_hydra_fsdp_exported -v`
Expected: FAIL — `cannot import name 'HydraFSDP' from 'mushin.lightning'`.

- [ ] **Step 3: Implement the exports**

In `src/mushin/lightning/__init__.py`: change `from .launchers import HydraDDP` to `from .launchers import HydraDDP, HydraFSDP` and add `"HydraFSDP"` to `__all__`.

In `src/mushin/__init__.py`: add `HydraFSDP` to the `from .lightning import (...)` block and to `__all__` (next to `HydraDDP`).

Note: on this branch (`multinode-ddp`), `mushin/__init__.py` also imports `submitit_slurm_config`/`seed_everything_per_rank` from `.lightning`; keep those. Do not remove existing names.

- [ ] **Step 4: Confirm pass**

Run: `uv run pytest tests/test_lightning_hydra_fsdp.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/lightning/__init__.py src/mushin/__init__.py tests/test_lightning_hydra_fsdp.py
uv run ruff format src/mushin/lightning/__init__.py src/mushin/__init__.py tests/test_lightning_hydra_fsdp.py
git add src/mushin/lightning/__init__.py src/mushin/__init__.py tests/test_lightning_hydra_fsdp.py
git commit -m "feat: export HydraFSDP from mushin and mushin.lightning"
```

---

### Task 4: Gated multi-GPU example + cluster placeholder

**Files:**
- Create: `examples/sharding_fsdp_multirun.py`
- Test: `tests/test_lightning_hydra_fsdp.py`

- [ ] **Step 1: Add the gated placeholder test**

Append to `tests/test_lightning_hydra_fsdp.py`:

```python
@pytest.mark.cluster
def test_hydra_fsdp_multirun_end_to_end():
    # Human-run gate: a real FSDP run under Hydra multirun needs 2+ GPUs, so it
    # cannot run in CI. Deselected by default (addopts excludes `cluster`); run
    # with `pytest -m cluster` on a multi-GPU host. See
    # examples/sharding_fsdp_multirun.py for the actual runnable demo.
    pytest.skip("requires >=2 GPUs; run the example on a multi-GPU host")
```

- [ ] **Step 2: Confirm it is deselected by default / skips under the marker**

Run: `uv run pytest tests/test_lightning_hydra_fsdp.py -v`
Expected: the cluster test is DESELECTED (not collected) by default.
Run: `uv run pytest tests/test_lightning_hydra_fsdp.py -m cluster -v`
Expected: it is SKIPPED with the message (proves the marker wiring).

- [ ] **Step 3: Write the runnable example**

Create `examples/sharding_fsdp_multirun.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""FSDP under Hydra multirun via HydraFSDP (requires >=2 GPUs; not run in CI).

Run a 2-job sweep:
  python examples/sharding_fsdp_multirun.py --multirun +run=a,b

Each Hydra job saves its config.yaml; HydraFSDP re-execs the ranks against THAT
config (not the sweep argv), so both jobs run correctly in one process. The model
is tiny — the point is the launcher wiring, not the workload. mushin's analysis
(compare/significance) is unchanged from a single-GPU run.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from hydra_zen import builds, make_config, store, zen
from torch.utils.data import DataLoader, TensorDataset

from mushin import HydraFSDP


class _TinyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        )

    def train_dataloader(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(256, 8, generator=g)
        y = x.sum(dim=1)
        return DataLoader(TensorDataset(x, y), batch_size=32)

    def training_step(self, batch, _idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x).squeeze(-1), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


TrainerConf = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=2,
    strategy=builds(HydraFSDP),
    max_epochs=1,
    enable_checkpointing=False,
    logger=False,
    populate_full_signature=False,
)
Config = make_config(trainer=TrainerConf, module=builds(_TinyModule))


def _task(trainer: pl.Trainer, module: pl.LightningModule) -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("This demo needs >=2 GPUs (FSDP shards across ranks).")
    trainer.fit(module)


if __name__ == "__main__":
    store(Config, name="config")
    store.add_to_hydra_store()
    zen(_task).hydra_main(config_name="config", config_path=None, version_base="1.1")
```

(If the exact `store`/`zen().hydra_main` wiring needs adjustment to match the installed hydra-zen API, adjust so `python examples/sharding_fsdp_multirun.py --multirun +run=a,b` launches two Hydra jobs; keep the `strategy=builds(HydraFSDP)` and `_task` intent. The script is NOT run in CI — only parsed.)

- [ ] **Step 4: Verify the example parses**

Run: `uv run python -c "import ast; ast.parse(open('examples/sharding_fsdp_multirun.py').read()); print('parses')"`
Expected: `parses`.
Run: `uv run ruff check examples/sharding_fsdp_multirun.py` — fix any import-order issues.

- [ ] **Step 5: Commit**

```bash
git add tests/test_lightning_hydra_fsdp.py examples/sharding_fsdp_multirun.py
git commit -m "test: gated multi-GPU HydraFSDP example + cluster placeholder"
```

---

### Task 5: Docs guide + nav + changelog

**Files:**
- Create: `docs/guides/sharding.md`
- Modify: `mkdocs.yml`
- Create: `changes/+hydra-fsdp.added.md`

- [ ] **Step 1: Write the guide**

Create `docs/guides/sharding.md`:

````markdown
# Sharded training under Hydra (FSDP)

`HydraDDP` is **data-parallel**: it replicates the whole model on each GPU. When a
model does **not** fit on one GPU you need **sharding** — splitting parameters,
gradients, and optimizer state across GPUs — which is Lightning's `FSDPStrategy`.
`HydraFSDP` is the sharded counterpart of `HydraDDP`: it makes FSDP work correctly
under Hydra `--multirun`.

## Why not stock `FSDPStrategy` under `--multirun`?

Lightning's `FSDPStrategy` (like `DDPStrategy`) launches ranks by **re-executing
the script with `sys.argv`**. Inside a Hydra sweep that argv is the *sweep*
command, so child ranks re-run the wrong job. `HydraFSDP` instead re-execs each
rank against the job's saved `config.yaml` (via `mushin.lightning._pl_main`),
exactly as `HydraDDP` does for DDP — so a `--multirun` sweep runs each job
correctly.

## When to shard vs DDP

| Question | Use |
|---|---|
| Model fits on one GPU | `HydraDDP` |
| Model / optimizer state does **not** fit on one GPU | `HydraFSDP` |

## Configure with hydra-zen

```python
import pytorch_lightning as pl
from hydra_zen import builds, make_config

from mushin import HydraFSDP

TrainerConf = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=4,
    strategy=builds(HydraFSDP),
    populate_full_signature=True,
)
Config = make_config(trainer=TrainerConf, module=...)  # your LightningModule config
```

`HydraFSDP` requires Hydra to save a `config.yaml` with `trainer` and `module`
keys in the job dir (the same contract as `HydraDDP`). See
`examples/sharding_fsdp_multirun.py` for a runnable 2-GPU sweep (needs real GPUs,
not run in CI).

## DeepSpeed

DeepSpeed ZeRO is the same idea; you would wrap `DeepSpeedStrategy` the same way.
mushin does not ship a `HydraDeepSpeed` class or the `deepspeed` dependency —
configure `DeepSpeedStrategy` directly if you need it.

## mushin is unchanged

Sharding only changes the Trainer `strategy`. `MetricsCallback`, checkpoints,
`load_experiment`, and the `compare`/significance pipeline are identical — mushin
compares sharded and unsharded runs the same way.
````

- [ ] **Step 2: Wire the nav**

In `mkdocs.yml`, in the `- Guides:` list, add after the `Multi-node training` entry (this branch has it):

```yaml
      - Sharded training: guides/sharding.md
```

If there is no `Multi-node training` entry, add `Sharded training` after `Studies` instead.

- [ ] **Step 3: Changelog fragment**

Create `changes/+hydra-fsdp.added.md`:

```markdown
`HydraFSDP`: a Fully-Sharded Data Parallel strategy that works under Hydra
`--multirun`. Like `HydraDDP`, it reattaches ranks via the job's saved
`config.yaml` instead of re-executing with `sys.argv` (which a sweep would run as
the wrong job), so FSDP sharded training composes with Hydra sweeps. Exported from
`mushin`; see the new "Sharded training under Hydra" guide.
```

- [ ] **Step 4: Verify docs build**

Run: `uv run --group docs mkdocs build --strict && rm -rf site`
Expected: builds with no warnings/errors.

- [ ] **Step 5: Commit**

```bash
git add docs/guides/sharding.md mkdocs.yml changes/+hydra-fsdp.added.md
git commit -m "docs: HydraFSDP sharded-training guide + changelog"
```

---

### Task 6: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: all pass; the `cluster` HydraFSDP test is deselected (as is any other cluster test). No new failures vs. the branch baseline.

- [ ] **Step 2: Lint, format, spelling**

```bash
uv run ruff check .
uv run ruff format --check .
uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes
```
Expected: all clean.

- [ ] **Step 3: Import smoke**

Run: `uv run python -c "from mushin import HydraDDP, HydraFSDP; print(HydraDDP.__name__, HydraFSDP.__name__)"`
Expected: `HydraDDP HydraFSDP`.

- [ ] **Step 4: Docs build (strict)**

Run: `uv run --group docs mkdocs build --strict && rm -rf site`
Expected: clean.

- [ ] **Step 5: Final formatting commit (if needed)**

```bash
git add -A ':!.worktrees'
git commit -m "chore: formatting/lint pass for HydraFSDP" || echo "nothing to commit"
```

---

## Done criteria

- `from mushin import HydraFSDP` works; in both `__all__`s; `issubclass(HydraFSDP, FSDPStrategy)` and `issubclass(HydraFSDP, _HydraReattachMixin)`.
- `HydraDDP` behavior unchanged after the mixin refactor (its existing tests pass).
- `HydraFSDP._configure_launcher` installs `_HydraReattachLauncher` single-node and steps aside under an external launcher.
- Docs guide + gated example + `@pytest.mark.cluster` placeholder present; full suite green (cluster test deselected); ruff/format/codespell/mkdocs `--strict` clean.
- **Not exercised in CI:** the real 2-GPU FSDP `--multirun` — the documented human gate. This branch **rides PR #50's cluster-validation merge gate; do not merge until validated.**
