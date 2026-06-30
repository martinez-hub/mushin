# Sharded Training under Hydra (FSDP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `DistributedTeardown` Callback that fixes the FSDP/DeepSpeed multirun process-group leak, plus a hydra-zen sharding guide — without implementing sharding itself.

**Architecture:** A small, strategy-agnostic `pytorch_lightning.Callback` whose `teardown` destroys the distributed process group (which `FSDPStrategy`/`DeepSpeedStrategy` do not) and reuses `launchers._teardown()` for env-var hygiene. Plus a docs guide + a gated example. mushin's analysis layer is untouched.

**Tech Stack:** Python 3.10+, pytorch_lightning (Callback, FSDPStrategy), hydra-zen, torch.distributed, pytest, mkdocs, uv.

**Reference (read once before starting):**
- Spec: `docs/superpowers/specs/2026-06-30-sharding-fsdp-design.md`
- The reused helper: `src/mushin/lightning/launchers.py` — `_teardown()` (pops `LOCAL_RANK`/`NODE_RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`/`PL_GLOBAL_SEED`; it does NOT destroy the process group).
- Where the callback lives: `src/mushin/lightning/callbacks.py` (already has `MetricsCallback`; imports `from pytorch_lightning import Callback, LightningModule, Trainer`).
- Exports: `src/mushin/lightning/__init__.py` (`from .callbacks import MetricsCallback` + `__all__`); `src/mushin/__init__.py` (line 22 `from .lightning import HydraDDP, MetricsCallback`, `__all__` at ~32).
- Callback hook signature (verified, PL 2.6.5): `teardown(self, trainer, pl_module, stage) -> None`.
- Test file: `tests/test_lightning_callbacks.py` (exists).
- mkdocs nav Guides block: `mkdocs.yml` (the `- Guides:` list; new entry goes after `Studies`).

**Conventions:**
- Source files start with the existing MIT/FAR copyright header (see `callbacks.py` lines 1-3).
- After edits: `uv run ruff check <paths>` + `uv run ruff format <paths>`. Target is py310 — use modern idioms (`X | None`, not `Optional`).
- Commit messages imperative; **no Claude attribution / no `Co-Authored-By` trailer**.
- Tests via `uv run pytest`.

---

### Task 1: The `DistributedTeardown` Callback + unit tests

Add the callback to `callbacks.py` and test it by monkeypatching `torch.distributed` (no real distributed needed).

**Files:**
- Modify: `src/mushin/lightning/callbacks.py`
- Test: `tests/test_lightning_callbacks.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lightning_callbacks.py`:

```python
def test_distributed_teardown_destroys_group_when_initialized(monkeypatch):
    import torch.distributed as dist

    from mushin.lightning.callbacks import DistributedTeardown

    calls = {"destroy": 0}
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist, "destroy_process_group", lambda: calls.__setitem__("destroy", calls["destroy"] + 1)
    )

    DistributedTeardown().teardown(trainer=None, pl_module=None, stage="fit")
    assert calls["destroy"] == 1


def test_distributed_teardown_noop_when_not_initialized(monkeypatch):
    import torch.distributed as dist

    from mushin.lightning.callbacks import DistributedTeardown

    calls = {"destroy": 0}
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dist, "destroy_process_group", lambda: calls.__setitem__("destroy", calls["destroy"] + 1)
    )

    # must not raise and must not destroy when no group is initialized
    DistributedTeardown().teardown(trainer=None, pl_module=None, stage="fit")
    assert calls["destroy"] == 0


def test_distributed_teardown_pops_leaked_env(monkeypatch):
    import torch.distributed as dist

    from mushin.lightning.callbacks import DistributedTeardown

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("PL_GLOBAL_SEED", "1")

    DistributedTeardown().teardown(trainer=None, pl_module=None, stage="fit")

    import os

    assert "LOCAL_RANK" not in os.environ
    assert "PL_GLOBAL_SEED" not in os.environ
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_lightning_callbacks.py -k distributed_teardown -v`
Expected: FAIL — `cannot import name 'DistributedTeardown'`.

- [ ] **Step 3: Implement the callback**

In `src/mushin/lightning/callbacks.py`, add the import near the top (after the existing imports):

```python
from .launchers import _teardown
```

(No circular import: `launchers.py` imports only stdlib + `.._compatibility`, never `callbacks`.)

Then add the class at the end of the file:

```python
class DistributedTeardown(Callback):
    """Destroy the ``torch.distributed`` process group at the end of each Trainer
    run so consecutive Hydra ``--multirun`` jobs (run in one process) start clean.

    Lightning's ``FSDPStrategy``/``DeepSpeedStrategy`` do not destroy the process
    group on teardown, so without this the next multirun job's
    ``init_process_group`` fails. ``HydraDDP`` does not need it (it clears any
    leftover group at the next job's setup); use this with sharded strategies (or
    any strategy) under ``--multirun``. Idempotent and safe in single-process / CPU
    runs (a no-op when no group is initialized).
    """

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        # Pop leaked LOCAL_RANK/NODE_RANK/.../PL_GLOBAL_SEED so the next multirun
        # job re-initializes from a clean environment.
        _teardown()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_lightning_callbacks.py -v`
Expected: PASS (the 3 new tests + the pre-existing `MetricsCallback` tests).

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/lightning/callbacks.py tests/test_lightning_callbacks.py
uv run ruff format src/mushin/lightning/callbacks.py tests/test_lightning_callbacks.py
git add src/mushin/lightning/callbacks.py tests/test_lightning_callbacks.py
git commit -m "feat: add DistributedTeardown callback for FSDP/DeepSpeed multirun hygiene"
```

---

### Task 2: Export `DistributedTeardown`

Surface the callback from `mushin.lightning` and top-level `mushin`.

**Files:**
- Modify: `src/mushin/lightning/__init__.py`
- Modify: `src/mushin/__init__.py`
- Test: `tests/test_lightning_callbacks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lightning_callbacks.py`:

```python
def test_distributed_teardown_exported():
    import mushin
    from mushin import DistributedTeardown as A
    from mushin.lightning import DistributedTeardown as B

    assert A is B
    assert "DistributedTeardown" in mushin.__all__
    assert "DistributedTeardown" in mushin.lightning.__all__
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_lightning_callbacks.py::test_distributed_teardown_exported -v`
Expected: FAIL — `cannot import name 'DistributedTeardown' from 'mushin.lightning'`.

- [ ] **Step 3: Implement the exports**

Replace `src/mushin/lightning/__init__.py` body with:

```python
from .callbacks import DistributedTeardown, MetricsCallback
from .launchers import HydraDDP

__all__ = ["MetricsCallback", "DistributedTeardown", "HydraDDP"]
```

In `src/mushin/__init__.py`:
- Change the line `from .lightning import HydraDDP, MetricsCallback` to:
  `from .lightning import DistributedTeardown, HydraDDP, MetricsCallback`
- Add `"DistributedTeardown",` to the `__all__` list (next to `"MetricsCallback"`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_lightning_callbacks.py::test_distributed_teardown_exported -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/lightning/__init__.py src/mushin/__init__.py tests/test_lightning_callbacks.py
uv run ruff format src/mushin/lightning/__init__.py src/mushin/__init__.py tests/test_lightning_callbacks.py
git add src/mushin/lightning/__init__.py src/mushin/__init__.py tests/test_lightning_callbacks.py
git commit -m "feat: export DistributedTeardown from mushin and mushin.lightning"
```

---

### Task 3: Config-construction test for FSDP via hydra-zen

Guard the documented config shape: a hydra-zen `builds(FSDPStrategy, ...)` instantiates to a real `FSDPStrategy` (CPU-only; no distributed init).

**Files:**
- Test: `tests/test_lightning_sharding.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_lightning_sharding.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest


def test_fsdp_strategy_builds_via_hydra_zen():
    # FSDP needs torch>=1.12 with the FSDP API; skip cleanly if unavailable.
    FSDPStrategy = pytest.importorskip(
        "pytorch_lightning.strategies"
    ).FSDPStrategy
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
```

- [ ] **Step 2: Run to verify it passes (already-importable construction)**

Run: `uv run pytest tests/test_lightning_sharding.py -v`
Expected: PASS where `FSDPStrategy` is importable (PL 2.x), else SKIP. This is a guard test; it has no production code to add. Confirm it passes on the dev environment.

- [ ] **Step 3: Lint, format, commit**

```bash
uv run ruff check tests/test_lightning_sharding.py
uv run ruff format tests/test_lightning_sharding.py
git add tests/test_lightning_sharding.py
git commit -m "test: FSDPStrategy builds via hydra-zen (config-shape guard)"
```

---

### Task 4: Docs guide + gated example + changelog

Write the sharding guide, wire the nav, add a runnable-but-human-gated FSDP example, and the changelog fragment.

**Files:**
- Create: `docs/guides/sharding.md`
- Modify: `mkdocs.yml`
- Create: `examples/sharding_fsdp_demo.py`
- Create: `changes/+sharding-fsdp.added.md`

- [ ] **Step 1: Write the guide**

Create `docs/guides/sharding.md`:

````markdown
# Sharded training (FSDP / DeepSpeed)

`HydraDDP` is **data-parallel**: it replicates the whole model on each GPU and
splits the batch, so it assumes the model fits on one GPU. When a model does
**not** fit, you need **sharding** — splitting parameters, gradients, and
optimizer state across GPUs. That is a **Lightning Strategy**
(`FSDPStrategy`, `DeepSpeedStrategy`), not something Hydra or mushin implements.
Hydra/hydra-zen configures it; mushin analyzes the results exactly as before.

## When to shard vs DDP

| Question | Use |
|---|---|
| Model fits on one GPU, want more throughput | `HydraDDP` (data-parallel) |
| Model (or its optimizer state) does **not** fit on one GPU | FSDP / DeepSpeed (sharded) |

## FSDP via hydra-zen

`FSDPStrategy` is built into PyTorch — no extra dependency. Configure it like any
other Trainer field with hydra-zen:

```python
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from hydra_zen import builds

from mushin import DistributedTeardown

TrainerConfig = builds(
    pl.Trainer,
    accelerator="gpu",
    devices=4,
    strategy=builds(FSDPStrategy, sharding_strategy="FULL_SHARD"),
    callbacks=[builds(DistributedTeardown)],   # see "Multirun hygiene" below
    populate_full_signature=True,
)
```

## Multirun hygiene

Lightning's `FSDPStrategy`/`DeepSpeedStrategy` do **not** destroy the
`torch.distributed` process group when a Trainer run ends. Under Hydra
`--multirun` (several jobs in one process), the leftover group makes the next
job's `init_process_group` fail. Add mushin's `DistributedTeardown` callback —
it destroys the group (and clears leaked rank/seed env vars) at the end of each
job, so the next one starts clean:

```python
from mushin import DistributedTeardown

trainer = pl.Trainer(..., strategy=FSDPStrategy(...), callbacks=[DistributedTeardown()])
```

It is idempotent and a no-op on CPU / single-process runs, so it is always safe
to include.

## DeepSpeed

DeepSpeed ZeRO follows the same pattern — `builds(DeepSpeedStrategy, stage=3, ...)`
— but needs the `deepspeed` package (Linux + GPU), which mushin does not depend
on. The `DistributedTeardown` callback covers it too.

## mushin is unchanged

Sharding only changes the Trainer `strategy`. Your `MetricsCallback`,
checkpoints, `load_experiment`, and the `compare`/significance pipeline are
identical — mushin compares sharded and unsharded runs the same way.

See `examples/sharding_fsdp_demo.py` for a runnable 2-GPU example (it needs real
GPUs, so it is not run in CI).
````

- [ ] **Step 2: Wire the nav**

In `mkdocs.yml`, in the `- Guides:` list, add this line immediately after the `Studies` entry:

```yaml
      - Sharded training: guides/sharding.md
```

- [ ] **Step 3: Write the gated example**

Create `examples/sharding_fsdp_demo.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Runnable FSDP sharding demo (requires >=2 real GPUs; not run in CI).

Run:  python examples/sharding_fsdp_demo.py

Shows the mushin pieces for sharded training: an FSDP-configured Trainer plus the
``DistributedTeardown`` callback so consecutive runs (e.g. a Hydra ``--multirun``)
leave a clean process group. The model/data are intentionally tiny; the point is
the wiring, not the workload. mushin's analysis layer (compare/significance) is
unchanged from a single-GPU run.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy

from mushin import DistributedTeardown


class _TinyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        )

    def training_step(self, batch, _idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x).squeeze(-1), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("This demo needs >=2 GPUs (FSDP shards across ranks).")

    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 8, generator=g)
    y = x.sum(dim=1)
    loader = DataLoader(TensorDataset(x, y), batch_size=32)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy=FSDPStrategy(sharding_strategy="FULL_SHARD"),
        callbacks=[DistributedTeardown()],
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(_TinyModule(), loader)
    print("FSDP run complete; process group torn down for the next job.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create the changelog fragment**

Create `changes/+sharding-fsdp.added.md`:

```markdown
`DistributedTeardown` callback: destroys the `torch.distributed` process group at
the end of each Trainer run so consecutive Hydra `--multirun` jobs work with
sharded strategies (`FSDPStrategy`/`DeepSpeedStrategy`), which do not clean it up
themselves. New "Sharded training (FSDP / DeepSpeed)" guide shows configuring
sharding via hydra-zen; mushin's analysis layer is unchanged.
```

- [ ] **Step 5: Verify docs build + example imports**

Run: `uv run --group docs mkdocs build --strict && rm -rf site`
Expected: builds with no warnings/errors.

Run: `uv run python -c "import ast; ast.parse(open('examples/sharding_fsdp_demo.py').read()); print('example parses')"`
Expected: `example parses` (it is not executed — it needs GPUs).

- [ ] **Step 6: Commit**

```bash
git add docs/guides/sharding.md mkdocs.yml examples/sharding_fsdp_demo.py changes/+sharding-fsdp.added.md
git commit -m "docs: sharded-training (FSDP) guide + gated example"
```

---

### Task 5: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: all pass; FSDP config test runs (or skips if FSDP unavailable). No new failures vs. `main`.

- [ ] **Step 2: Lint, format check, spelling**

```bash
uv run ruff check .
uv run ruff format --check .
uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes
```
Expected: all clean.

- [ ] **Step 3: Import smoke**

Run: `uv run python -c "from mushin import DistributedTeardown; print(DistributedTeardown.__name__)"`
Expected: `DistributedTeardown`.

- [ ] **Step 4: Docs build (strict)**

Run: `uv run --group docs mkdocs build --strict && rm -rf site`
Expected: clean.

- [ ] **Step 5: Final formatting commit (if needed)**

```bash
git add -A
git commit -m "chore: formatting/lint pass for sharding guide + callback" || echo "nothing to commit"
```

---

## Done criteria

- `from mushin import DistributedTeardown` works; it's in both `__all__`s.
- `DistributedTeardown.teardown` destroys the process group when one is initialized, is a no-op otherwise, and pops the leaked env vars (unit-tested without GPU).
- `builds(FSDPStrategy, ...)` config-construction test passes/skips cleanly.
- `docs/guides/sharding.md` in the nav; `examples/sharding_fsdp_demo.py` parses; changelog fragment present.
- Full suite green; ruff/format/codespell/mkdocs `--strict` clean.

The only piece not exercised in CI is a real multi-GPU FSDP run — that's the documented human gate (the example script), same posture as #43/PR #50.
