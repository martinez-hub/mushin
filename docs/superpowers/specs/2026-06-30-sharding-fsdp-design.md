# Sharded training under Hydra (FSDP/DeepSpeed) — Design

**Status:** Approved for planning
**Date:** 2026-06-30
**Issue:** #45 (sibling to #43).

## Context

`HydraDDP` is **data-parallel**: the whole model is replicated per GPU. Models that don't fit on one GPU need **sharding** (FSDP / DeepSpeed ZeRO), which is a different axis. Sharding itself is implemented by **Lightning Strategies** (`FSDPStrategy`, `DeepSpeedStrategy`); Hydra/hydra-zen only configures and sweeps them, and mushin analyzes the results unchanged. mushin ships only `HydraDDP` today, so the answer for a too-big model is "configure Lightning's `FSDPStrategy` directly via hydra-zen" — which works but is undocumented and has one real gap.

**Confirmed gap (investigated, PL 2.6.5):** neither `FSDPStrategy.teardown` nor its parents (`ParallelStrategy`, `Strategy`) destroy the distributed process group. So under consecutive Hydra `--multirun` jobs in **one process**, the process group leaks and the next job's `init_process_group` fails/hangs. `HydraDDP` avoids this for DDP because `_setup_environment()` defensively destroys any leftover group at the *next* job's setup — machinery that FSDP/DeepSpeed runs (which don't use `HydraDDP`) never invoke.

## Goals

1. Ship a small, strategy-agnostic **`DistributedTeardown` Callback** that destroys the process group (and pops leaked PL/DDP env vars) at the end of each job, fixing multirun hygiene for FSDP/DeepSpeed/DDP alike.
2. Document **sharded training configured with hydra-zen** (FSDP primary; DeepSpeed noted), including when to reach for it vs `HydraDDP` and that mushin's results/significance layer is unchanged.

## Non-Goals

- Implementing sharding itself (that is Lightning/PyTorch).
- A `HydraFSDP` strategy subclass — `FSDPStrategy` already spawns its own ranks, so a reattach wrapper adds little; the issue flags it low-value. Out of scope.
- Multi-node sharded orchestration — folds into #43/PR #50's external-launcher seam (`creates_processes_externally`), which is strategy-agnostic.
- Adding a `deepspeed` dependency or testing DeepSpeed (heavy, Linux/GPU-only). Documented as "same pattern," covered by the strategy-agnostic Callback.

## Architecture

### The `DistributedTeardown` Callback

New class in `src/mushin/lightning/callbacks.py` (alongside `MetricsCallback`), exported from `mushin.lightning` and `mushin`:

```python
class DistributedTeardown(Callback):
    """Destroy the torch.distributed process group at the end of each Trainer run
    so consecutive Hydra ``--multirun`` jobs (run in one process) start clean.

    Lightning's FSDP/DeepSpeed strategies do not destroy the process group on
    teardown, so without this the next multirun job's ``init_process_group`` fails.
    ``HydraDDP`` does not need it (it clears any leftover group at the next job's
    setup); use this with ``FSDPStrategy``/``DeepSpeedStrategy`` (or any strategy)
    under ``--multirun``. Idempotent and safe single-process (a no-op when no group
    is initialized)."""

    def teardown(self, trainer, pl_module, stage) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        _teardown()  # pop leaked LOCAL_RANK/NODE_RANK/.../PL_GLOBAL_SEED env vars
```

Design points:
- **Reuses `_teardown()` from `launchers.py`** (imported at module top — siblings, no circular import: `launchers.py` imports only `.._compatibility`). `_teardown()` is the single source of truth for the leaked-env-var list; the Callback adds the process-group destruction that `_teardown()` (on main) does not do.
- Hooks `Callback.teardown(trainer, pl_module, stage)` — called at the end of every `fit`/`validate`/`test`/`predict`, so the group is gone before the next multirun job constructs its Trainer.
- Idempotent: guarded by `dist.is_initialized()`; a no-op in single-process/CPU runs, so it is always safe to include.

### Docs guide

New `docs/guides/sharding.md`, added to `mkdocs.yml` nav under **Guides** (after Studies). Contents:
- **When to shard vs DDP** — does the model fit on one GPU? DDP (`HydraDDP`) replicates; FSDP shards params/grads/optimizer state.
- **FSDP via hydra-zen** — a `builds(pl.Trainer, strategy=builds(FSDPStrategy, sharding_strategy="FULL_SHARD", ...), devices=...)` example.
- **Multirun hygiene** — add `DistributedTeardown()` to the Trainer callbacks for `--multirun`; explain the leaked-process-group gap.
- **DeepSpeed** — short note: same pattern, `builds(DeepSpeedStrategy, ...)`; needs the `deepspeed` package (Linux/GPU); the `DistributedTeardown` callback covers it too.
- **mushin layer unchanged** — results/significance/`load_experiment` are identical; only the Trainer `strategy` differs.

## Data flow

Unchanged. The user builds a hydra-zen `TrainerConfig` whose `strategy` is an FSDP/DeepSpeed strategy and whose `callbacks` include `DistributedTeardown`; training runs sharded; mushin reads `metrics.pt`/checkpoints and runs the same `compare`/significance pipeline.

## Error handling

- `DistributedTeardown.teardown` guards `dist.is_available()`/`dist.is_initialized()`, so it never raises in CPU/single-process runs.
- No new failure modes introduced; the Callback only *removes* leaked global state.

## Testing

Hermetic, no GPU:
- **Config-construction test**: `builds(FSDPStrategy, sharding_strategy="FULL_SHARD")` instantiates via hydra-zen (`instantiate`) to a real `FSDPStrategy` — guards the documented config shape. Skipped cleanly if the installed torch lacks FSDP.
- **Callback unit tests** (`tests/test_lightning_callbacks.py`): monkeypatch `torch.distributed` so `is_initialized()` returns True/False; assert `destroy_process_group` is called exactly when a group is initialized and not otherwise, and that `_teardown()` pops the env vars. No real distributed needed.
- **Export test**: `DistributedTeardown` importable from `mushin` and `mushin.lightning`; present in both `__all__`.
- **Gated real-run example**: an `examples/` script (or a `@pytest.mark.cluster` placeholder) for a 2-GPU FSDP `--multirun`, documented as the human-validated gate (multi-GPU can't run in CI/locally), mirroring #43/PR #50.

## Docs/changelog

- `docs/guides/sharding.md` + `mkdocs.yml` nav entry.
- `changes/+sharding-fsdp.added.md` towncrier fragment.

## Build order

One spec, one plan: `DistributedTeardown` Callback + exports → unit/export tests → config-construction test → docs guide + nav + gated example → changelog → verification. Small, mostly additive; the only runtime-validated piece (a real FSDP run) is a documented human gate.
