# HydraFSDP: sharded training under Hydra multirun — Design

**Status:** Approved for planning
**Date:** 2026-07-01
**Issue:** #45. **Branch base:** `multinode-ddp` (PR #50). **DO NOT MERGE until cluster-validated** (rides PR #50's hardware gate).

## Context

`HydraDDP` makes **data-parallel** training work under Hydra `--multirun`: instead of Lightning's stock `_SubprocessScriptLauncher` (which re-execs the script with `sys.argv[1:]` and, in a sweep, spawns child ranks with the *original multirun argv* → wrong/nested jobs), it re-execs `mushin.lightning._pl_main` with the job's **saved `config.yaml`** so each rank reattaches to the correct per-job config.

Sharded training (**FSDP**) has the identical problem: `FSDPStrategy._configure_launcher` also uses `_SubprocessScriptLauncher` (verified), so stock FSDP under Hydra multirun spawns wrong jobs. A previous attempt (closed PR #57) tried a teardown *callback*, but that cannot fix it: Lightning calls `Callback.teardown` **before** `LightningModule.teardown` (so destroying the process group there is too early), and a callback cannot replace the launcher. The correct fix is a **strategy** that mirrors `HydraDDP` — this design.

This is built on the `multinode-ddp` branch to reuse its refactored, launcher-safe machinery (scoped `_teardown`, external-launcher validation), which also resolves the Codex findings from PR #57.

## Goal

Ship `HydraFSDP` — an `FSDPStrategy` subclass that installs mushin's Hydra-aware reattach launcher (saved-`config.yaml`, not `argv`), so FSDP sharded training works correctly under Hydra `--multirun`, exactly as `HydraDDP` does for DDP.

## Non-Goals

- Multi-node sharded orchestration beyond what the shared external-launcher seam already provides (folds into #43).
- Implementing sharding itself (Lightning's `FSDPStrategy` does that).
- DeepSpeed (documented as "same idea," no `HydraDeepSpeed` class, no dependency).

## Key insight

On the `multinode-ddp` branch, `HydraDDP`'s three method overrides are **strategy-agnostic**:
- `setup_environment()`: `_setup_environment()` (destroy leftover group) → `_validate_external_world_size(...)` → `super().setup_environment()`.
- `_configure_launcher()`: install the reattach launcher on the single-node path; step aside (`_rank_0_will_call_children_scripts = True`) when `cluster_environment.creates_processes_externally`.
- `teardown()`: `super().teardown()` → scoped `_teardown()` (clears only `_MUSHIN_SET_ENV` vars).

None of these reference DDP-specific behavior — they use `ParallelStrategy`-level attributes (`num_processes`, `num_nodes`, `cluster_environment`) present on `FSDPStrategy` too. The reattach launcher (`_HydraDDPLauncher`) likewise uses only those attributes.

## Architecture

### Share the overrides via a mixin

Refactor the three overrides into a mixin and have both strategies inherit it (in `src/mushin/lightning/launchers.py`, the live PL≥1.6 branch; the dead PL<1.6 branch is left as-is):

```python
class _HydraReattachMixin:
    """Hydra-aware launcher behavior shared by HydraDDP and HydraFSDP: reattach
    each rank via the saved config.yaml (not sys.argv), fail fast on an external
    world-size mismatch, and scope env-var teardown to what mushin set."""

    def setup_environment(self) -> None:
        _setup_environment()
        _validate_external_world_size(self.num_nodes, self.num_processes, self.cluster_environment)
        super().setup_environment()

    def _configure_launcher(self) -> None:
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _HydraReattachLauncher(
                self.cluster_environment, self.num_processes, self.num_nodes
            )
        else:
            self._rank_0_will_call_children_scripts = True

    def teardown(self) -> None:
        super().teardown()
        _teardown()


class HydraDDP(_HydraReattachMixin, DDPStrategy): ...
class HydraFSDP(_HydraReattachMixin, FSDPStrategy): ...
```

- `_HydraDDPLauncher` is **renamed `_HydraReattachLauncher`** (it is not DDP-specific). Its `_call_children_scripts` (which `_set_env`s MASTER_ADDR/PORT/NODE_RANK/LOCAL_RANK/WORLD_SIZE and re-execs `_pl_main` per rank) is unchanged.
- `HydraDDP` keeps its existing docstring/examples; the mixin carries the logic. `HydraFSDP` gets its own docstring (FSDP-focused, notes it shards; same Hydra `config.yaml` reattach contract).
- MRO: `_HydraReattachMixin` first so its overrides win, then the concrete `FSDPStrategy`/`DDPStrategy` provide `super()`.

### FSDP-specific verifications (at build time)

1. **`_rank_0_will_call_children_scripts`** is assigned (not pre-existing) — confirm `FSDPStrategy`'s external-launcher path *reads* it the way `DDPStrategy` does; if FSDP ignores it, the single-node reattach path (the primary goal) is unaffected, and the external path folds into #43. Document whichever is true.
2. **`FSDPStrategy._configure_launcher`** is cleanly overridable by the mixin (verified it exists).
3. **Reattach correctness**: `_pl_main` reloads `config.yaml` whose `strategy` is `builds(HydraFSDP)`; each rank runs `trainer.fit`, and `FSDPStrategy` shards. No `_pl_main` change needed.
4. **`_validate_external_world_size`** message is genericized ("DDP world size mismatch" → "distributed world size mismatch") since it now serves FSDP.

### Exports

`HydraFSDP` exported from `src/mushin/lightning/__init__.py` and `src/mushin/__init__.py` (both `__all__`), next to `HydraDDP`.

## Data flow

Unchanged from `HydraDDP`. A user configures `builds(pl.Trainer, strategy=builds(HydraFSDP), devices=N, ...)` with hydra-zen; under `--multirun` each job's `config.yaml` is saved; rank 0's `_HydraReattachLauncher` re-execs `_pl_main` per rank with that `config.yaml`; ranks reattach, `FSDPStrategy` shards, training runs; mushin reads `metrics.pt`/checkpoints and runs the same `compare`/significance pipeline.

## Error handling

- External world-size mismatch → the shared `_validate_external_world_size` `RuntimeError` (fail fast).
- Missing `config.yaml` at the reattach path → same behavior as `HydraDDP` today (the launcher reads it from the Hydra output dir).
- No new failure modes beyond `HydraDDP`'s.

## Testing

Hermetic, no GPU (mirroring the existing `HydraDDP` tests):
- **Mixin/launcher unit tests**: `HydraFSDP.setup_environment` calls validation before `super().setup_environment()`; `_configure_launcher` installs `_HydraReattachLauncher` on the single-node path and sets the external flag when `creates_processes_externally` (both via a stub `cluster_environment`); `HydraDDP` still passes its existing tests after the mixin refactor (regression guard).
- **Refactor safety**: rename references (`_HydraDDPLauncher` → `_HydraReattachLauncher`) updated in tests; `HydraDDP` behavior byte-identical.
- **Export test**: `HydraFSDP` importable from `mushin` and `mushin.lightning`; in both `__all__`.
- **Gated multi-GPU end-to-end**: a `@pytest.mark.cluster` placeholder test (the marker exists on this branch) + an `examples/` FSDP-under-multirun script — the human-validated gate (2+ GPUs, can't run in CI), same posture as `HydraDDP`/#43.

## Docs

- Rewrite `docs/guides/sharding.md` (the closed-PR-57 callback version is gone) to document **`HydraFSDP`**: the FSDP-under-Hydra-`--multirun` recipe via `builds(HydraFSDP)`, why stock FSDP re-execs wrongly (the reattach rationale), when to shard vs `HydraDDP`, a DeepSpeed "same idea" note, and that mushin's analysis layer is unchanged. Add to `mkdocs.yml` nav.
- `changes/+hydra-fsdp.added.md` towncrier fragment.
- Update the multi-node guide cross-reference if it mentions sharding.

## Build order

One spec, one plan: rename `_HydraDDPLauncher` → `_HydraReattachLauncher` + extract `_HydraReattachMixin` (HydraDDP regression-safe) → add `HydraFSDP` + genericize the validation message → exports → unit + export tests → gated example + `@pytest.mark.cluster` placeholder → docs guide + nav + changelog → verification. Rides PR #50's **cluster-validation merge gate**.
