# Design: Refocus mushin on the core

**Date:** 2026-07-13
**Status:** Approved (brainstorming) — pending implementation plan
**Branch:** `refocus-core`

## Problem

mushin's stated goal is **boilerplate-free, reproducible ML experiment
workflows**: a researcher writes a `task()` function, sweeps parameters, and
gets results back as a labeled `xarray.Dataset`. That core path is still good
(the 6-line quickstart is honest; it rates ~7/10 on "boilerplate-free").

But over 52 commits the repo has accreted scope that dilutes the identity and
makes the tool *feel* harder to use, even though the simple path still works:

- **Import weight / sprawl.** Periphery — `benchmark/` (~1028 LOC), `llm/` (~448),
  `mcp/` (~509) — is ~45% of the codebase. `src/mushin/__init__.py` **eagerly**
  imports `llm` and all of `benchmark`, so every `import mushin` drags in
  `torchmetrics` and two separate eval "products". The public API is ~25 exports
  across 6 subsystems.
- **Auto-tuning over-engineering.** `src/mushin/_tuning.py` (open PR #60) is
  385 LOC with 15 `raise`s guarding ~15 lines of real work, backed by ~910 lines
  of tests that mostly exercise rejection paths. The find-once-and-pin idea is
  sound but inflated ~10×.
- **Legacy + hierarchy noise.** RAI-robustness jargon bleeds into the core
  (`config_name="rai_workflow"`, `RobustnessCurve`, `epsilon`). Of 4 exported
  "workflow" classes, `BaseWorkflow` is unusable alone and `RobustnessCurve` is
  niche.
- **Platform drift.** Cluster-gated HPC PRs (#50 multi-node DDP, #58 FSDP, #59
  GPU-packing) pull toward an HPC platform, away from the sweep→dataset researcher.

## Goal

Re-focus, not rebuild: keep the core, shrink the surface and the sprawl so mushin
reads as "one thing done well." No functionality that a researcher relies on is
removed; back-compat is preserved via lazy access + deprecation shims.

## Non-goals

- Rewriting or removing `benchmark`/`llm`/`mcp` functionality. They stay in the
  repo; only their *loading* and *prominence* change.
- Splitting into separate PyPI packages (considered and rejected: too much
  multi-repo maintenance for a solo maintainer at 0.x).
- Changing the `task()` staticmethod contract or the Hydra execution model
  (out of scope this round; only the cwd footgun is addressed via docs + a helper).

## Design

### 1. Lazy-load the periphery (light import + full back-compat)

Reconciles the two requirements — `import mushin` should be light, yet existing
imports like `from mushin import compare` must keep working — using a
package-level `__getattr__` (PEP 562).

- **Eager core** (imported directly in `__init__.py`): `MultiRunMetricsWorkflow`,
  `multirun`, `hydra_list`, `Study`, `MetricsCallback`, `HydraDDP`,
  `load_experiment`, `load_from_checkpoint`. None of these pull `torchmetrics`
  or `llm`.
- **Lazy on first access** via `__getattr__(name)`: every `benchmark` export
  (`compare`, `BenchmarkResult`, `Task`, `register_task`, `get_task`,
  `list_tasks`, `*_battery`) and the `llm` submodule. First attribute access
  imports the owning submodule and caches the binding in module globals.
- `__all__` continues to advertise the full set for discoverability and
  tab-completion; only the *import time* moves.
- **Acceptance:** `python -c "import mushin"` imports neither `torchmetrics` nor
  `mushin.llm` (assert via `sys.modules`); `from mushin import compare` and
  `mushin.llm.compare_llms` still resolve and work.

### 2. Strip auto-tuning to the core idea

Reduce `src/mushin/_tuning.py` (385 → ~60 LOC) to find-once-and-pin:

```python
def tune_batch_size(trainer, module, datamodule=None, *,
                    pin_path=None, batch_arg="batch_size", retune=False) -> BatchPin: ...
def tune_learning_rate(trainer, module, datamodule=None, *,
                       pin_path=None, lr_attr="lr", retune=False) -> LRPin: ...
```

Behavior: run Lightning's `Tuner.scale_batch_size` / `lr_find` once, write the
found value to a sidecar YAML (`pin_path`, defaulting under
`trainer.default_root_dir`), set the attribute, and on a later call read the pin
and skip the (stochastic) search. `retune=True` forces a fresh search.

**Dropped** (removes ~half the failure modes): `effective_batch_size`, the
`accumulate_grad_batches` computation, `num_devices`, `safety_margin`,
scale-out clamp, `drift`, dual-`batch_arg`-owner ambiguity, divisibility checks,
the pre-existing-callback rejections, and the multirun-requires-explicit-pin_path
rule.

**Kept guards (~3):** corrupt/invalid pin file, tuner returned no suggestion,
target attribute missing. `BatchPin`/`LRPin` shrink to `{value, pin_path,
reused: bool}`.

Prune `tests/test_tuning.py` / `test_tuning_integration.py` to the surviving
surface. This reshapes the still-open PR #60 before it merges.

### 3. Trim + de-RAI the top-level surface

- Remove `BaseWorkflow` and `RobustnessCurve` from the flat `mushin` namespace
  (`__all__`). Both stay importable from `mushin.workflows`; both remain
  accessible via `__getattr__` for one release with a `DeprecationWarning`
  pointing to the submodule path.
- Change the `run()` defaults `config_name` / `job_name` from `"rai_workflow"`
  to `"mushin_workflow"`.
- Demote `RobustnessCurve` / `epsilon` out of the headline docs and API surface
  (kept in `mushin.workflows` for anyone who needs the robustness-curve helper).
- **Acceptance:** core flat namespace = `{MultiRunMetricsWorkflow, Study,
  multirun, hydra_list, MetricsCallback, HydraDDP, load_experiment,
  load_from_checkpoint}` + lazily-resolved benchmark/llm names; importing the
  demoted names still works but warns.

### 4. Hydra cwd footgun

`task()` runs inside Hydra's per-job working directory, so relative paths in a
task silently resolve against the wrong directory. Add:

- A prominent callout in `docs/quickstart.md` / `docs/concepts.md`.
- A small helper (e.g. `mushin.original_cwd()` wrapping
  `hydra.utils.get_original_cwd()`, with a sensible fallback outside a Hydra run)
  so researchers can resolve paths against the launch directory explicitly.

### 5. HPC branches — keep open, pending cluster validation

The cluster-gated distributed PRs are HPC infrastructure orthogonal to the
sweep→dataset researcher, so they are **not** part of the core's near-term
direction. **Revised decision (2026-07-13):** do *not* close PRs #50 (multi-node
DDP), #58 (FSDP), #59 (GPU-packing) as part of this refocus — they remain open,
CI-green, and gated on real GPU/cluster validation, which closing would not
advance. `HydraDDP` (single-node multi-GPU, already on `main`) stays regardless.
Their long-term home (core vs. an opt-in `mushin[distributed]` extra vs. a
separate package) is a decision to make *after* they are validated on hardware,
not now.

## Sequencing / branches

- **`refocus-core`** (this branch, off `main`): items 1, 3, 4 — package surface +
  cwd helper. Ships as its own PR.
- **Auto-tuning strip** (item 2): applied to the existing `auto-tuning` branch,
  reshaping PR #60 in place (does not belong in `refocus-core`).
- **HPC branches** (item 5): no action — PRs #50/#58/#59 stay open, pending
  cluster validation (revised from the original close/park plan).

## Testing

- New unit test asserting `import mushin` does not import `torchmetrics` / `llm`
  (inspect `sys.modules`).
- Back-compat test: previously-flat names still resolve (with the expected
  `DeprecationWarning` for `BaseWorkflow`/`RobustnessCurve`).
- Rewritten `_tuning` tests covering the ~3 surviving guards + pin/reuse/retune.
- Existing benchmark/llm/workflow tests must still pass unchanged (they import
  the submodules directly, which lazy-loading does not break).

## Risks

- **`__getattr__` correctness:** must cache resolved names and cooperate with
  `__all__`/`dir()`; a typo could make a top-level name unresolvable. Covered by
  the back-compat test.
- **Auto-tuning behavior change is a real feature cut:** anyone depending on the
  auto-accumulation / effective-batch math loses it. Acceptable at 0.x and the
  feature is unreleased (PR #60 not merged); note it in the PR description.
- **Deprecation churn:** removing RAI names is a mild break; mitigated by the
  one-release warning shim.
