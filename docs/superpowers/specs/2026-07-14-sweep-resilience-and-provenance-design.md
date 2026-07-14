# Design: Sweep resilience + provenance capture

**Date:** 2026-07-14
**Status:** Approved (brainstorming) — pending implementation plan
**Branch:** `sweep-resilience`
**Target release:** 0.5.0

## Problem

`MultiRunMetricsWorkflow` runs a Hydra `--multirun` and assembles the jobs'
returned metrics into a labeled `xarray.Dataset`. Two reliability gaps hurt the
core "sweep → dataset → compare" loop:

1. **No failure tolerance / no resume.** Metrics are collected in memory from each
   job's `return_value` (`jobs_post_process`: `[j.return_value for j in self.jobs]`),
   and Hydra's `JobReturn.return_value` **re-raises** the exception for a `FAILED`
   job. So a single failed job crashes the whole sweep at collection time, and a
   died 100-job sweep leaves nothing to recover — you re-run everything.
2. **No provenance capture.** Nothing records the git SHA, environment, or resolved
   config per run, so "reproducible" is aspirational, not automatic.

## Goals

- A sweep survives individual job failures **when explicitly asked to**, and you
  can **resume** a partially-completed sweep, re-running only what failed/missing.
- Statistics can **never** be computed on incomplete data — you fix, resume, then
  compare.
- Every run automatically records enough provenance to reproduce it.

## Non-goals

- Not changing the default failure behavior to silently tolerate errors (see
  "Design rationale" — resilient-by-default was considered and rejected).
- Not a distributed/queue scheduler; resume is single-machine, re-launching via the
  existing Hydra launcher.
- Not experiment tracking / a dashboard (off-mission).

## Design rationale (why opt-in, not resilient-by-default)

`xarray` reductions **silently skip NaN** (`skipna=True` for floats): a failed job
→ NaN row means `ds["acc"].mean("seed")` averages that method over *fewer* seeds
than the others, with no warning, and `compare(...)` then runs significance tests
on unequal sample sizes — a silent scientific-integrity bug. Since mushin's entire
pitch is *trustworthy, reproducible* results, the default must fail **loudly**.
**Resume** solves the actual pain (don't lose a long sweep) *honestly*: crash → fix
→ re-run only the failures. Fail-soft (`on_error="nan"`) is the opt-in escape hatch
for when you *expect* some configs to fail and want the partial grid to inspect.

## Shared mechanic: per-job metrics sidecar + sweep manifest

Both features build on persisting results to disk (today they live only in the
in-memory `return_value`, so a died process recovers nothing).

- **Per-job metrics sidecar.** mushin wraps the task (via the existing
  `task_fn_wrapper` seam) so that after the user's `task` returns its `dict`, the
  dict is written to `mushin_metrics.json` in that job's working dir. A job dir
  **with** the sidecar = completed; **without** = failed/never-run. (The existing
  explicit `metrics_filename=`/`tr.save(...)` path is preserved for custom cases;
  the sidecar is the new default source.)
- **Sweep manifest.** In the sweep's `working_dir` root, mushin maintains
  `mushin_sweep_manifest.json` mapping each **requested grid combo** to its status:

  ```json
  {
    "schema": 1,
    "params": ["lr", "seed"],
    "cells": {
      "lr=0.1,seed=0": {"dir": "0", "status": "completed"},
      "lr=0.1,seed=2": {"dir": "5", "status": "failed",
                         "error": "CUDA out of memory"}
    }
  }
  ```

  The manifest is the single source of truth for resume and for the completeness
  signal. A combo key is the canonical string of its swept `(param=value)` pairs,
  derived from each job dir's `.hydra/config.yaml` (already read by
  `load_from_dir`).

## Feature 1: Sweep resilience

### 1a. Fail-soft (`on_error`)

`BaseWorkflow.run` (and subclasses) gains:

```python
on_error: str = "raise"   # "raise" | "nan"
```

- `"raise"` (**default**, backward-compatible): a `FAILED` job re-raises, crashing
  the sweep as today.
- `"nan"`: on a `FAILED` job, record it and continue. The realized behavior:
  - `jobs_post_process` branches on `j.status` instead of touching `j.return_value`
    unconditionally: `COMPLETED` → read the sidecar/`return_value`; `FAILED` →
    append to `self.failures` (`{overrides, exception, working_dir}`) and mark that
    combo `failed` in the manifest.
  - A **loud** `warnings.warn` names every failed combo and the count.
  - `to_xarray` NaN-fills the failed combos (see §3) and stamps
    `ds.attrs["mushin_failures"]` = the list of failed combos.

### 1b. Statistics refuse on incomplete data

The safety net that makes `on_error="nan"` sound. The completeness signal is the
manifest / `wf.failures` / `ds.attrs["mushin_failures"]` — **not** raw NaN (a job
may legitimately return a NaN metric; failed-job ≠ NaN-metric).

- `MultiRunMetricsWorkflow` exposes `is_complete` (no `failed`/`pending` cells) and
  `failures`.
- `benchmark.compare`, `benchmark._stats.compare_methods`, and `Study.run` raise a
  clear, actionable error when handed data with recorded failures:

  > `IncompleteSweepError: 3/12 runs failed (lr=0.1,seed=2; …). Fix the cause and
  > re-run with resume=True to complete the sweep, then compare.`

  Detection: when the input is a mushin dataset, check `ds.attrs["mushin_failures"]`;
  when called from `Study`, check the workflow's `failures`. A plain user-supplied
  dataset with no mushin attrs is treated as complete (unchanged behavior).

### 1c. Resume (full-grid relaunch + per-job short-circuit, config-keyed)

`run(working_dir="prev_sweep", resume=True)` **relaunches the full requested grid**
as a normal Hydra multirun — so Hydra handles the cartesian product *and* the
configured launcher's parallelism natively — but a task wrapper makes already-
completed combos return instantly instead of recomputing. This sidesteps the
"launch an arbitrary subset of a product grid" problem entirely (there is no clean
native Hydra way to launch a non-rectangular subset in one multirun; see Risks).

Mechanism:

1. Parse the requested multirun grid from the `run()` call → the set of requested
   combos (anchors the dataset shape; resume can never add points or duplicate).
2. Load `mushin_sweep_manifest.json` from `working_dir` (or rebuild it by scanning
   job dirs + sidecars if absent, e.g. a pre-manifest sweep).
3. Wrap the task so each job first checks the manifest for its own combo:

   ```python
   def _resume_wrapper(task, manifest):
       def wrapped(cfg):
           combo = _combo_of(cfg)
           if manifest.status(combo) == "completed":
               return _read_sidecar(manifest.dir(combo))  # ms: no training
           return task(cfg)                                # run missing/failed
       return wrapped
   ```

4. Launch the full grid as usual. `completed` combos short-circuit (a near-instant
   process spawn, no training); `failed`/never-run combos actually execute.
5. **Update the manifest in place** as jobs finish: a re-run that succeeds
   overwrites that combo's entry (`status: completed`, new `dir`) — it *replaces*,
   never appends. Combos that fail again stay `failed` (subject to `on_error`).
6. Assemble (§3) over the full requested grid.

Result: same-shaped `xarray` every run; resumed cells filled in place; unresolved
cells remain NaN; no growth, no duplicates. Once the manifest has no
`failed`/`pending` cells, `is_complete` is true and statistics run.

**Overhead:** each already-completed combo still spawns a job process that returns
in milliseconds (a manifest lookup + sidecar read, no training) — negligible next
to training cost. If that spawn overhead ever matters (resuming a handful of
failures out of thousands), the scale-up path is a custom Hydra sweeper that
dispatches only the remaining combos (see Risks, Option B) — a drop-in replacement
for the launch step that leaves the rest of this design unchanged.

## Feature 2: Provenance capture

### 2a. Always-on (minimal)

The task wrapper also writes `mushin_provenance.json` to each job dir:

```json
{
  "git": {"sha": "abc123", "dirty": true, "branch": "main"},
  "timestamp": "2026-07-14T10:00:00Z",
  "python": "3.11.9",
  "packages": {"mushin-py": "0.5.0", "torch": "2.4.1", "numpy": "2.0.0",
               "pytorch-lightning": "2.4.0", "hydra-core": "1.3.2"},
  "config": { ... resolved job config ... },
  "seeds": {"seed": 2}
}
```

- `git.sha`/`dirty`/`branch` via `git` subprocess; **all `None` when not a git
  repo / git absent** — never raises.
- `packages` via `importlib.metadata.version` for a fixed key set; `seeds` and
  `config` from the resolved job config (mushin does not manage seeding — the swept
  `seed` param is captured via the config).
- `to_xarray` aggregates into `ds.attrs["provenance"]` (per-run if they differ,
  else a single sweep-level record) and the workflow exposes `wf.provenance`.

### 2b. Opt-in full env freeze (`capture_env=True`)

`run(..., capture_env=True)` additionally writes, once per sweep, a full dependency
snapshot to `mushin_env.txt` in `working_dir`: prefer `uv export`/`uv pip freeze`;
fall back to a full `importlib.metadata` dump if `uv` is unavailable. Heavier
artifact; off by default.

## 3. Config-keyed assembly (replaces order-based collection)

Today `to_xarray` maps job **order** (row-major) onto the grid, silently assuming a
clean, complete cartesian product — which breaks under failures/resume (a failed
job contributes no metrics → the per-metric lists misalign; resume adds extra
dirs). The new assembly is **keyed by config combo**:

1. Build the full requested grid (cartesian product of the swept params) → fixes the
   dataset dims/coords/shape.
2. For each grid combo, look up its manifest entry: `completed` → load that dir's
   `mushin_metrics.json`; `failed`/missing → NaN for every data var (metric key set
   is the union across completed cells).
3. Build the `xarray.Dataset` from (1)+(2). Dedup is automatic (one current entry
   per combo). Non-multirun params remain singleton dims as today.

Backward-compat: for a clean, fully-completed sweep this yields the identical
dataset the order-based path produced; existing tests should pass unchanged.

## API surface (additions)

- `run(..., on_error="raise", resume=False, capture_env=False)` — three new kwargs.
- `wf.failures: list[dict]`, `wf.is_complete: bool`, `wf.provenance: dict`.
- New exception `IncompleteSweepError` (raised by `compare`/`Study` on incomplete
  data).
- New files per job: `mushin_metrics.json`, `mushin_provenance.json`; per sweep:
  `mushin_sweep_manifest.json`, and (opt-in) `mushin_env.txt`.
- `ds.attrs["provenance"]`, `ds.attrs["mushin_failures"]`.

## Interactions & edge cases

- **Legitimate NaN vs failure:** completeness keys on the manifest, never on NaN, so
  a diverged-loss NaN metric from a *successful* job does not block statistics.
- **`compare` on non-workflow data:** a plain xarray/loader with no mushin attrs is
  treated as complete — unchanged behavior for existing `compare` users.
- **Existing `metrics_filename=`/`tr.save`:** still supported; the auto-sidecar is
  the default source, `metrics_filename=` overrides it.
- **Single (non-multirun) runs:** manifest has one cell; resilience/resume degrade
  gracefully.
- **Resume with a changed grid:** if the `run()` grid differs from the prior sweep,
  only combos present in the new request are assembled; combos in the manifest but
  not requested are ignored (and warned).

## Testing strategy

- Fail-soft: a task that raises for one combo → `on_error="raise"` propagates;
  `on_error="nan"` → that cell is NaN, `wf.failures` records it, warning emitted,
  `ds.attrs["mushin_failures"]` set.
- Incomplete-stats gate: `compare`/`Study.run` on a workflow with failures raises
  `IncompleteSweepError`; on a complete one, runs normally; a plain user dataset is
  unaffected.
- Resume: run a grid with an injected failure → resume with the fix. Assert the
  completed combos **short-circuit** — their `task` body does not re-execute (spy on
  a call counter) and their sidecars are not rewritten — while only the failed combo
  actually runs; the dataset is same-shaped with that cell now filled and
  `is_complete` true. Resume of a fully-complete sweep re-executes no task bodies.
- Config-keyed assembly: out-of-order / duplicate dirs resolve to one value per
  combo; equals the order-based result for a clean sweep (regression).
- Provenance: `mushin_provenance.json` written per job with git/packages/config;
  `git=None` in a non-git temp dir (no raise); `capture_env=True` writes
  `mushin_env.txt`.

## Risks / open questions

- **Resume launch (resolved by the short-circuit design; scale-up path noted).**
  Hydra `--multirun` only sweeps the full cartesian product, and the resume
  `remaining` set is an arbitrary (non-rectangular) subset — there is no clean
  native way to launch it directly. The chosen first cut (§1c, Option C) avoids the
  problem: relaunch the full grid and short-circuit completed combos in the task
  wrapper — native Hydra product + parallelism, trivial to implement, at the cost of
  a millisecond process spawn per already-done combo. **Scale-up path (Option B),
  if that spawn overhead ever matters:** a custom Hydra sweeper that dispatches only
  the remaining combos in one launch. It is a drop-in replacement for the launch
  step and leaves the manifest, assembly, and everything else unchanged — so it can
  be added later without redesign. (A rejected alternative — a loop of one
  single-run `launch()` per remaining combo — is correct but pays a full Hydra init
  per combo and gets no parallelism.)
- **Manifest ↔ dir mapping robustness** across interrupted runs (partial writes).
  Write the manifest atomically (temp + rename); rebuild-from-scan as a fallback.
- **`ds.attrs` serialization** to netCDF (provenance/failures must be
  JSON-stringified for netCDF export, which rejects nested dicts).
- **Behavior change scope:** `on_error` defaults to today's behavior, so no silent
  change; but the assembly refactor (§3) touches the core collection path — guard
  with the regression test above.

## Rollout

Ships as **0.5.0**. Additive kwargs (safe defaults); the new sidecar/manifest files
are written for every sweep but ignored by old code paths. The one internal change
is the config-keyed assembly replacing order-based collection, covered by a
regression test asserting identical output on a clean sweep.
